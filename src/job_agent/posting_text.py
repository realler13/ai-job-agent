"""Posting text for jobs whose source lists them without it.

Glassdoor, ZipRecruiter, ShowbizJobs, Podnews and several other boards give
the title, company and place but not the posting. The judge then grades the
title alone, and the documents for such a job can only be the general ones
(docs_builder's thin-posting path). Glassdoor and ZipRecruiter rows in
particular come in with no text at all, and a Strong job from a small board
can otherwise get its CV without its posting.

So before a tiered job's documents are made (fill_tiered_jobs, right after a
run tiers its jobs) and again inside generate_package (fill_posting_text, for
jobs tiered earlier or added by hand), the text is looked up:

1. Another saved copy of the same job (same title and company, saved in the
   last 45 days) that has its text. No request at all.
2. The posting page with a plain request: the schema.org JobPosting
   description (application/ld+json; ShowbizJobs, Podnews and most boards
   publish one), else the largest block of paragraphs on the page.
3. ZipRecruiter only: its posting pages sit behind Cloudflare. The stealth
   browser the search pages use (ziprecruiter_scraper._fetch_html) loads it,
   and the text is a React Server Components text chunk that the job's
   htmlFullDescription points at ("$48" -> "48:T<hex length>,<html>").

Glassdoor's posting pages refuse even the stealth browser (403): a Glassdoor
job gets text only from another copy.

The text found is posting text: it is data, never instructions. Every prompt
that embeds it already runs it through jd_sanitizer.
"""
from __future__ import annotations

import html as html_lib
import ipaddress
import json
import logging
import math
import re
import socket
import threading
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import CV_THIN_POSTING_CHARS

logger = logging.getLogger(__name__)

PAGE_TIMEOUT_S = 20
MAX_TEXT_CHARS = 20000
SIBLING_DAYS = 45
RUN_CAP = 40              # posting pages read per run at most
RUN_BUDGET_S = 300        # and at most this long
BROWSER_TIMEOUT_S = 150   # one stealth-browser read (two attempts) at most
MAX_REDIRECTS = 5
_REDIRECTS = (301, 302, 303, 307, 308)

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
_BLOCKS = ("p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div", "ul", "ol", "tr",
           "section", "article", "blockquote", "table")
_NEXT_F_RE = re.compile(r"self\.__next_f\.push\((\[.*?\])\)</script>", re.S)


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def is_thin(text) -> bool:
    """Too little posting text to tailor documents to (the documents' own
    threshold, CV_THIN_POSTING_CHARS)."""
    return len(_clean(text).strip()) < CV_THIN_POSTING_CHARS


def host_of(link: str) -> str:
    return (urlparse(_clean(link)).netloc or "").lower()


def html_to_text(fragment: str) -> str:
    """Readable text from posting HTML: one line per block, "- " before list
    items, no runs of blank lines. Entity-escaped HTML (&lt;p&gt;) is decoded
    first, as some JSON-LD descriptions ship it that way."""
    fragment = _clean(fragment)
    if "&lt;" in fragment and "<" not in fragment:
        fragment = html_lib.unescape(fragment)
    soup = BeautifulSoup(fragment, "html.parser")
    for tag in soup(["script", "style", "noscript", "button", "svg"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert_before("- ")
    for el in soup.find_all(_BLOCKS):
        el.insert_after("\n")
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in soup.get_text("").splitlines()]
    out: list[str] = []
    for line in lines:
        if line:
            out.append(line)
        elif out and out[-1]:
            out.append("")
    return "\n".join(out).strip()[:MAX_TEXT_CHARS]


def _jsonld_description(soup: BeautifulSoup) -> str:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else (data.get("@graph") or [data]) if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict) and "JobPosting" in str(item.get("@type")):
                text = html_to_text(_clean(item.get("description")))
                if text:
                    return text
    return ""


def _largest_block(soup: BeautifulSoup) -> str:
    """The longest element that reads like a posting: at least three
    paragraphs or list items, under MAX_TEXT_CHARS (a whole page is not a
    posting)."""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    best, best_len = None, 0
    for el in soup.find_all(["article", "main", "section", "div"]):
        if len(el.find_all(["p", "li"])) < 3:
            continue
        n = len(el.get_text(" ", strip=True))
        if best_len < n < MAX_TEXT_CHARS:
            best, best_len = el, n
    return html_to_text(str(best)) if best is not None else ""


def text_from_html(page_html: str) -> str:
    """Posting text from a posting page: JSON-LD first, then the largest block."""
    if not page_html:
        return ""
    soup = BeautifulSoup(page_html, "html.parser")
    return _jsonld_description(soup) or _largest_block(soup)


def _rsc_stream(page_html: str) -> str:
    """The React Server Components stream a Next.js page carries in its
    self.__next_f.push([1, "..."]) scripts, joined in page order."""
    parts = []
    for m in _NEXT_F_RE.finditer(page_html or ""):
        try:
            arr = json.loads(m.group(1))
        except Exception:
            continue
        if len(arr) > 1 and isinstance(arr[1], str):
            parts.append(arr[1])
    return "".join(parts)


def ziprecruiter_text(page_html: str) -> str:
    """The job's htmlFullDescription. It is either inline or a "$<id>"
    reference to a text chunk "<id>:T<hex byte length>,<html>" elsewhere in
    the stream (the chunk can span several push scripts)."""
    stream = _rsc_stream(page_html)
    m = re.search(r'"htmlFullDescription":"((?:[^"\\]|\\.)*)"', stream)
    if not m:
        return ""
    value = json.loads(f'"{m.group(1)}"')
    ref = re.fullmatch(r"\$([0-9a-f]+)", value)
    if not ref:
        return html_to_text(value)
    chunk = re.search(r"(?:^|\n)" + re.escape(ref.group(1)) + r":T([0-9a-f]+),", stream)
    if not chunk:
        return ""
    size = int(chunk.group(1), 16)
    raw = stream[chunk.end():].encode("utf-8")[:size].decode("utf-8", "ignore")
    return html_to_text(raw)


def _ziprecruiter_page(link: str) -> str:
    """The stealth browser ZipRecruiter's search pages use, on a daemon thread
    abandoned after BROWSER_TIMEOUT_S: the Cloudflare solver's own loop can
    swallow an alarm and wait for hours."""
    from .ziprecruiter_scraper import _fetch_html
    box: dict = {}

    def work():
        try:
            box["body"] = _fetch_html(link, attempts=2) or ""
        except Exception as e:
            box["error"] = e

    worker = threading.Thread(target=work, name="posting-text-zip", daemon=True)
    worker.start()
    worker.join(BROWSER_TIMEOUT_S)
    if worker.is_alive():
        logger.warning(f"Posting text: ZipRecruiter page still loading after {BROWSER_TIMEOUT_S} s, abandoned")
        return ""
    if "error" in box:
        logger.info(f"Posting text: ZipRecruiter page failed: {box['error']}")
        return ""
    return ziprecruiter_text(box.get("body", ""))


def is_public_web_url(link: str) -> bool:
    """An http(s) URL whose host resolves only to public addresses. A posting
    link comes from a job board (or a hand-added job), so it must never make
    this machine fetch its own services (the API, the UI, LangFuse), the local
    network or a private overlay network: every address the host resolves to must be
    globally routable (no loopback, private, link-local / cloud metadata,
    shared 100.64/10, reserved or multicast), IPv4-mapped ones included."""
    try:
        parts = urlparse(_clean(link))
        host, port = parts.hostname, parts.port
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not host:
        return False
    try:
        infos = socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return False
    for info in infos:
        try:
            addr = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            return False
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        if not addr.is_global or addr.is_multicast:
            return False
    return bool(infos)


def _get_public(link: str):
    """GET that follows redirects by hand, checking every hop with
    is_public_web_url. Returns the response, or None when a hop is refused."""
    url = link
    for _ in range(MAX_REDIRECTS + 1):
        if not is_public_web_url(url):
            logger.info(f"Posting text: refused a link that is not a public web address ({host_of(url) or '?'})")
            return None
        resp = requests.get(url, headers=_HEADERS, timeout=PAGE_TIMEOUT_S, allow_redirects=False)
        location = (getattr(resp, "headers", None) or {}).get("Location")
        if resp.status_code in _REDIRECTS and location:
            url = urljoin(url, location)
            continue
        return resp
    logger.info(f"Posting text: more than {MAX_REDIRECTS} redirects from {host_of(link)}")
    return None


def _is_ziprecruiter(host: str) -> bool:
    return host == "ziprecruiter.com" or host.endswith(".ziprecruiter.com")


def fetch_posting_text(link: str) -> str:
    """The posting text on the job's own page, or "" (blocked, gone, no text,
    not a public web address). Never raises."""
    link = _clean(link)
    host = host_of(link)
    if not host:
        return ""
    try:
        if _is_ziprecruiter(host):
            return _ziprecruiter_page(link) if is_public_web_url(link) else ""
        resp = _get_public(link)
        if resp is None:
            return ""
        if resp.status_code != 200 or "Just a moment" in resp.text[:5000]:
            logger.info(f"Posting text: {host} answered {resp.status_code} for a posting page")
            return ""
        return text_from_html(resp.text)
    except Exception as e:
        if type(e).__name__ == "_ScrapeTimeout":   # a phase alarm must reach its phase
            raise
        logger.info(f"Posting text: {host} posting page failed: {type(e).__name__}: {e}")
        return ""


def find_text(job: dict, db=None) -> tuple[str, str]:
    """(text, source): source is "copy" (another saved copy of the job),
    "page" (its posting page) or "" (nothing found)."""
    title, company = _clean(job.get("title")).strip(), _clean(job.get("company")).strip()
    if db is not None and hasattr(db, "find_posting_text") and title and company:
        try:
            text = db.find_posting_text(title, company, exclude_id=_db_id(job.get("id")), days=SIBLING_DAYS)
        except Exception as e:
            logger.warning(f"Posting text: copy lookup failed: {e}")
            text = ""
        if not is_thin(text):
            return text, "copy"
    link = _clean(job.get("job_url")) or _clean(job.get("link"))
    text = fetch_posting_text(link)
    return (text, "page") if not is_thin(text) else ("", "")


def _db_id(value):
    """The scrapes row id, or None: a scraper's own id ("li-1234567890") is
    not one."""
    if isinstance(value, int):
        return value
    text = _clean(value).strip()
    return int(text) if text.isdigit() else None


def _store(db, job: dict, text: str, source: str) -> None:
    if db is None or not hasattr(db, "set_posting_text"):
        return
    try:
        db.set_posting_text(text, source=f"posting_text:{source}", job_id=_db_id(job.get("id")),
                            link=_clean(job.get("job_url")) or _clean(job.get("link")))
    except Exception as e:
        logger.warning(f"Posting text: could not save it for {job.get('title')!r}: {e}")


def fill_posting_text(job: dict, db=None) -> str:
    """Give *job* its posting text when it has none: sets job['description']
    and saves it on the job's row. Returns the source ("copy", "page") or ""."""
    if not is_thin(job.get("description")):
        return ""
    text, source = find_text(job, db)
    if not text:
        logger.info(f"Posting text: none found for {job.get('title')!r} at {job.get('company')!r} "
                    f"({host_of(job.get('link') or job.get('job_url'))})")
        return ""
    job["description"] = text
    job["description_source"] = f"posting_text:{source}"
    _store(db, job, text, source)
    logger.info(f"Posting text: {len(text)} chars from {source} for {job.get('title')!r} at {job.get('company')!r}")
    return source


def fill_tiered_jobs(df, db, log=None, should_stop=None, cap: int = RUN_CAP, budget_s: float = RUN_BUDGET_S) -> int:
    """Fill the posting text of this run's tiered jobs that came without it,
    before their documents are made and before the digest. Updates the
    DataFrame's description column in place and the rows in the database.
    Returns how many got text."""
    log = log or logger
    if df is None or df.empty:
        return 0
    if "description" not in df.columns:
        df["description"] = ""
    start, tried, filled, by_host = time.monotonic(), 0, 0, {}
    for idx, row in df.iterrows():
        if not is_thin(row.get("description")):
            continue
        if (should_stop and should_stop()) or tried >= cap or time.monotonic() - start > budget_s:
            log.info(f"Posting text: stopped after {tried} jobs (cap {cap}, {budget_s:.0f} s budget)")
            break
        tried += 1
        # The DataFrame's "id" is the scraper's own id, not the row's: the row
        # is found by its link.
        job = {k: row.get(k) for k in ("title", "company", "link", "job_url", "description")}
        source = fill_posting_text(job, db)
        host = host_of(job.get("job_url") or job.get("link"))
        by_host.setdefault(host, [0, 0])[0] += 1
        if source:
            df.at[idx, "description"] = job["description"]
            filled += 1
            by_host[host][1] += 1
    if tried:
        detail = ", ".join(f"{h or '?'} {got}/{n}" for h, (n, got) in sorted(by_host.items()))
        log.info(f"Posting text: {filled} of {tried} tiered jobs without text got it ({detail})")
    return filled
