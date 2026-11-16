"""B2 engine: the candidate's truth-audited CV bullets, selected and ordered per job, written word for word.

The 8 domain base CVs were truth-audited line by line. B2 never writes bullet
text. It takes the union of those files' bullets (the "pool"), keeps every
bullet bound to the role it was audited under, and for one posting picks which
bullets to show and in what order. What reaches an employer is therefore always
a sentence that already sits in an audited base, under the same employer and
dates.

Pieces (the content-mode flag in config decides whether any of this runs):

- build_pool()          union of the bases -> cv_pool.json (scripts/build_cv_pool.py)
- load_pool()           the pool, or None when it is missing or the bases changed
- select_for_job()      one small model call: requirement quotes + bullet ids,
                        every id and quote checked in code, per-role fallback to
                        the routed base's own bullets
- select_deterministic  the same output without a model (keyword-ranked)
- verify_membership()   the saved .docx holds only pool bullets, each under its own role
- top_third_coverage()  is the proof of the important requirements where a skim sees it

Roles are keyed by the normalised company/location/date line, never by title:
a title can repeat across roles, and two eras can share an employer label and
differ only by their dates.

Facts. A pool holds far more texts than facts: variants reword the same
bullet, and some bullets are compounds that carry two facts. A CV must not say
the same thing twice, so every bullet carries the fact labels it touches and a
selection may hold at most one bullet per fact. Two bullets share a fact when
they are the same statement reworded (same cluster) or their texts overlap in a
run of words (linked clusters). Two bullets that sit together in one audited
base never share a fact: the bases do not repeat themselves, which also keeps
every base's own bullet set a valid selection.
"""
import hashlib
import json
import logging
import math
import os
import re
import threading
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import (DOMAIN_CV_MAP, CV_POOL_PATH, CV_B2_SELECTION_MODEL,
                     CV_B2_SELECTION_EFFORT, CV_B2_SELECTION_TIMEOUT_SEC,
                     CV_B2_ROLE_FLOOR, CV_B2_MAX_CHARS_RATIO,
                     CV_B2_DETERMINISTIC_FALLBACK, TRUNCATION_JOB_DESC_CV,
                     THIN_DESCRIPTION_THRESHOLD)
from .jd_sanitizer import clean_jd, UNTRUSTED_JD_CLAUSE

logger = logging.getLogger(__name__)

POOL_SCHEMA = 1

# Fact thresholds, calibrated on a full pool (every within-role pair of
# texts) against a hand-labelled set of facts in which compound bullets carry
# every fact they state. The pairs that sit together in a base
# are known-different and were used as hard negatives.
#   - Same statement: Jaccard of content-word sets >= 0.70. Results are
#     identical anywhere in 0.68-0.75; at 0.60 a bullet starts merging with a
#     longer bullet on the same topic and blocks unrelated bullets.
#   - Shared fact (compound bullets): shared word-pairs / the shorter bullet's
#     word-pairs >= 0.20. The highest known-different pair scores 0.154 (two
#     shared word-pairs, typically a number and its unit), so 0.16-0.20 gives
#     no false link; 0.21 already drops true ones. 0.20 keeps a margin above
#     0.154.
# At these values: no false links, and a few true links missed (compounds that
# also reword the shared part). Missing a link risks a repeated fact, a false
# link only narrows the choice, so the residual goes to the optional model pass
# and the maintainer's review (cv_pool_overrides.json), never to a looser threshold.
SAME_STATEMENT_JACCARD = 0.70
SHARED_FACT_BIGRAM_OVERLAP = 0.20

B2_SELECTION_PROMPT_VERSION = "b2-select-v1"
B2_SELECTION_PROMPT = """You are choosing which of a candidate's EXISTING, verified CV bullets to show for one job posting, and in what order. You never write, edit, merge or shorten a bullet. You return bullet ids only.

{untrusted_clause}

<job_posting title="{job_title}" company="{company}">
{jd_text}
</job_posting>

<candidate_roles>
{roles_block}</candidate_roles>

<instructions>
1) REQUIREMENTS: list the 4 to 8 most important requirements of the posting. Each "quote" is copied character for character from the job posting: one contiguous span of 4 to 30 words, no paraphrase, no ellipsis. "importance" is "critical", "important" or "nice".
2) SELECTION: for EACH role choose bullet ids from THAT role's list only, at least "min" and at most "max" of them, together no longer than "max_chars" characters (the CV must stay at its current length).
   - Choose the bullets that best prove the requirements, the critical ones first.
   - ORDER matters. Within a role the bullet that proves the most important requirement comes first, then the next most important. A recruiter reads the first two bullets of a role and skims the rest.
   - Bullets that share a "facts" label say the same thing in different words. Never choose two bullets that share a label.
   - Keep each role readable on its own: keep a bullet that says what the candidate made, for whom and at what scale, unless a stronger bullet you chose already says it.
   - When nothing in a role speaks to the posting, choose the bullets that best show scope and results, at the "min" count.
3) Use only ids that appear in the lists above. Never invent an id.
</instructions>

<output_format>
Return ONLY one JSON object, no prose and no code fences:
{{"requirements": [{{"quote": "<verbatim span>", "importance": "critical"}}], "roles": [{{"role": "R1", "bullets": ["<id>", "<id>"]}}]}}
Include every role exactly once, in the order given.
</output_format>"""

_IMPORTANCE = ("critical", "important", "nice")


class PoolBuildError(RuntimeError):
    """The base files cannot form a pool (wrong layout, roles differ, missing file)."""


class _SelectionFailed(RuntimeError):
    """The model answer cannot be used at all (the caller falls back)."""


# === Normalisation ===

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_SEPARATORS = re.compile(r"\s*[•·|]\s*")
_QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize_text(text) -> str:
    """Whitespace-normalised text: the comparison form of every bullet."""
    return " ".join(str(text or "").split())


def role_key(header_line) -> str:
    """A role's identity: its full company/location/date line, normalised
    (case, dash and separator variants, whitespace). Never the title."""
    line = unicodedata.normalize("NFKC", str(header_line or "")).translate(_DASHES)
    line = _SEPARATORS.sub(" | ", line)
    return " ".join(line.casefold().split())


def bullet_id(key: str, text) -> str:
    """Stable id: first 10 hex of sha1(role key + normalised text)."""
    return hashlib.sha1((key + normalize_text(text)).encode("utf-8")).hexdigest()[:10]


def _bullets_of(entry) -> List[str]:
    description = entry.get("description") if isinstance(entry, dict) else ""
    return [normalize_text(line) for line in str(description or "").split("\n") if line.strip()]


def base_roles(parsed) -> Optional[List[Dict]]:
    """The roles of a parsed CV in the domain-base layout, or None.

    The layout (parse_docx_to_structure convention): entries alternate between
    an empty title placeholder and the company/location/date line that carries
    the bullets. Anything else — an odd entry count, a role without bullets, a
    repeated company/date line — is refused.

    This is a shape check only. A legacy generic base_cv.docx can parse into
    this shape too, so its refusal rests on _selection_context: it is not a
    pooled variant, and its role lines and bullets are not the pool's."""
    experience = parsed.get("experience") if isinstance(parsed, dict) else None
    if not isinstance(experience, list) or not experience or len(experience) % 2:
        return None
    roles, seen = [], set()
    for i in range(0, len(experience), 2):
        title_entry, line_entry = experience[i], experience[i + 1]
        if not isinstance(title_entry, dict) or not isinstance(line_entry, dict):
            return None
        bullets = _bullets_of(line_entry)
        if _bullets_of(title_entry) or not bullets:
            return None
        header = normalize_text(line_entry.get("title"))
        key = role_key(header)
        # A company/location/date line always carries a year.
        if not key or key in seen or not re.search(r"\b(?:19|20)\d\d\b", key):
            return None
        seen.add(key)
        roles.append({"role_key": key, "header": header,
                      "title": normalize_text(title_entry.get("title")),
                      "title_index": i, "entry_index": i + 1, "bullets": bullets})
    return roles


# === Similarity (deterministic fact clustering, keyword ranking) ===

_STOPWORDS = frozenset("""a an the and or of for to in on with by from as at through across into its their each such including
plus up per that this is was were be been it's it than then so while when who which what via under over out off
all any both more most other some own same too very can will just also not no nor only against between during
before after above below about again further once here there where why how""".split())


def _stem(word: str) -> str:
    """Light suffix stripping (plural, -ing, -ed, final e). Crude on purpose:
    the fact thresholds above were calibrated on exactly this."""
    if re.fullmatch(r"\d+[kmb]?", word):
        return word
    for suffix, replacement in (("ies", "y"), ("sses", "ss")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[:-len(suffix)] + replacement
            break
    else:
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            word = word[:-len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def content_tokens(text) -> List[str]:
    """Lower-cased, stemmed content words and numbers, in order ("2M
    viewers" -> 2m, viewer; "1,200+" -> 1200)."""
    lowered = str(text or "").lower().replace("’", "'").replace("'s", "")
    tokens = []
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?[kmb]?|[a-z]+", lowered):
        raw = raw.replace(",", "")
        if raw not in _STOPWORDS:
            tokens.append(_stem(raw))
    return tokens


# Words every posting uses: they say nothing about what the role needs.
_JD_GENERIC = frozenset(_stem(w) for w in """experience experienced year ability able strong skill work working team
role must required require preferred excellent proven knowledge understanding etc you your our we have has are
candidate position job responsibility qualification requirement minimum degree bachelor demonstrated new high
level well within ensure related relevant similar equivalent company opportunity looking seeking ideal join help
day time need want like make use using based across""".split())


def _jd_terms(text) -> List[str]:
    return [t for t in content_tokens(text) if t not in _JD_GENERIC and len(t) > 1]


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _bigrams(tokens: List[str]) -> set:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _bigram_overlap(a: List[str], b: List[str]) -> float:
    ga, gb = _bigrams(a), _bigrams(b)
    return len(ga & gb) / min(len(ga), len(gb)) if ga and gb else 0.0


def _pair(a, b) -> Tuple:
    return (a, b) if a <= b else (b, a)


def _cluster_statements(texts: List[str], together: set) -> Tuple[List[int], set]:
    """Group one role's texts.

    Returns (cluster number per text, linked cluster pairs).
    - Clusters: the same statement reworded — single-link on the content-word
      Jaccard, never joining two texts that sit together in a base (*together*
      holds those index pairs).
    - Links: two clusters whose texts share a run of words (a compound bullet
      and the single-fact bullet it contains)."""
    tokens = [content_tokens(t) for t in texts]
    sets = [set(t) for t in tokens]
    n = len(texts)
    members = {i: {i} for i in range(n)}
    root = list(range(n))

    def find(x):
        while root[x] != x:
            x = root[x]
        return x

    def blocked(a, b):
        return any(_pair(x, y) in together for x in members[a] for y in members[b])

    scored = sorted(((-_jaccard(sets[i], sets[j]), i, j) for i in range(n) for j in range(i + 1, n)))
    for negative, i, j in scored:
        if -negative < SAME_STATEMENT_JACCARD:
            break
        a, b = find(i), find(j)
        if a == b or blocked(a, b):
            continue
        root[b] = a
        members[a] |= members.pop(b)

    numbering = {r: k for k, r in enumerate(sorted(members, key=lambda r: min(members[r])))}
    cluster = [numbering[find(i)] for i in range(n)]
    links = set()
    for i in range(n):
        for j in range(i + 1, n):
            if cluster[i] == cluster[j] or _pair(i, j) in together:
                continue
            if _bigram_overlap(tokens[i], tokens[j]) >= SHARED_FACT_BIGRAM_OVERLAP:
                links.add(_pair(cluster[i], cluster[j]))
    return cluster, links


def _maximal_cliques(nodes: List[int], adjacent: Dict[int, set]) -> List[List[int]]:
    """Bron-Kerbosch with pivoting. A role has at most a few dozen bullets."""
    cliques = []

    def expand(r, p, x):
        if not p and not x:
            cliques.append(sorted(r))
            return
        pivot = max(p | x, key=lambda v: len(adjacent[v] & p))
        for v in sorted(p - adjacent[pivot]):
            expand(r | {v}, p & adjacent[v], x & adjacent[v])
            p = p - {v}
            x = x | {v}

    expand(set(), set(nodes), set())
    return sorted(cliques, key=lambda c: (c[0], len(c), c))


# === Building the pool ===

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_pool_path() -> str:
    return CV_POOL_PATH if os.path.isabs(CV_POOL_PATH) else os.path.join(_project_root(), CV_POOL_PATH)


def compute_pool_version(base_dir: str) -> Optional[str]:
    """Hash of the domain base files (name + bytes). None when one is missing."""
    digest = hashlib.sha1()
    for filename in sorted(DOMAIN_CV_MAP.values()):
        try:
            with open(os.path.join(base_dir, filename), "rb") as f:
                digest.update(filename.encode("utf-8") + b"\0" + f.read() + b"\0")
        except OSError:
            return None
    return digest.hexdigest()


def build_pool(base_dir: str, extra_links: Optional[List] = None,
               unlinks: Optional[List] = None) -> Dict:
    """Build the pool dict from the domain bases in *base_dir*.

    extra_links: [(bullet id, bullet id, source)] — pairs that state the same
    fact although the text similarity missed it (model pass, the maintainer's review).
    unlinks: [(bullet id, bullet id)] — pairs wrongly linked. Both only change
    the fact labels; no bullet text comes from anywhere but the base files.
    A pair that sits together in a base is never linked (it is reported under
    clustering.refused_links instead)."""
    from .cv_tailor import parse_docx_to_structure

    version = compute_pool_version(base_dir)
    if version is None:
        raise PoolBuildError(f"a domain base CV is missing in {base_dir}")

    role_order, roles, variants = None, {}, {}
    for variant, filename in DOMAIN_CV_MAP.items():
        parsed_roles = base_roles(parse_docx_to_structure(os.path.join(base_dir, filename)))
        if parsed_roles is None:
            raise PoolBuildError(f"{filename}: not the title-line / company-line layout")
        keys = [r["role_key"] for r in parsed_roles]
        if role_order is None:
            role_order = keys
        elif keys != role_order:
            raise PoolBuildError(f"{filename}: roles differ from the other bases")
        variants[variant] = {"file": filename, "roles": {}}
        for parsed in parsed_roles:
            role = roles.setdefault(parsed["role_key"], {
                "role_key": parsed["role_key"], "header": parsed["header"],
                "titles": [], "texts": {}})
            if parsed["title"] not in role["titles"]:
                role["titles"].append(parsed["title"])
            for text in parsed["bullets"]:
                role["texts"].setdefault(text, []).append(variant)
            variants[variant]["roles"][parsed["role_key"]] = [
                bullet_id(parsed["role_key"], t) for t in parsed["bullets"]]

    owner = {}
    for key, role in roles.items():
        for text in role["texts"]:
            if owner.setdefault(text, key) != key:
                raise PoolBuildError("one bullet text appears under two roles")

    id_role = {bullet_id(key, text): key for key, role in roles.items() for text in role["texts"]}
    if len(id_role) != sum(len(role["texts"]) for role in roles.values()):
        raise PoolBuildError("bullet id collision")

    refused = []
    extra_by_role, unlink_pairs = {}, set()
    for a, b, source in (extra_links or []):
        if a == b or a not in id_role or b not in id_role:
            refused.append([a, b, source, "unknown id"])
        elif id_role[a] != id_role[b]:
            refused.append([a, b, source, "different roles"])
        else:
            extra_by_role.setdefault(id_role[a], []).append((a, b, source))
    for a, b in (unlinks or []):
        unlink_pairs.add(_pair(a, b))

    out_roles = []
    for number, key in enumerate(role_order, start=1):
        role = roles[key]
        texts = list(role["texts"])
        ids = [bullet_id(key, t) for t in texts]
        index = {bid: i for i, bid in enumerate(ids)}
        together = {_pair(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))
                    if set(role["texts"][texts[i]]) & set(role["texts"][texts[j]])}
        cluster, links = _cluster_statements(texts, together)

        extra_pairs, kept_extra = set(), []
        for a, b, source in extra_by_role.get(key, []):
            pair = _pair(index[a], index[b])
            if pair in together:
                refused.append([a, b, source, "both sit in one base CV"])
                continue
            extra_pairs.add(pair)
            kept_extra.append([a, b, source])

        adjacent = {i: set() for i in range(len(texts))}
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if _pair(i, j) in together or _pair(ids[i], ids[j]) in unlink_pairs:
                    continue
                if (cluster[i] == cluster[j] or _pair(cluster[i], cluster[j]) in links
                        or (i, j) in extra_pairs):
                    adjacent[i].add(j)
                    adjacent[j].add(i)

        facts = [[] for _ in texts]
        fact_rows = []
        for k, clique in enumerate(_maximal_cliques(list(range(len(texts))), adjacent), start=1):
            fact = f"R{number}F{k:02d}"
            fact_rows.append({"id": fact, "bullets": [ids[i] for i in clique]})
            for i in clique:
                facts[i].append(fact)

        out_roles.append({
            "role_key": key, "header": role["header"], "titles": role["titles"],
            "bullets": [{"id": ids[i], "text": texts[i], "cluster": cluster[i], "facts": facts[i],
                         "variants": role["texts"][texts[i]]} for i in range(len(texts))],
            "facts": fact_rows,
            "cluster_links": sorted([a, b] for a, b in links),
            "extra_links": kept_extra,
            "unlinks": sorted([a, b] for a, b in unlink_pairs if a in index and b in index),
        })

    return {
        "schema": POOL_SCHEMA,
        "pool_version": version,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "clustering": {"same_statement_jaccard": SAME_STATEMENT_JACCARD,
                       "shared_fact_bigram_overlap": SHARED_FACT_BIGRAM_OVERLAP,
                       "refused_links": refused},
        "variants": variants,
        "roles": out_roles,
    }


POOL_MERGE_PROMPT_VERSION = "pool-merge-v1"
POOL_MERGE_PROMPT = """You are checking a pool of CV bullets for hidden repeats. The bullets are grouped by role. A CV built from this pool must never say the same thing twice.

Two bullets "state the same fact" when a CV that contained both would repeat itself: the same achievement, responsibility or result described twice, in different words or as one part of a longer bullet. Two bullets that only mention the same employer, project or number while describing DIFFERENT work do not state the same fact.

Bullets that already share a "facts" label are known repeats. Find the repeats that are NOT yet labelled: pairs of bullets in the SAME role that state the same fact but share no label.

<pool>
{roles_block}</pool>

Rules:
- Only pair ids from the same role.
- Include a pair only when a recruiter reading both bullets in one CV would notice the repetition.
- Do not rewrite, correct or comment on any bullet. Return ids only.

Return ONLY one JSON object, no prose and no code fences:
{{"pairs": [["<id>", "<id>"]]}}
An empty list is a valid answer."""


def build_merge_prompt(pool: Dict) -> str:
    """Prompt for the optional one-off model pass that proposes extra fact
    links. The model returns id pairs only; build_pool() decides what is kept."""
    blocks = []
    for number, role in enumerate(pool["roles"], start=1):
        lines = [f'<role ref="R{number}" line="{role["header"]}">']
        for bullet in role["bullets"]:
            lines.append(f'<bullet id="{bullet["id"]}" facts="{" ".join(bullet["facts"])}">{bullet["text"]}</bullet>')
        lines.append("</role>")
        blocks.append("\n".join(lines))
    return POOL_MERGE_PROMPT.format(roles_block="\n".join(blocks) + "\n")


def parse_merge_pairs(answer: str) -> Optional[List[Tuple[str, str]]]:
    """The model pass's answer as id pairs — only when complete and correctly
    typed ({"pairs": [[id, id], ...]}), else None and the pass is discarded."""
    obj = _parse_json_object(answer)
    pairs = obj.get("pairs") if obj is not None else None
    if not isinstance(pairs, list):
        return None
    out = []
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(i, str) and i.strip() for i in pair)):
            return None
        out.append((pair[0].strip(), pair[1].strip()))
    return out


def save_pool(pool: Dict, path: Optional[str] = None) -> str:
    """Write the pool JSON atomically. Returns the path."""
    path = path or default_pool_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


# === Loading ===

class Pool:
    """Read-only view of a pool dict."""

    def __init__(self, data: Dict):
        self.data = data
        self.version = data["pool_version"]
        self.variants = data["variants"]
        self.roles = {r["role_key"]: r for r in data["roles"]}
        self.role_order = [r["role_key"] for r in data["roles"]]
        self._by_id = {}
        self._by_text = {}
        self._role_of_text = {}
        for role in data["roles"]:
            for bullet in role["bullets"]:
                self._by_id[bullet["id"]] = (role["role_key"], bullet)
                self._by_text[(role["role_key"], bullet["text"])] = bullet
                self._role_of_text[bullet["text"]] = role["role_key"]

    def bullet(self, bullet_id_: str) -> Optional[Dict]:
        found = self._by_id.get(bullet_id_)
        return found[1] if found else None

    def role_of(self, bullet_id_: str) -> Optional[str]:
        found = self._by_id.get(bullet_id_)
        return found[0] if found else None

    def find(self, key: str, text) -> Optional[Dict]:
        """The pool bullet of role *key* whose text equals *text* after
        whitespace normalisation, else None."""
        return self._by_text.get((key, normalize_text(text)))

    def role_of_text(self, text) -> Optional[str]:
        return self._role_of_text.get(normalize_text(text))

    def repeated_facts(self, ids: List[str]) -> Dict[str, List[str]]:
        """{fact: [ids]} for every fact that more than one of *ids* states."""
        holders = {}
        for bid in ids:
            bullet = self.bullet(bid)
            for fact in (bullet["facts"] if bullet else []):
                holders.setdefault(fact, []).append(bid)
        return {fact: who for fact, who in holders.items() if len(who) > 1}


def load_pool(path: Optional[str] = None) -> Optional[Pool]:
    """The bullet pool, or None when the file is missing, unreadable, or was
    built from other base files than the ones on disk now (the bases live next
    to the pool file). A stale pool is never used: B2 is refused instead."""
    path = path or default_pool_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.info(f"CV pool not found at {path} — run scripts/build_cv_pool.py")
        return None
    except Exception as e:
        logger.warning(f"CV pool unreadable ({path}): {e}")
        return None
    try:
        if data.get("schema") != POOL_SCHEMA:
            logger.warning("CV pool has another schema — rebuild with scripts/build_cv_pool.py")
            return None
        current = compute_pool_version(os.path.dirname(os.path.abspath(path)))
        if current is None or data.get("pool_version") != current:
            logger.warning("CV pool does not match the base CVs on disk — "
                           "rebuild with scripts/build_cv_pool.py (B2 refused until then)")
            return None
        return Pool(data)
    except Exception as e:
        logger.warning(f"CV pool malformed ({path}): {e}")
        return None


# === Selection ===

def _resolve_variant(base_variant, pool: Pool) -> Optional[str]:
    """Accepts a domain key ("producer_core"), a base file name or its path."""
    name = os.path.basename(str(base_variant or "").strip())
    if name in pool.variants:
        return name
    for variant, info in pool.variants.items():
        if name == info.get("file"):
            return variant
    return None


def _role_floor(position: int, base_count: int) -> int:
    floor = CV_B2_ROLE_FLOOR[0] if position == 0 else CV_B2_ROLE_FLOOR[1]
    return max(1, min(base_count, floor))


def _selection_context(job: Dict, base_parsed: Dict, base_variant, pool: Optional[Pool]) -> Optional[Dict]:
    """Everything a selection needs, or None when B2 must not run for this job."""
    pool = pool or load_pool()
    if pool is None:
        return None
    variant = _resolve_variant(base_variant, pool)
    if variant is None:
        logger.info(f"B2 refused: base {base_variant!r} is not one of the pooled domain bases")
        return None
    roles = base_roles(base_parsed)
    # The old base_cv.docx stops here even when passed under a domain key: its
    # role lines are not the pooled ones.
    if roles is None or [r["role_key"] for r in roles] != pool.role_order:
        logger.warning(f"B2 refused: base {base_variant!r} does not have the pooled role layout")
        return None
    for position, role in enumerate(roles):
        found = [pool.find(role["role_key"], text) for text in role["bullets"]]
        base_ids = [b["id"] if b else None for b in found]
        # The parsed base must be exactly what the pool was built from (guards
        # against a parser change and against the wrong file for this variant).
        # A bullet the pool does not hold stays in the list as None, so it can
        # never be dropped silently.
        if base_ids != pool.variants[variant]["roles"].get(role["role_key"]):
            logger.warning(f"B2 refused: {base_variant!r} bullets differ from the pool for {role['header']!r}")
            return None
        role["base_ids"] = base_ids
        role["max"] = len(base_ids)
        role["floor"] = _role_floor(position, len(base_ids))
        role["ref"] = f"R{position + 1}"

    description = job.get("description") if isinstance(job, dict) else None
    if description is None or isinstance(description, float):
        description = ""
    posting = clean_jd(str(description))[:TRUNCATION_JOB_DESC_CV].strip()
    if len(posting) < THIN_DESCRIPTION_THRESHOLD:
        logger.info("B2 refused: the posting is too thin to select against")
        return None
    base_chars = sum(len(text) for role in roles for text in role["bullets"])
    return {"pool": pool, "variant": variant, "roles": roles, "posting": posting, "job": job,
            "char_limit": int(base_chars * CV_B2_MAX_CHARS_RATIO)}


def _role_error(pool: Pool, role: Dict, ids) -> Optional[str]:
    """Why *ids* cannot be this role's bullets, or None when they can."""
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return "bullets is not a list of ids"
    if len(set(ids)) != len(ids):
        return "an id is repeated"
    for bid in ids:
        if pool.role_of(bid) != role["role_key"]:
            return f"id {bid[:20]!r} is not in this role's pool"
    if not role["floor"] <= len(ids) <= role["max"]:
        return f"{len(ids)} bullets, allowed {role['floor']}-{role['max']}"
    if pool.repeated_facts(ids):
        return "two bullets state the same fact"
    return None


def _assemble(ctx: Dict, picks: Dict[str, Optional[List[str]]], base_parsed: Dict,
              requirements: List[Dict], method: str, role_errors: Dict[str, str]) -> Dict:
    """Validate the picks role by role, enforce the length limit, and build the
    contract dict. A role that fails keeps the routed base's own bullets."""
    pool = ctx["pool"]
    chosen, fallback_roles, trimmed = {}, [], {}
    for role in ctx["roles"]:
        key = role["role_key"]
        ids = picks.get(key)
        error = role_errors.get(key) or ("no selection returned" if ids is None else _role_error(pool, role, ids))
        if error:
            role_errors[key] = error
            fallback_roles.append(key)
            chosen[key] = list(role["base_ids"])
        else:
            chosen[key] = list(ids)

    # Ordering rule, enforced in code: inside a role, the bullets that prove
    # the posting's most important requirements come first (critical, then
    # important, then nice, then the rest). The sort is stable, so the
    # selection's own order decides inside a tier. A role that fell back keeps
    # the base's audited order.
    reordered = []
    for role in ctx["roles"]:
        key = role["role_key"]
        if key in fallback_roles:
            continue
        ordered = sorted(chosen[key], key=lambda i: _proof_tier(requirements, pool.bullet(i)["text"]))
        if ordered != chosen[key]:
            chosen[key] = ordered
            reordered.append(key)

    def chars(ids):
        return sum(len(pool.bullet(i)["text"]) for i in ids)

    # Length: total characters <= the routed base's total + 5%, so B2 never
    # needs a trim pass. Over the limit, the role furthest above its own base
    # loses its LAST bullet (the least important by the ordering rule), down to
    # the floor; then that role goes back to its base bullets.
    while sum(chars(ids) for ids in chosen.values()) > ctx["char_limit"]:
        over = [(chars(chosen[r["role_key"]]) - chars(r["base_ids"]), position, r)
                for position, r in enumerate(ctx["roles"]) if r["role_key"] not in fallback_roles]
        over = [o for o in over if o[0] > 0]
        if not over:
            break
        _, _, role = max(over, key=lambda o: (o[0], o[1]))
        key = role["role_key"]
        if len(chosen[key]) > role["floor"]:
            chosen[key].pop()
            trimmed[key] = trimmed.get(key, 0) + 1
        else:
            chosen[key] = list(role["base_ids"])
            fallback_roles.append(key)
            role_errors[key] = "over the length limit"
            trimmed.pop(key, None)

    # One bullet per fact in the whole CV (facts are per role today; this also
    # holds if a later pool shares a fact across roles).
    repeated = pool.repeated_facts([i for ids in chosen.values() for i in ids])
    if repeated:
        raise _SelectionFailed(f"a fact is repeated across the CV: {sorted(repeated)}")

    by_entry = {role["entry_index"]: role for role in ctx["roles"]}
    experience = []
    for index, entry in enumerate(base_parsed["experience"]):
        new_entry = {k: v for k, v in entry.items() if not str(k).startswith("_")}
        if index in by_entry:
            ids = chosen[by_entry[index]["role_key"]]
            new_entry["description"] = "\n".join(pool.bullet(i)["text"] for i in ids)
        else:
            new_entry["description"] = ""   # title placeholders stay empty
        experience.append(new_entry)

    return {
        "experience": experience,
        "selection": {
            "requirements": requirements,
            "roles": chosen,
            "fallback_roles": fallback_roles,
            "prompt_version": B2_SELECTION_PROMPT_VERSION,
            "pool_version": pool.version,
            "method": method,
            "base_variant": ctx["variant"],
            "role_errors": role_errors,
            "trimmed": trimmed,
            "reordered": reordered,
        },
    }


def _proof_tier(requirements: List[Dict], text) -> int:
    """0 when *text* proves a critical requirement, 1 important, 2 nice,
    3 none (the same lexical test as top_third_report)."""
    tiers = [_IMPORTANCE.index(r.get("importance")) for r in requirements
             if r.get("importance") in _IMPORTANCE
             and (terms := set(_jd_terms(r.get("quote")))) and _proves(terms, text)]
    return min(tiers, default=len(_IMPORTANCE))


def _quote_form(text) -> str:
    """Comparison form for requirement quotes: case, quote/dash variants and
    whitespace are ignored, nothing else."""
    text = unicodedata.normalize("NFKC", str(text or "")).translate(_DASHES).translate(_QUOTE_CHARS)
    return " ".join(text.casefold().split())


def _verified_requirements(raw: List[Dict], posting: str) -> Tuple[List[Dict], int]:
    """Keep only requirements whose quote is a span of the posting. Returns
    (kept, dropped count)."""
    haystack = _quote_form(posting)
    kept, dropped, seen = [], 0, set()
    for item in raw:
        quote = normalize_text(item["quote"]).strip(" \"'“”‘’•*-–—.;:")
        form = _quote_form(quote)
        if len(form) < 8 or len(form.split()) < 2 or form not in haystack or form in seen:
            dropped += 1
            continue
        seen.add(form)
        kept.append({"quote": quote, "importance": item["importance"]})
    return kept, dropped


def _parse_json_object(answer: str) -> Optional[Dict]:
    start = (answer or "").find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(answer[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _typed_answer(answer: str, refs: List[str]) -> Tuple[List[Dict], Dict[str, list]]:
    """The model answer as (requirements, {role ref: ids}) — only when it is
    complete and correctly typed. Anything else raises _SelectionFailed: a
    half-understood answer is never patched into a usable one."""
    obj = _parse_json_object(answer)
    if obj is None:
        raise _SelectionFailed("answer is not a JSON object")
    requirements, roles = obj.get("requirements"), obj.get("roles")
    if not isinstance(requirements, list) or not isinstance(roles, list):
        raise _SelectionFailed("requirements/roles missing or not lists")
    for item in requirements:
        if (not isinstance(item, dict) or not isinstance(item.get("quote"), str)
                or item.get("importance") not in _IMPORTANCE):
            raise _SelectionFailed("a requirement is not {quote: str, importance: critical|important|nice}")
    by_ref = {}
    for item in roles:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise _SelectionFailed("a role entry is not {role: str, bullets: [...]}")
        ref, ids = item["role"].strip(), item.get("bullets")
        if ref not in refs or ref in by_ref:
            raise _SelectionFailed(f"unknown or repeated role {ref[:20]!r}")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise _SelectionFailed("bullets is not a list of id strings")
        by_ref[ref] = [i.strip() for i in ids]
    return requirements, by_ref


def _build_prompt(ctx: Dict) -> str:
    pool = ctx["pool"]
    blocks = []
    for role in ctx["roles"]:
        max_chars = int(sum(len(pool.bullet(i)["text"]) for i in role["base_ids"]) * CV_B2_MAX_CHARS_RATIO)
        lines = [f'<role ref="{role["ref"]}" title="{role["title"]}" line="{role["header"]}" '
                 f'min="{role["floor"]}" max="{role["max"]}" max_chars="{max_chars}">']
        for bullet in pool.roles[role["role_key"]]["bullets"]:
            lines.append(f'<bullet id="{bullet["id"]}" facts="{" ".join(bullet["facts"])}" '
                         f'chars="{len(bullet["text"])}">{bullet["text"]}</bullet>')
        lines.append("</role>")
        blocks.append("\n".join(lines))
    job = ctx["job"]
    return B2_SELECTION_PROMPT.format(
        untrusted_clause=UNTRUSTED_JD_CLAUSE,
        job_title=normalize_text(job.get("title")).replace('"', "'"),
        company=normalize_text(job.get("company")).replace('"', "'"),
        jd_text=ctx["posting"],
        roles_block="\n".join(blocks) + "\n",
    )


def _call_with_timeout(func, timeout_sec: float):
    """Run *func* in a daemon thread with a hard time budget (SIGALRM does not
    work in API worker threads). A timed-out thread is abandoned, never joined."""
    box = {}

    def _target():
        try:
            box["value"] = func()
        except BaseException as e:  # noqa: BLE001 — surfaced into the caller
            box["error"] = e

    thread = threading.Thread(target=_target, daemon=True, name="b2-select")
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"selection call exceeded {timeout_sec}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _select_with_model(ctx: Dict, base_parsed: Dict) -> Dict:
    from .gpt_filter import call_llm

    prompt = _build_prompt(ctx)
    usage = {}
    answer = _call_with_timeout(
        lambda: call_llm(prompt, model=CV_B2_SELECTION_MODEL, effort=CV_B2_SELECTION_EFFORT,
                         usage_out=usage),
        CV_B2_SELECTION_TIMEOUT_SEC)
    if not answer:
        raise _SelectionFailed("empty answer")
    raw_requirements, by_ref = _typed_answer(answer, [r["ref"] for r in ctx["roles"]])
    requirements, dropped = _verified_requirements(raw_requirements, ctx["posting"])
    if not requirements:
        raise _SelectionFailed("no requirement quote could be found in the posting")
    picks = {role["role_key"]: by_ref.get(role["ref"]) for role in ctx["roles"]}
    result = _assemble(ctx, picks, base_parsed, requirements, "model", {})
    result["selection"].update({"model": CV_B2_SELECTION_MODEL, "effort": CV_B2_SELECTION_EFFORT,
                                "usage": usage, "dropped_requirements": dropped})
    return result


def select_for_job(job: Dict, base_parsed: Dict, base_variant: str,
                   pool: Optional[Pool] = None) -> Optional[Dict]:
    """B2 selection for one job.

    Returns {'experience': [...], 'selection': {...}} or None.
    - experience: role dicts in the same order and shape as
      base_parsed['experience']; 'description' = pool bullets joined by
      newline, title placeholders empty.
    - selection: requirements [{quote, importance}] (quotes verified against the
      sanitised posting), roles {role_key: [bullet ids]}, fallback_roles
      [role_key] (roles that kept the base's own bullets), prompt_version,
      pool_version, plus method / base_variant / role_errors / trimmed /
      reordered (roles whose order code changed to put the proof of the most
      important requirements first).

    None means "no B2 for this job" (pool missing or stale, old base layout,
    thin posting, or any error): the caller ships the base bullets. *base_variant*
    is the domain key, the base file name or its path."""
    try:
        ctx = _selection_context(job, base_parsed, base_variant, pool)
        if ctx is None:
            return None
        try:
            return _select_with_model(ctx, base_parsed)
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            logger.warning(f"B2 selection call unusable ({reason})")
            if not CV_B2_DETERMINISTIC_FALLBACK:
                return None
            result = _select_ranked(ctx, base_parsed)
            result["selection"]["model_error"] = reason
            return result
    except Exception as e:
        logger.warning(f"B2 selection failed, base bullets stay: {e!r}")
        return None


# === Deterministic arm (no model) ===

def _pool_idf(pool: Pool) -> Dict[str, float]:
    bullets = [b for role in pool.data["roles"] for b in role["bullets"]]
    df = Counter(t for b in bullets for t in set(content_tokens(b["text"])))
    return {t: math.log(1 + len(bullets) / n) for t, n in df.items()}


def _posting_requirements(posting: str, idf: Dict[str, float], tf: Counter) -> List[Dict]:
    """Requirement quotes without a model: the posting's lines/sentences that
    carry the most of its recurring terms the candidate has bullets for. Top 3
    critical, next 3 important. Every quote is a verbatim span by construction."""
    scored = []
    for match in re.finditer(r"[^\n.!?;]+", posting):
        chunk = match.group().strip(" \t•*-–—:")
        if not 4 <= len(chunk.split()) <= 40:
            continue
        terms = {t for t in _jd_terms(chunk) if t in idf}
        if len(terms) < 2:
            continue
        score = sum(idf[t] * (1 + math.log(tf[t])) for t in terms)
        scored.append((-score, match.start(), chunk))
    requirements, seen = [], set()
    for _, _, chunk in sorted(scored):
        if _quote_form(chunk) in seen:
            continue
        seen.add(_quote_form(chunk))
        requirements.append({"quote": chunk, "importance": "critical" if len(requirements) < 3 else "important"})
        if len(requirements) == 6:
            break
    return requirements


def _select_ranked(ctx: Dict, base_parsed: Dict) -> Dict:
    pool = ctx["pool"]
    idf = _pool_idf(pool)
    tf = Counter(_jd_terms(ctx["posting"]))
    requirements = _posting_requirements(ctx["posting"], idf, tf)
    critical = {t for r in requirements if r["importance"] == "critical" for t in _jd_terms(r["quote"])}
    weight = {t: idf[t] * (1 + math.log(n)) * (2.0 if t in critical else 1.0)
              for t, n in tf.items() if t in idf}

    picks = {}
    for role in ctx["roles"]:
        base_rank = {bid: i for i, bid in enumerate(role["base_ids"])}
        ranked = []
        for order, bullet in enumerate(pool.roles[role["role_key"]]["bullets"]):
            score = sum(weight.get(t, 0.0) for t in set(content_tokens(bullet["text"])))
            # Ties (and a posting with no usable terms) resolve to the base's
            # own bullets in the base's own order.
            ranked.append((-score, base_rank.get(bullet["id"], len(base_rank)), order, bullet))
        ranked = [r[3] for r in sorted(ranked, key=lambda r: r[:3])]
        budget = int(sum(len(pool.bullet(i)["text"]) for i in role["base_ids"]) * CV_B2_MAX_CHARS_RATIO)
        chosen, used = [], 0
        for position, bullet in enumerate(ranked):
            if len(chosen) == role["max"]:
                break
            if pool.repeated_facts(chosen + [bullet["id"]]):
                continue
            # Keep room for the floor: after this bullet, the shortest of the
            # bullets still to come must be able to fill the remaining slots.
            short = sorted(len(b["text"]) for b in ranked[position + 1:])
            reserve = sum(short[:max(0, role["floor"] - len(chosen) - 1)])
            if used + len(bullet["text"]) + reserve > budget:
                continue
            chosen.append(bullet["id"])
            used += len(bullet["text"])
        if len(chosen) < role["floor"]:
            # The greedy pass could not reach the floor inside the budget: the
            # base's own bullets (always a valid set), ranked for this posting.
            rank_of = {b["id"]: n for n, b in enumerate(ranked)}
            chosen = sorted(role["base_ids"], key=rank_of.__getitem__)
        picks[role["role_key"]] = chosen
    return _assemble(ctx, picks, base_parsed, requirements, "deterministic", {})


def select_deterministic(job: Dict, base_parsed: Dict, base_variant: str,
                         pool: Optional[Pool] = None) -> Optional[Dict]:
    """select_for_job's output without a model: each role's pool bullets ranked
    by overlap with the posting's terms. Offline comparison arm, and the
    fallback when the selection call fails."""
    try:
        ctx = _selection_context(job, base_parsed, base_variant, pool)
        return _select_ranked(ctx, base_parsed) if ctx else None
    except Exception as e:
        logger.warning(f"B2 deterministic selection failed, base bullets stay: {e!r}")
        return None


# === Checks on the saved file ===

def verify_membership(docx_path: str, base_parsed: Dict, pool: Optional[Pool] = None) -> Dict:
    """Membership gate on the SAVED file: every Experience bullet must equal
    (after whitespace normalisation) a pool bullet of the SAME role, and the
    role lines must be the base's.

    Returns {'ok': bool, 'violations': [...], 'roles': {role_key: [ids]}}.
    Each violation: {'kind': 'pool_unavailable' | 'layout' | 'not_in_pool' |
    'wrong_role', 'role_key', 'entry_index', 'position', 'text', 'found_in'}.
    entry_index points into base_parsed['experience'] so the caller can put the
    base bullets back for that role. An unverifiable file is never ok."""
    from .cv_tailor import parse_docx_to_structure

    def fail(kind, **details):
        return {"ok": False, "violations": [{"kind": kind, **details}], "roles": {}}

    try:
        pool = pool or load_pool()
        if pool is None:
            return fail("pool_unavailable")
        expected = base_roles(base_parsed)
        saved = base_roles(parse_docx_to_structure(docx_path))
        if expected is None or saved is None:
            return fail("layout", text="the file or the base is not in the title-line / company-line layout")
        if ([(r["role_key"], r["title"]) for r in saved]
                != [(r["role_key"], r["title"]) for r in expected]):
            return fail("layout", text="role lines differ from the base")

        violations, found_ids = [], {}
        for role in saved:
            key = role["role_key"]
            found_ids[key] = []
            for position, text in enumerate(role["bullets"]):
                bullet = pool.find(key, text) if key in pool.roles else None
                if bullet is not None:
                    found_ids[key].append(bullet["id"])
                    continue
                other = pool.role_of_text(text)
                violations.append({
                    "kind": "wrong_role" if other else "not_in_pool",
                    "role_key": key, "entry_index": role["entry_index"],
                    "position": position, "text": text[:120], "found_in": other})
        return {"ok": not violations, "violations": violations, "roles": found_ids}
    except Exception as e:
        logger.warning(f"verify_membership could not check {docx_path}: {e!r}")
        return fail("layout", text=f"check failed: {type(e).__name__}")


def _proves(requirement_terms: set, text) -> bool:
    """Lexical proof: the text shares two content terms with the requirement
    (one, when the requirement has only one)."""
    return len(requirement_terms & set(content_tokens(text))) >= min(2, len(requirement_terms))


def top_third_report(docx_path: str, requirements: List[Dict]) -> Dict:
    """Where the proof of the important requirements sits in the saved file.

    Considered: the 'critical' requirements (when there is none: 'important',
    then all). A requirement is MET when some Experience bullet proves it
    (shared content terms — a deterministic, lexical test, the same for every
    arm). It is in the TOP when the summary proves it, or a proving bullet is
    one of the first two bullets of its role.

    Returns {'coverage', 'considered', 'met', 'top', 'details'}; coverage =
    top / met. No requirement to check -> 0.0 (unchecked is never passed);
    requirements but none met -> 1.0 (nothing to bury — read 'met')."""
    from .cv_tailor import parse_docx_to_structure

    usable = [r for r in (requirements or []) if isinstance(r, dict) and isinstance(r.get("quote"), str)]
    considered = []
    for level in _IMPORTANCE:
        considered = [r for r in usable if r.get("importance") == level]
        if considered:
            break
    considered = considered or usable
    parsed = parse_docx_to_structure(docx_path)
    summary = parsed.get("summary", "")
    roles = [_bullets_of(entry) for entry in parsed.get("experience", [])]
    roles = [bullets for bullets in roles if bullets]

    details, met, top = [], 0, 0
    for requirement in considered:
        terms = set(_jd_terms(requirement["quote"]))
        positions = [position for bullets in roles for position, text in enumerate(bullets)
                     if terms and _proves(terms, text)]
        is_met = bool(positions)
        in_top = is_met and (min(positions) < 2 or _proves(terms, summary))
        met += is_met
        top += in_top
        details.append({"quote": requirement["quote"], "met": is_met, "top": bool(in_top)})
    if not considered or not roles:
        coverage = 0.0
    else:
        coverage = top / met if met else 1.0
    return {"coverage": coverage, "considered": len(considered), "met": met, "top": top, "details": details}


def top_third_coverage(docx_path: str, requirements: List[Dict]) -> float:
    """Share of the met important requirements whose proof a skim reaches
    (see top_third_report)."""
    try:
        return float(top_third_report(docx_path, requirements)["coverage"])
    except Exception as e:
        logger.warning(f"top_third_coverage could not check {docx_path}: {e!r}")
        return 0.0
