"""Build the B2 bullet pool (cv_pool.json) from the 8 domain base CVs.

The pool is the union of the truth-audited bullets of base_cv_*.docx, each bound
to the role (company/location/date line) it was audited under, with the fact
labels that stop a CV from saying the same thing twice. No bullet text is ever
generated here: every text is a paragraph of a base file.

Run from the project root (or a worktree with the bases symlinked in):

    PYTHONPATH=src .venv/bin/python scripts/build_cv_pool.py
    PYTHONPATH=src .venv/bin/python scripts/build_cv_pool.py --llm-merge --budget 1.00

Outputs
  cv_pool.json                 next to the bases (gitignored — it holds the candidate's CV text)
  results/cv_pool_review.md    a table for a 10-minute review: every fact, its
                               size, three sample ids (gitignored)

Facts come from text similarity first (thresholds calibrated on a real pool,
see cv_pool.py). --llm-merge adds ONE model pass that may only propose more
pairs of bullets that state the same fact; code keeps a pair only when both ids
exist, sit in the same role and never appear together in a base. Pairs kept by
an earlier pass are carried over on a rebuild (they are ids, not text).

Corrections after the review go into cv_pool_overrides.json (ids only):

    {"link": [["<id>", "<id>"]], "unlink": [["<id>", "<id>"]]}

"link" = these two bullets state the same fact (never both in one CV);
"unlink" = they do not, although the build linked them.

Rebuild after any change to a base CV: the pool is refused once its version no
longer matches the base files.
"""
import argparse
import json
import os
import sys

from dotenv import find_dotenv, load_dotenv

# usecwd: a worktree checkout has no dotenv of its own.
load_dotenv(find_dotenv(usecwd=True))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from job_agent import cv_pool  # noqa: E402
from job_agent.config import CV_B2_SELECTION_MODEL, CV_POOL_PATH, DOMAIN_CV_MAP  # noqa: E402

# $ per MTok for the budget guard. Deliberately the HIGHER published sonnet-5
# rate (cascade_filter books $2/$10): a guard should over-count, not under-count.
RATE_IN, RATE_OUT = 3.0, 15.0
MAX_OUTPUT_TOKENS = 16384        # call_llm's cap for the thinking models
MERGE_TIMEOUT_SEC = 420
# Low effort on purpose. At "medium" the model spent the whole 16K output
# budget thinking through the pairs and returned no answer at all (2026-10-03,
# $0.28 for nothing); the answer itself is a short list of ids.
MERGE_EFFORT = "low"


def _preflight(base_dir: str) -> None:
    missing = [f for f in DOMAIN_CV_MAP.values() if not os.path.exists(os.path.join(base_dir, f))]
    if missing:
        raise SystemExit(
            f"ABORT: base CVs missing under {base_dir}: {missing}. In a worktree, symlink "
            f"them in from the main checkout (they are gitignored).")


def _load_overrides(path: str):
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    links = [(a, b, "override") for a, b in data.get("link", [])]
    unlinks = [(a, b) for a, b in data.get("unlink", [])]
    return links, unlinks


def _carried_model_links(out_path: str):
    """Model-proposed links kept by an earlier build (ids only). Pairs whose
    bullets no longer exist are refused by build_pool."""
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        return [], None, None
    links = [(a, b, source) for role in old.get("roles", [])
             for a, b, source in role.get("extra_links", []) if str(source).startswith("model")]
    return links, old.get("pool_version"), (old.get("clustering") or {}).get("model_pass")


def _model_pass(pool: dict, model: str, budget: float):
    """One call. Returns ([(id, id, source)], spend in USD, proposed count)."""
    from job_agent.gpt_filter import call_llm

    prompt = cv_pool.build_merge_prompt(pool)
    worst = len(prompt) / 3 / 1e6 * RATE_IN + MAX_OUTPUT_TOKENS / 1e6 * RATE_OUT
    if worst > budget:
        raise SystemExit(f"ABORT: worst-case cost ${worst:.2f} exceeds --budget ${budget:.2f}")
    usage = {}
    answer = cv_pool._call_with_timeout(
        lambda: call_llm(prompt, model=model, effort=MERGE_EFFORT, usage_out=usage), MERGE_TIMEOUT_SEC)
    tokens_in = usage.get("input_tokens") or len(prompt) / 3
    tokens_out = usage.get("output_tokens") or MAX_OUTPUT_TOKENS   # unknown usage: book the cap
    spend = tokens_in / 1e6 * RATE_IN + tokens_out / 1e6 * RATE_OUT
    pairs = cv_pool.parse_merge_pairs(answer)
    if pairs is None:
        print(f"model pass: answer unusable (not a complete typed JSON) — discarded [${spend:.3f}]")
        return [], spend, 0
    source = f"model:{cv_pool.POOL_MERGE_PROMPT_VERSION}"
    return [(a, b, source) for a, b in pairs], spend, len(pairs)


def _review_table(pool: dict) -> str:
    bullets = sum(len(r["bullets"]) for r in pool["roles"])
    facts = sum(len(r["facts"]) for r in pool["roles"])
    lines = [
        f"# CV pool review — pool {pool['pool_version'][:12]} (built {pool['built_at']})",
        "",
        f"{bullets} bullets, {len(pool['roles'])} roles, {facts} facts. A CV may hold at most one bullet per fact.",
        f"Thresholds: same statement Jaccard >= {pool['clustering']['same_statement_jaccard']}, "
        f"shared fact word-pair overlap >= {pool['clustering']['shared_fact_bigram_overlap']}.",
        "",
        "What to check (10 minutes): (1) a fact whose bullets say DIFFERENT things -> add the pair "
        "under \"unlink\"; (2) two facts that say the SAME thing -> add one pair of their ids under "
        "\"link\" in cv_pool_overrides.json, then re-run scripts/build_cv_pool.py.",
        "",
    ]
    for number, role in enumerate(pool["roles"], start=1):
        text_of = {b["id"]: b["text"] for b in role["bullets"]}
        fact_count = {b["id"]: len(b["facts"]) for b in role["bullets"]}
        lines += [f"## R{number} — {' / '.join(role['titles'])} — {role['header']}",
                  f"{len(role['bullets'])} bullets, {len(role['facts'])} facts", "",
                  "| Fact | Bullets | Sample ids | First sample |", "|---|---|---|---|"]
        for fact in role["facts"]:
            # Samples: the bullets that state ONLY this fact first (sorted is stable).
            sample = sorted(fact["bullets"], key=lambda bid: fact_count[bid])[:3]
            first = text_of[sample[0]]
            first = first if len(first) <= 110 else first[:107] + "..."
            lines.append(f"| {fact['id']} | {len(fact['bullets'])} | {', '.join(sample)} | {first} |")
        compound = [b for b in role["bullets"] if len(b["facts"]) > 1]
        if compound:
            lines += ["", "Compound bullets (carry more than one fact):"]
            lines += [f"- {b['id']} ({', '.join(b['facts'])})" for b in compound]
        if role["extra_links"]:
            lines += ["", "Links added on top of text similarity:"]
            lines += [f"- {a} + {b} ({source})" for a, b, source in role["extra_links"]]
        lines.append("")
    refused = pool["clustering"].get("refused_links") or []
    if refused:
        lines += ["## Proposed links that were refused", ""]
        lines += [f"- {a} + {b} ({source}): {why}" for a, b, source, why in refused]
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build the B2 bullet pool from the domain base CVs.")
    parser.add_argument("--base-dir", default=ROOT, help="directory holding base_cv_*.docx (default: project root)")
    parser.add_argument("--out", default=None, help="pool file (default: <base-dir>/cv_pool.json)")
    parser.add_argument("--review", default=None, help="review table (default: <base-dir>/results/cv_pool_review.md)")
    parser.add_argument("--overrides", default=None, help="default: <base-dir>/cv_pool_overrides.json")
    parser.add_argument("--llm-merge", action="store_true", help="one model pass proposing extra same-fact pairs")
    parser.add_argument("--budget", type=float, default=1.0, help="USD cap for --llm-merge (default 1.00)")
    parser.add_argument("--model", default=CV_B2_SELECTION_MODEL)
    parser.add_argument("--drop-model-links", action="store_true",
                        help="do not carry over links kept by an earlier model pass")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_path = args.out or os.path.join(base_dir, os.path.basename(CV_POOL_PATH))
    review_path = args.review or os.path.join(base_dir, "results", "cv_pool_review.md")
    overrides_path = args.overrides or os.path.join(base_dir, "cv_pool_overrides.json")
    _preflight(base_dir)

    override_links, unlinks = _load_overrides(overrides_path)
    model_links, old_version, old_pass = (([], None, None) if args.drop_model_links
                                          else _carried_model_links(out_path))

    pool = cv_pool.build_pool(base_dir, extra_links=model_links + override_links, unlinks=unlinks)
    if model_links and old_pass:
        pool["clustering"]["model_pass"] = old_pass   # the record of the pass those links came from
    if model_links and old_version != pool["pool_version"] and not args.llm_merge:
        print("NOTE: the base CVs changed since the last model pass — links between unchanged "
              "bullets were carried over; re-run with --llm-merge to cover the new texts.")

    spend = 0.0
    if args.llm_merge:
        proposed, spend, count = _model_pass(pool, args.model, args.budget)
        pool = cv_pool.build_pool(base_dir, extra_links=model_links + proposed + override_links, unlinks=unlinks)
        kept = sum(1 for role in pool["roles"] for link in role["extra_links"] if link[2].startswith("model"))
        pool["clustering"]["model_pass"] = {
            "model": args.model, "prompt_version": cv_pool.POOL_MERGE_PROMPT_VERSION,
            "pairs_proposed": count, "model_links_in_pool": kept, "usd": round(spend, 4)}
        print(f"model pass: {count} pairs proposed, {kept} model links in the pool, ${spend:.3f}")

    cv_pool.save_pool(pool, out_path)
    os.makedirs(os.path.dirname(review_path), exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(_review_table(pool))

    loaded = cv_pool.load_pool(out_path) if os.path.dirname(os.path.abspath(out_path)) == base_dir else None
    bullets = sum(len(r["bullets"]) for r in pool["roles"])
    facts = sum(len(r["facts"]) for r in pool["roles"])
    print(f"pool {pool['pool_version'][:12]}: {bullets} bullets, {len(pool['roles'])} roles, {facts} facts")
    for number, role in enumerate(pool["roles"], start=1):
        print(f"  R{number}: {len(role['bullets']):3d} bullets, {len(role['facts']):2d} facts")
    refused = pool["clustering"].get("refused_links") or []
    if refused:
        print(f"refused links: {len(refused)} (listed in the review table)")
    print(f"wrote {out_path}" + ("" if loaded else "  (not next to the bases: load_pool() will not find them)"))
    print(f"review table: {review_path}")
    if args.llm_merge:
        print(f"LLM spend: ${spend:.3f}")


if __name__ == "__main__":
    main()
