"""LangFuse observability for the AI filtering cascade.

One trace per posting per run (named ``cascade:<board>:<posting_id>``, tagged
with the run date), one span per cascade stage, every model call logged as a
generation (prompt, completion, model, token usage, cost), and each stage's
verdict recorded as a boolean score with the model's stated reason.

Fail-safe by contract: if the langfuse package is missing, the keys are not
configured, or the server is down, every method silently no-ops — tracing must
never break the pipeline. ``finish()`` flushes and shuts the client down
explicitly because pipeline runs are short-lived processes (buffered events
would be lost at exit otherwise).
"""

import hashlib
import logging
import threading
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


def _job_key(job: dict) -> str:
    """Stable per-posting key: the link when present, else title+company."""
    link = str(job.get("link") or job.get("job_url") or "").strip()
    if link:
        return link
    return f"{job.get('title', '')}|{job.get('company', '')}"


def _posting_id(job: dict) -> str:
    return hashlib.md5(_job_key(job).encode()).hexdigest()[:12]


def _board(job: dict) -> str:
    return str(job.get("site") or "unknown").strip().lower() or "unknown"


class CascadeTracer:
    """Per-run LangFuse tracing for the 3-stage cascade. All methods no-op
    when disabled and swallow every exception."""

    def __init__(self):
        self._lf = None
        self._traces: dict = {}   # job_key -> {"root": span, "last": summary}
        self._lock = threading.Lock()
        self.run_date = date.today().isoformat()
        import os
        if os.getenv("JOB_AGENT_DISABLE_LANGFUSE"):
            return  # explicit kill-switch (tests set this — no live clients in CI)
        try:
            from .config import (
                LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
            if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
                return
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from langfuse import Langfuse
            self._lf = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                base_url=LANGFUSE_HOST,
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"LangFuse tracer disabled: {e}")
            self._lf = None

    @property
    def enabled(self) -> bool:
        return self._lf is not None

    # ── internals ────────────────────────────────────────────────────────

    def _root_for(self, job: dict):
        """Get or lazily create the posting's root span (the trace)."""
        key = _job_key(job)
        with self._lock:
            entry = self._traces.get(key)
            if entry:
                return entry["root"]
            from langfuse import propagate_attributes
            with propagate_attributes(tags=[f"run:{self.run_date}"]):
                root = self._lf.start_observation(
                    name=f"cascade:{_board(job)}:{_posting_id(job)}",
                    as_type="span",
                    input={
                        "title": job.get("title", ""),
                        "company": job.get("company", ""),
                        "location": job.get("location", ""),
                        "link": job.get("link") or job.get("job_url") or "",
                    },
                    metadata={"run_date": self.run_date, "board": _board(job)},
                )
            self._traces[key] = {"root": root}
            return root

    def _log_stage(self, job: dict, stage: str, *, prompt: str, completion: str,
                   model: str, usage: Optional[dict], cost: Optional[float],
                   passed: bool, reason: str, verdict: str,
                   batch_meta: Optional[dict] = None):
        root = self._root_for(job)
        meta = {"verdict": verdict}
        if batch_meta:
            meta.update(batch_meta)
        span = root.start_observation(name=stage, as_type="span", metadata=meta)
        gen_kwargs = dict(
            name=f"{stage}-llm",
            as_type="generation",
            model=model,
            input=prompt,
            output=completion,
            metadata=meta,
        )
        if usage:
            gen_kwargs["usage_details"] = usage
        if cost is not None:
            gen_kwargs["cost_details"] = {"total": round(cost, 6)}
        gen = span.start_observation(**gen_kwargs)
        gen.end()
        span.score_trace(
            name=f"{stage}_pass",
            value=1 if passed else 0,
            data_type="BOOLEAN",
            comment=reason[:1000] if reason else None,
        )
        span.update(output={"passed": passed, "verdict": verdict})
        span.end()
        with self._lock:
            self._traces[_job_key(job)]["last"] = {
                "stage": stage, "passed": passed, "verdict": verdict}

    @staticmethod
    def _split_usage(usage: Optional[dict], n: int, i: int) -> Optional[dict]:
        """Even share of a batched call's usage for posting i of n (remainder
        to the first posting, so run totals still sum correctly)."""
        if not usage or n <= 1:
            return usage
        out = {}
        for k, v in usage.items():
            share, rem = divmod(int(v), n)
            out[k] = share + (rem if i == 0 else 0)
        return out

    # ── cascade hooks (all fail-safe) ────────────────────────────────────

    def log_stage1(self, job: dict, *, prompt: str, completion: str, model: str,
                   usage: Optional[dict], cost: Optional[float], passed: bool):
        if not self._lf:
            return
        try:
            self._log_stage(
                job, "stage1-plausibility", prompt=prompt, completion=completion,
                model=model, usage=usage, cost=cost, passed=passed,
                reason=completion, verdict="YES" if passed else "NO",
            )
        except Exception as e:
            logger.debug(f"LangFuse stage1 log failed: {e}")

    def log_stage2_batch(self, jobs: list, *, prompt: str, completion: str,
                         model: str, usage: Optional[dict],
                         cost: Optional[float], results: list):
        """One batched call graded N postings — log the shared generation into
        each posting's trace with an even usage/cost share."""
        if not self._lf:
            return
        try:
            n = len(jobs)
            for i, (job, res) in enumerate(zip(jobs, results)):
                level = str(res.get("match_level", ""))
                passed = level in ("HIGH_MATCH", "POTENTIAL_MATCH", "WORTH_A_LOOK")
                self._log_stage(
                    job, "stage2-grading", prompt=prompt, completion=completion,
                    model=model, usage=self._split_usage(usage, n, i),
                    cost=(cost / n if cost is not None else None),
                    passed=passed, reason=str(res.get("reason", "")), verdict=level,
                    batch_meta={"batch_size": n, "batch_index": i,
                                "usage_note": "even share of one batched call"},
                )
        except Exception as e:
            logger.debug(f"LangFuse stage2 log failed: {e}")

    def log_stage3(self, job: dict, *, prompt: str, completion: str, model: str,
                   usage: Optional[dict], cost: Optional[float], result: dict):
        if not self._lf:
            return
        try:
            level = str(result.get("match_level", ""))
            self._log_stage(
                job, "stage3-judge", prompt=prompt, completion=completion,
                model=model, usage=usage, cost=cost,
                passed=level != "IRRELEVANT",
                reason=str(result.get("reason", "")), verdict=level,
            )
        except Exception as e:
            logger.debug(f"LangFuse stage3 log failed: {e}")

    def finish(self, final_df=None):
        """Write final verdicts, end all root spans, flush, and shut down.

        Must be called at the end of every cascade run — short-lived process,
        buffered events are lost otherwise.
        """
        if not self._lf:
            return
        try:
            finals = {}
            if final_df is not None:
                try:
                    for row in final_df.to_dict("records"):
                        finals[_job_key(row)] = {
                            "match_level": str(row.get("match_level", "")),
                            "tier": row.get("tier"),
                        }
                except Exception:
                    pass
            with self._lock:
                entries = list(self._traces.items())
                self._traces.clear()
            for key, entry in entries:
                try:
                    root = entry["root"]
                    final = finals.get(key) or entry.get("last") or {}
                    root.update(output=final)
                    verdict = final.get("match_level") or final.get("verdict")
                    if verdict:
                        root.score_trace(
                            name="final_verdict", value=str(verdict),
                            data_type="CATEGORICAL",
                            comment=f"tier={final.get('tier')}" if "tier" in final else None,
                        )
                    root.end()
                except Exception:
                    pass
            self._lf.flush()
            self._lf.shutdown()
        except Exception as e:
            logger.warning(f"LangFuse finish failed: {e}")
        finally:
            self._lf = None
