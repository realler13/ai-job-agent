"""
Structured two-stage job matching facade.

Uses schemas and prompts from two_stage_matcher.py but routes through
the existing call_llm() infrastructure (which handles Gemini/OpenAI
fallback chains and rate limiting).
"""

import json
import logging
from datetime import date
from typing import Optional, Callable

import jsonschema
import pandas as pd

from .config import TRUNCATION_JOB_DESC_FILTER
from .gpt_filter import call_llm
from .jd_sanitizer import clean_jd
from .two_stage_matcher import (
    SCHEMA_CANDIDATE_HIGHLIGHTS,
    SCHEMA_JOB_EXTRACTION,
    SCHEMA_JOB_DECISION,
    SYSTEM_PROMPT_CANDIDATE,
    USER_PROMPT_CANDIDATE_TEMPLATE,
    SYSTEM_PROMPT_EXTRACTION,
    USER_PROMPT_EXTRACTION_TEMPLATE,
    SYSTEM_PROMPT_DECISION,
    USER_PROMPT_DECISION_TEMPLATE,
    parse_json_strict,
)

logger = logging.getLogger(__name__)


def _call_llm_with_schema(system_prompt: str, user_prompt: str, schema: dict,
                           model: str, max_retries: int = 2) -> dict:
    """
    Call LLM synchronously, parse JSON, validate against schema.
    Retries with a repair prompt on failure.
    """
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    last_response = ""

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Repair prompt — keep the original system context so the model still
            # understands the matching task, not just "fix this JSON in a vacuum".
            combined_prompt = (
                f"{system_prompt}\n\n"
                "REPAIR: Your previous response was not valid JSON. Return ONLY valid "
                "JSON matching the schema exactly — no markdown, no commentary, no extra keys.\n\n"
                f"Previous invalid response:\n{last_response}"
            )

        response_text = call_llm(combined_prompt, model=model)
        last_response = response_text

        if not response_text:
            continue

        try:
            data = parse_json_strict(response_text)
            jsonschema.validate(instance=data, schema=schema)
            return data
        except Exception as e:
            logger.debug(f"Validation failed (attempt {attempt+1}): {e}")
            continue

    raise ValueError(f"Failed to generate valid JSON after {max_retries+1} attempts")


class JobMatcher:
    """
    Structured two-stage job matcher.

    Stage 0: Extract candidate highlights (once per session).
    Stage A: Extract structured data from each job description.
    Stage B: Make match decision for each job.
    """

    def __init__(self, profile_text: str, model_name: str,
                 progress_callback: Optional[Callable] = None):
        self.profile_text = profile_text
        self.model_name = model_name
        self.progress_callback = progress_callback
        self._today = date.today().isoformat()

        # Stage 0: Extract candidate highlights once
        self.candidate_highlights = self._extract_candidate_highlights()
        self.candidate_id = self.candidate_highlights.get("candidate_id", "CAND_UNKNOWN")

    def _extract_candidate_highlights(self) -> dict:
        """Stage 0: Extract structured candidate profile."""
        logger.info("Stage 0: Extracting candidate highlights...")
        user_prompt = USER_PROMPT_CANDIDATE_TEMPLATE.format(
            date=self._today,
            profile_text=self.profile_text,
        )
        try:
            return _call_llm_with_schema(
                SYSTEM_PROMPT_CANDIDATE,
                user_prompt,
                SCHEMA_CANDIDATE_HIGHLIGHTS,
                self.model_name,
            )
        except Exception as e:
            logger.error(f"Candidate highlight extraction failed: {e}")
            # Return minimal valid highlights so matching can proceed
            return {
                "candidate_id": "CAND_FALLBACK",
                "primary_role": "OTHER",
                "seniority": "UNKNOWN",
                "location_preference": "UNKNOWN",
                "industries": [],
                "core_strengths": [],
                "keywords": [],
            }

    def _extract_job(self, job_id: str, job_text: str) -> dict:
        """Stage A: Extract structured data from a job description."""
        user_prompt = USER_PROMPT_EXTRACTION_TEMPLATE.format(
            date=self._today,
            job_id=job_id,
            job_text=job_text[:TRUNCATION_JOB_DESC_FILTER],
        )
        return _call_llm_with_schema(
            SYSTEM_PROMPT_EXTRACTION,
            user_prompt,
            SCHEMA_JOB_EXTRACTION,
            self.model_name,
        )

    def _decide_match(self, job_id: str, extraction: dict, job_text: str) -> dict:
        """Stage B: Decide match level for a job."""
        user_prompt = USER_PROMPT_DECISION_TEMPLATE.format(
            date=self._today,
            candidate_id=self.candidate_id,
            job_id=job_id,
            candidate_highlights_json=json.dumps(self.candidate_highlights),
            job_extraction_json=json.dumps(extraction),
            job_text=job_text[:TRUNCATION_JOB_DESC_FILTER],
        )
        return _call_llm_with_schema(
            SYSTEM_PROMPT_DECISION,
            user_prompt,
            SCHEMA_JOB_DECISION,
            self.model_name,
        )

    def match_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run two-stage matching on a DataFrame of jobs.

        Returns the DataFrame with added columns:
        - match_level: HIGH_MATCH/POTENTIAL_MATCH/SOCIAL_TRAP/JUNIOR_WEAK/IRRELEVANT/ERROR
        - confidence: float 0-1
        - reason: str
        - rejection_rule: str
        - gpt_relevance: str (legacy compat)
        - gpt_reasoning: str (legacy compat)
        """
        if df.empty:
            return df

        df = df.copy()
        match_levels = []
        confidences = []
        reasons = []
        rejection_rules = []

        total = len(df)
        for idx, (i, row) in enumerate(df.iterrows()):
            job_id = f"JOB_{idx+1}"
            title = row.get("title", "Unknown")
            company = row.get("company", "Unknown")
            # Untrusted posting text: AI-directed / applicant-instruction
            # sentences stripped, as on the cascade path.
            desc = clean_jd(str(row.get("description", "")))

            if self.progress_callback:
                pct = 0.80 + (idx / max(total, 1)) * 0.10
                self.progress_callback(f"AI Filtering: {title} at {company} ({idx+1}/{total})", pct)

            try:
                # Stage A: Extract
                try:
                    extraction = self._extract_job(job_id, desc)
                except Exception as e:
                    raise RuntimeError(f"Extraction error: {str(e)[:120]}") from e

                # Stage B: Decide
                try:
                    decision = self._decide_match(job_id, extraction, desc)
                except Exception as e:
                    raise RuntimeError(f"Decision error: {str(e)[:120]}") from e

                match_levels.append(decision["match_level"])
                confidences.append(decision["confidence"])
                reasons.append(decision["reason"])
                rejection_rules.append(decision["rejection_rule"])

            except Exception as e:
                logger.warning(f"Matching failed for {title} at {company}: {e}")
                match_levels.append("ERROR")
                confidences.append(0.0)
                reasons.append(f"Error: {e}")
                rejection_rules.append("NONE")

        df["match_level"] = match_levels
        df["confidence"] = confidences
        df["reason"] = reasons
        df["rejection_rule"] = rejection_rules

        # Legacy compatibility columns
        df["gpt_relevance"] = df["match_level"].apply(
            lambda ml: "relevant" if ml in ("HIGH_MATCH", "POTENTIAL_MATCH") else "irrelevant"
        )
        df["gpt_reasoning"] = df["reason"]

        return df
