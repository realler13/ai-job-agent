import argparse
import json
import os
import sys
import hashlib
import datetime
import random
import asyncio
import aiohttp
import sqlite3
import jsonschema
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# --- Schemas ---
#
# HOME_METRO in the location enums stands for the candidate's home metro;
# which metro that is comes from the candidate profile text at run time.

SCHEMA_CANDIDATE_HIGHLIGHTS = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CandidateHighlights",
  "type": "object",
  "additionalProperties": False,
  "required": ["candidate_id","primary_role","seniority","location_preference","industries","core_strengths","keywords"],
  "properties": {
    "candidate_id": { "type": "string" },
    "primary_role": { "type": "string", "enum": ["PRODUCER","OTHER"] },
    "seniority": { "type": "string", "enum": ["JUNIOR","MID","SENIOR","LEAD","UNKNOWN"] },
    "location_preference": { "type": "string", "enum": ["HOME_METRO","REMOTE_OK","OTHER","UNKNOWN"] },
    "industries": { "type": "array", "items": { "type": "string" }, "maxItems": 10 },
    "core_strengths": { "type": "array", "items": { "type": "string" }, "maxItems": 12 },
    "keywords": { "type": "array", "items": { "type": "string" }, "maxItems": 60 }
  }
}

SCHEMA_JOB_EXTRACTION = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "JobExtraction",
  "type": "object",
  "additionalProperties": False,
  "required": ["job_id","role_type","seniority","location_type","must_haves","nice_to_haves","keywords","trap_flags"],
  "properties": {
    "job_id": { "type": "string" },
    "role_type": { "type": "string", "enum": ["PRODUCER","SOCIAL_MEDIA_MANAGER","GROWTH_MARKETING","CONTENT_STRATEGY_WRITING","OTHER"] },
    "seniority": { "type": "string", "enum": ["JUNIOR","MID","SENIOR","LEAD","UNKNOWN"] },
    "location_type": { "type": "string", "enum": ["HOME_METRO","REMOTE_OK","OTHER_US","INTERNATIONAL","UNKNOWN"] },
    "must_haves": { "type": "array", "items": { "type": "string" }, "maxItems": 15 },
    "nice_to_haves": { "type": "array", "items": { "type": "string" }, "maxItems": 15 },
    "keywords": { "type": "array", "items": { "type": "string" }, "maxItems": 40 },
    "trap_flags": {
      "type": "object",
      "additionalProperties": False,
      "required": ["social_trap","junior_trap","location_mismatch_possible"],
      "properties": {
        "social_trap": { "type": "boolean" },
        "junior_trap": { "type": "boolean" },
        "location_mismatch_possible": { "type": "boolean" }
      }
    }
  }
}

SCHEMA_JOB_DECISION = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "JobDecision",
  "type": "object",
  "additionalProperties": False,
  "required": ["match_level","reason","rejection_rule","confidence"],
  "properties": {
    "match_level": { "type": "string", "enum": ["HIGH_MATCH","POTENTIAL_MATCH","SOCIAL_TRAP","JUNIOR_WEAK","IRRELEVANT"] },
    "reason": { "type": "string", "minLength": 3, "maxLength": 500 },
    "rejection_rule": { "type": "string", "enum": ["SOCIAL_TRAP","JUNIOR_WEAK","LOCATION_INDUSTRY","NONE"] },
    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
  }
}

SCHEMA_RUN_SUMMARY = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RunSummary",
  "type": "object",
  "additionalProperties": False,
  "required": ["counts","highlights"],
  "properties": {
    "counts": {
      "type": "object",
      "additionalProperties": False,
      "required": ["HIGH_MATCH","POTENTIAL_MATCH","SOCIAL_TRAP","JUNIOR_WEAK","IRRELEVANT"],
      "properties": {
        "HIGH_MATCH": { "type": "integer", "minimum": 0 },
        "POTENTIAL_MATCH": { "type": "integer", "minimum": 0 },
        "SOCIAL_TRAP": { "type": "integer", "minimum": 0 },
        "JUNIOR_WEAK": { "type": "integer", "minimum": 0 },
        "IRRELEVANT": { "type": "integer", "minimum": 0 }
      }
    },
    "highlights": { "type": "array", "items": { "type": "string" }, "maxItems": 8 }
  }
}

# --- Prompts ---

SYSTEM_PROMPT_CANDIDATE = """You extract structured candidate highlights for job matching. Output ONLY valid JSON matching the schema exactly. No extra keys. No markdown."""

USER_PROMPT_CANDIDATE_TEMPLATE = """Current Date: {date}

=== CANDIDATE PROFILE ===
{profile_text}

=== INSTRUCTION ===
You MUST extract the following fields exactly:
- candidate_id (generate a string)
- primary_role (enum: PRODUCER, OTHER)
- seniority (enum: JUNIOR, MID, SENIOR, LEAD)
- location_preference (enum: HOME_METRO, REMOTE_OK, OTHER; HOME_METRO = the candidate's home metro)
- industries (list of strings)
- core_strengths (list of strings)
- keywords (list of strings)

Do NOT output a generic resume structure. Follow the schema EXACTLY."""

SYSTEM_PROMPT_EXTRACTION = """You extract structured facts from a job description for downstream matching. Output ONLY valid JSON matching the schema exactly. No extra keys. No markdown.

Classification rules:
\t•\trole_type: PRODUCER, SOCIAL_MEDIA_MANAGER, GROWTH_MARKETING, CONTENT_STRATEGY_WRITING, OTHER
\t•\tseniority: JUNIOR, MID, SENIOR, LEAD, UNKNOWN
\t•\tlocation_type: HOME_METRO, REMOTE_OK, OTHER_US, INTERNATIONAL, UNKNOWN

Trap logic:
\t•\ttrap_flags.social_trap = true if primarily community management, posting schedules, replying to comments, social analytics, growth hacking, performance marketing, or “content strategist” that is basically copy/blog writing.
\t•\ttrap_flags.junior_trap = true if entry-level/junior or “one-person band” (shoot/edit/post) with low seniority.
\t•\ttrap_flags.location_mismatch_possible = true if the job is neither in the candidate's home metro (location_type HOME_METRO) nor explicitly remote-friendly.

Keep lists short (top 5–12). Prefer explicit requirements."""

USER_PROMPT_EXTRACTION_TEMPLATE = """Current Date: {date}
Job ID: {job_id}

=== JOB DESCRIPTION ===
{job_text}

=== INSTRUCTION ===
You MUST extract the following fields exactly:
- job_id (string)
- role_type (enum: PRODUCER, SOCIAL_MEDIA_MANAGER, GROWTH_MARKETING, CONTENT_STRATEGY_WRITING, OTHER)
- seniority (enum: JUNIOR, MID, SENIOR, LEAD, UNKNOWN)
- location_type (enum: HOME_METRO, REMOTE_OK, OTHER_US, INTERNATIONAL, UNKNOWN)
- must_haves (list of strings: top hard requirements)
- nice_to_haves (list of strings: preferred skills)
- keywords (list of strings: specific technologies/tools)
- trap_flags (object with booleans: social_trap, junior_trap, location_mismatch_possible)

Do NOT use keys like 'skills' or 'software'. Follow schema EXACTLY."""

SYSTEM_PROMPT_DECISION = """Act as a strict Executive Recruiter. Determine if this role is a STRATEGIC MATCH for the candidate.

CRITICAL: REJECTION RULES OVERRIDE POTENTIAL MATCHES. If a job violates a rejection rule, it MUST be rejected.

Rejection rules:
\t1.\tSOCIAL MEDIA/GROWTH TRAP (critical reject):
\t•\tThe candidate's profession is the one stated in the candidate highlights; reject roles whose core is social media management or growth marketing.
\t•\tReject if the core role is community management, posting schedules, comment management, pure social analytics/growth/performance marketing, or “content strategist” that is basically copy/blog writing.
\t•\tAccept if producing video content for social platforms or managing production budgets / creative strategy.
\t2.\tLEVEL CHECK:
\t•\tMatch the role's level against the seniority in the candidate highlights.
\t•\tReject entry/junior/one-person-band roles.
\t3.\tLOCATION:
\t•\tMust be the candidate's home metro (enum label HOME_METRO) OR Remote. If clearly neither, reject with LOCATION_INDUSTRY.
\t4.\tCAPABILITY MATCH:
\t•\tThe core strengths in the candidate highlights must map to job needs.

Output ONLY valid JSON matching schema JobDecision exactly. No extra keys. No markdown.
Do not provide step-by-step reasoning."""

USER_PROMPT_DECISION_TEMPLATE = """Current Date: {date}
Candidate ID: {candidate_id}
Job ID: {job_id}

=== CANDIDATE HIGHLIGHTS ===
{candidate_highlights_json}

=== JOB EXTRACTION ===
{job_extraction_json}

=== JOB DESCRIPTION (context) ===
{job_text}

=== INSTRUCTION ===
You MUST determine the match level and return JSON with these EXACT keys:
- match_level (enum: HIGH_MATCH, POTENTIAL_MATCH, SOCIAL_TRAP, JUNIOR_WEAK, IRRELEVANT)
- reason (string: 3-120 chars)
- rejection_rule (enum: SOCIAL_TRAP, JUNIOR_WEAK, LOCATION_INDUSTRY, NONE)
- confidence (float: 0.0 to 1.0)

Do NOT use keys like 'decision' or 'analysis'. Follow schema EXACTLY."""

# --- Helper Functions ---

def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Extracts the first valid {...} object and parses it.

    Uses raw_decode (not find/rfind) so braces inside string values and any
    trailing prose after the object don't break parsing.
    """
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON block found")
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")

async def validate_and_retry(
    provider_func, 
    model: str, 
    system_prompt: str, 
    user_prompt: str, 
    schema: Dict[str, Any],
    max_retries: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Calls the provider, validates against schema, and retries with repair prompt if needed.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Repair prompt
                print(f"DEBUG: Retry attempt {attempt} for JSON fix.")
                repair_prompt = f"Fix to valid JSON exactly per schema. Input:\\n{kwargs.get('last_response_text', '')}"
                messages = [
                    {"role": "system", "content": "Return ONLY valid JSON matching schema. No extra keys."},
                    {"role": "user", "content": repair_prompt}
                ]
            
            response_text = await provider_func(model, messages, **kwargs)
            kwargs['last_response_text'] = response_text # parse for next loop
            
            data = parse_json_strict(response_text)
            jsonschema.validate(instance=data, schema=schema)
            return data
            
        except Exception as e:
            print(f"DEBUG: Validation failed: {e}")
            if 'response_text' in locals():
                print(f"DEBUG: Invalid Response (len={len(response_text)}):\\n{response_text[:500]}...")
            last_error = e
            # continue to retry
            
    raise ValueError(f"Failed to generate valid JSON after retries: {last_error}")

# --- API Clients ---

async def gemini_generate(model: str, messages: List[Dict], **kwargs) -> str:
    """
    Gemini API client (Async).
    """
    api_key = kwargs['api_key']
    url = kwargs['url_template'].format(model=model)
    
    # Convert standard messages to Gemini "contents"
    full_user_text = ""
    for msg in messages:
        if msg['role'] == "system":
            full_user_text += f"<SYSTEM>\\n{msg['content']}\\n</SYSTEM>\\n"
        elif msg['role'] == "user":
            full_user_text += msg['content'] + "\\n"
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": full_user_text}]
        }],
        "generationConfig": {
            "temperature": kwargs.get('temperature', 0.0),
            "maxOutputTokens": kwargs.get('max_output_tokens', 1024)
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{url}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Gemini API Error {response.status}: {text}")
            
            res_json = await response.json()
            try:
                return res_json['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                raise RuntimeError(f"Gemini returned no content: {res_json}")

async def ollama_chat(model: str, messages: List[Dict], **kwargs) -> str:
    """
    Ollama API client (Async).
    """
    url = f"{kwargs['ollama_url']}/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": kwargs.get('temperature', 0.0)
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Ollama API Error {response.status}: {text}")
            
            res_json = await response.json()
            return res_json['message']['content']

async def openai_generate(model: str, messages: List[Dict], **kwargs) -> str:
    """
    OpenAI API client (Async).
    """
    client = AsyncOpenAI(api_key=kwargs['api_key'], timeout=600.0)
    args = {
        "model": model,
        "messages": messages
    }
    if model not in ["gpt-5", "o1-preview", "o1-mini"]:
        args["response_format"] = {"type": "json_object"}

    if kwargs.get('temperature') is not None and model not in ["gpt-5", "o1-preview", "o1-mini"]:
         args["temperature"] = kwargs.get('temperature', 0.0)
         
    print(f"[DEBUG] Sending request to OpenAI ({model})...")
    try:
        response = await client.chat.completions.create(**args)
    except Exception as e:
        print(f"[DEBUG] OpenAI Request Failed: {e}")
        raise e
        
    print(f"[DEBUG] Response received from OpenAI.")
    return response.choices[0].message.content


# --- Core Logic ---

def print_progress(current, total, prefix="Progress:", suffix="Complete", decimals=1, length=40, fill="█", print_end="\r", verbose=False):
    """
    Call in a loop to create terminal progress bar
    """
    if verbose:
        # Just print checkmarks or milestones to not spam if detailed logs are flowing
        if current % 1 == 0: # Print every job start
             sys.stdout.write(f'{prefix} {current}/{total} {suffix}\n')
        return

    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if current == total: 
        sys.stdout.write('\n')

def get_job_source_list(args) -> List[Dict]:
    """
    Returns list of job dicts {job_id, job_text, title, ...}
    """
    jobs_out = []
    limit = args.limit
    
    if args.source == 'db':
        if not os.path.exists(args.db_path):
            print(f"Error: DB not found at {args.db_path}")
            sys.exit(1)
            
        print(f"Reading from DB: {args.db_path}")
        conn = sqlite3.connect(args.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Strategy Logic
        if args.strategy == 'mixed':
            print(f"Strategy: MIXED (All relevant + {args.irrelevant_count} irrelevant)")
            
            # 1. Get Relevant
            cursor.execute("SELECT id, title, company, location, raw_data, created_at FROM scrapes WHERE relevance LIKE 'relevant%' ORDER BY created_at DESC")
            rows_relevant = cursor.fetchall()
            print(f"Found {len(rows_relevant)} relevant jobs.")
            
            # 2. Get Irrelevant Sample
            cursor.execute(f"SELECT id, title, company, location, raw_data, created_at FROM scrapes WHERE relevance LIKE 'irrelevant%' ORDER BY RANDOM() LIMIT {args.irrelevant_count}")
            rows_irrelevant = cursor.fetchall()
            print(f"Sampling {len(rows_irrelevant)} irrelevant jobs.")
            
            # Combine
            all_rows = list(rows_relevant) + list(rows_irrelevant)
            # Shuffle in-memory
            random.shuffle(all_rows)
            
            cursor_source = all_rows
        else:
            # Default ALL
            cursor.execute("SELECT id, title, company, location, raw_data, created_at FROM scrapes ORDER BY created_at DESC")
            cursor_source = cursor.fetchall()
            
        for i, row in enumerate(cursor_source):
            if limit and i >= limit:
                break
                
            data = dict(row)
            job_text = ""
            
            # Try to extract text from raw_data
            if data.get('raw_data'):
                try:
                    raw = json.loads(data['raw_data'])
                    # Prefer raw_data description/text
                    job_text = raw.get('description') or raw.get('job_description') or raw.get('text') or ""
                except:
                    pass
            
            # Fallback if text is empty? Construct from partials
            if not job_text:
                job_text = f"Title: {data['title']}\nCompany: {data['company']}\nLocation: {data['location']}"
            
            jobs_out.append({
                "job_id": f"DB_{data['id']}",
                "job_text": job_text,
                "title": data['title'],
                "company": data['company'],
                "location": data['location']
            })
        conn.close()

    else:
        # File Source
        if not args.jobs:
            print("Error: --jobs required if --source file")
            sys.exit(1)
            
        count = 0 
        if args.jobs_format == 'jsonl' or args.jobs.endswith('.jsonl'):
            with open(args.jobs, 'r') as f:
                for line in f:
                    if limit and count >= limit: break
                    if not line.strip(): continue
                    j = json.loads(line)
                    # Normalize
                    j_id = j.get('job_id', f"JOB_{count+1}")
                    jobs_out.append({
                        "job_id": j_id,
                        "job_text": j.get('job_text') or j.get('description', ''),
                        "title": j.get('title'),
                        "company": j.get('company'),
                        "location": j.get('location')
                    })
                    count += 1
        else:
            # TXT format
            # TODO: Implement complex TXT parsing if needed, assumed JSONL primarily for now per simple req
            # Per prompt: "TXT: separated by delimiter line --- JOB ---"
            with open(args.jobs, 'r') as f:
                content = f.read()
            chunks = content.split("--- JOB ---")
            for chunk in chunks:
                if limit and count >= limit: break
                if not chunk.strip(): continue
                # Parse optional ID? 
                lines = chunk.strip().split('\n')
                j_id = f"JOB_{count+1}"
                if lines[0].startswith("Job ID:"):
                    j_id = lines[0].replace("Job ID:", "").strip()
                    chunk = "\n".join(lines[1:])
                
                jobs_out.append({
                    "job_id": j_id,
                    "job_text": chunk,
                    "title": "?", "company": "?", "location": "?"
                })
                count += 1
                
    return jobs_out

def ensure_gitignore(outdir):
    """
    Ensure the outdir is in root .gitignore
    """
    try:
        if not os.path.exists(".gitignore"):
            return
        
        # Normalize relative path for gitignore
        rel = os.path.relpath(outdir)
        if rel.startswith(".."): return # Outside root
        
        with open(".gitignore", "r") as f:
            lines = [l.strip() for l in f.readlines()]
            
        if rel not in lines and f"{rel}/" not in lines:
            print(f"Adding {rel} to .gitignore")
            with open(".gitignore", "a") as f:
                f.write(f"\n{rel}/\n")
    except Exception as e:
        print(f"Warning: could not update .gitignore: {e}")

async def process_single_job(
    job: Dict, 
    sem: asyncio.Semaphore, 
    args, 
    mode: str, 
    outdir: str, 
    provider_func, 
    provider_kwargs: Dict, 
    model: str, 
    cand_id: str, 
    cand_highlights: Dict, 
    results_jobs: List, 
    job_index: int,
    progress_callback
):
    async with sem:
        jid = job['job_id']
        progress_callback(f"Processing {jid}...")
        
        # Stage A: Extraction
        user_prompt_ext = USER_PROMPT_EXTRACTION_TEMPLATE.format(
            date=args.date,
            job_id=jid,
            job_text=job['job_text']
        )
        
        try:
            extraction = await validate_and_retry(
                provider_func, model, 
                SYSTEM_PROMPT_EXTRACTION.format(schema=json.dumps(SCHEMA_JOB_EXTRACTION, indent=2)), 
                user_prompt_ext,
                SCHEMA_JOB_EXTRACTION,
                **provider_kwargs
            )
        except Exception as e:
            print(f"  Extraction failed for {jid}: {e}")
            progress_callback(f"Failed {jid}")
            return

        # Stage B: Decision
        user_prompt_dec = USER_PROMPT_DECISION_TEMPLATE.format(
            date=args.date,
            candidate_id=cand_id,
            job_id=jid,
            candidate_highlights_json=json.dumps(cand_highlights),
            job_extraction_json=json.dumps(extraction),
            job_text=job['job_text'] # Context
        )
        
        try:
            decision = await validate_and_retry(
                provider_func, model, 
                SYSTEM_PROMPT_DECISION.format(schema=json.dumps(SCHEMA_JOB_DECISION, indent=2)), 
                user_prompt_dec,
                SCHEMA_JOB_DECISION,
                **provider_kwargs
            )
        except Exception as e:
            print(f"  Decision failed for {jid}: {e}")
            progress_callback(f"Failed {jid}")
            return
            
        # Collect Result
        res_entry = {
            "job_id": jid,
            "job_index": job_index,
            "job_meta": {
                "title": job['title'],
                "company": job['company'],
                "location": job['location']
            },
            "extraction": extraction,
            "decision": decision,
            "provenance": {
                "provider": args.label_provider if mode == 'label' else "ollama",
                "model": model,
                "prompt_version": args.prompt_version,
                "timestamp_utc": datetime.datetime.utcnow().isoformat()
            },
            "meta": {"errors": []},
            "job_text_context": job['job_text']
        }
        results_jobs.append(res_entry)
        append_result(outdir, res_entry)
        progress_callback(f"Completed {jid}")


async def run_label_or_run_mode(args):
    mode = args.mode
    outdir = args.outdir
    
    os.makedirs(outdir, exist_ok=True)
    ensure_gitignore(outdir)
    ensure_gitignore("data")
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        print("Detailed logging enabled.")

    # 0. Setup Provider
    if mode == 'label' and args.label_provider == 'gemini':
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY or GOOGLE_API_KEY env var missing.")
            sys.exit(1)
        
        provider_func = gemini_generate
        provider_kwargs = {
            "api_key": api_key,
            "url_template": args.gemini_url,
            "max_output_tokens": args.gemini_max_output_tokens,
            "temperature": args.temperature
        }
        model = args.gemini_model
        print(f"Running in LABEL mode with {model}...")
        
    elif mode == 'label' and args.label_provider == 'openai':
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY env var missing.")
            sys.exit(1)
            
        provider_func = openai_generate
        provider_kwargs = {
            "api_key": api_key,
            "temperature": args.temperature
        }
        model = args.openai_model
        print(f"Running in LABEL mode with OpenAI {model}...")

    else: # mode == 'run'
        provider_func = ollama_chat
        provider_kwargs = {
            "ollama_url": args.ollama_url,
            "temperature": args.temperature
        }
        model = args.ollama_model
        print(f"Running in RUN mode with {model}...")

    # 1. Candidate Stage
    with open(args.candidate, 'r') as f:
        profile_text = f.read()
    
    print("Stage 0: Extracting Candidate Highlights...")
    user_prompt_cand = USER_PROMPT_CANDIDATE_TEMPLATE.format(
        date=args.date,
        profile_text=profile_text
    )
    
    cand_highlights = await validate_and_retry(
        provider_func, model, 
        SYSTEM_PROMPT_CANDIDATE.format(schema=json.dumps(SCHEMA_CANDIDATE_HIGHLIGHTS, indent=2)), 
        user_prompt_cand, 
        SCHEMA_CANDIDATE_HIGHLIGHTS, 
        **provider_kwargs
    )
    
    cand_id = cand_highlights.get("candidate_id") or "CAND_" + hashlib.sha1(profile_text.encode('utf-8')).hexdigest()[:8]
    cand_highlights["candidate_id"] = cand_id
    
    # 2. Jobs Loop
    processed_ids, results_jobs = load_existing_results(outdir)
    
    all_jobs_raw = get_job_source_list(args)
    jobs_to_process = [j for j in all_jobs_raw if j['job_id'] not in processed_ids]
    
    total_jobs = len(jobs_to_process)
    print(f"loaded {len(processed_ids)} previously processed jobs.")
    print(f"Starting processing for {total_jobs} remaining jobs with batch processing...")
    
    sem = asyncio.Semaphore(5) # Concurrency Limit (Batch Size 5 safe for GPT-5 rate limits)
    tasks = []
    
    completed_count = 0
    
    def progress_callback(msg):
        nonlocal completed_count
        if "Completed" in msg or "Failed" in msg:
            completed_count += 1
            print_progress(completed_count, total_jobs, prefix="Progress:", suffix=msg, length=40, verbose=args.verbose)
    
    # Start all tasks (Semaphore limits execution)
    for i, job in enumerate(jobs_to_process):
        idx = len(results_jobs) + 1 + i
        tasks.append(process_single_job(
            job, sem, args, mode, outdir, provider_func, provider_kwargs, model, 
            cand_id, cand_highlights, results_jobs, idx, progress_callback
        ))
        
    await asyncio.gather(*tasks)

    # Final Output
    final_json = {
        "run_id": f"RUN_{int(datetime.datetime.now().timestamp())}",
        "date": args.date,
        "mode": mode,
        "teacher": {"provider": args.label_provider, "model": model} if mode == "label" else None,
        "student": {"provider": "ollama", "model": model} if mode == "run" else None,
        "prompt_version": args.prompt_version,
        "candidate": {
            "candidate_id": cand_id,
            "profile_sha1": hashlib.sha1(profile_text.encode('utf-8')).hexdigest(),
            "highlights": cand_highlights
        },
        "jobs": results_jobs
    }
    
    json_path = os.path.join(outdir, "result.json")
    with open(json_path, 'w') as f:
        json.dump(final_json, f, indent=2)
        
    print(f"Done. Final results written to {json_path}")


def load_existing_results(outdir):
    """
    Load incomplete run results from jsonl to support resume.
    Returns: processed_job_ids (set), existing_jobs (list)
    """
    jsonl_path = os.path.join(outdir, "result.jsonl")
    processed_ids = set()
    existing_jobs = []
    
    if os.path.exists(jsonl_path):
        print(f"Resuming from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    processed_ids.add(entry['job_id'])
                    existing_jobs.append(entry)
                except:
                    pass
    return processed_ids, existing_jobs

def append_result(outdir, result_entry):
    """
    Append a single result to jsonl
    """
    jsonl_path = os.path.join(outdir, "result.jsonl")
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(result_entry) + "\n")
        
    # Also append to txt log
    txt_path = os.path.join(outdir, "result.txt")
    match_map = {
        "HIGH_MATCH": "HIGH MATCH",
        "POTENTIAL_MATCH": "POTENTIAL MATCH",
        "SOCIAL_TRAP": "SOCIAL TRAP",
        "JUNIOR_WEAK": "JUNIOR/WEAK",
        "IRRELEVANT": "IRRELEVANT"
    }
    decision = result_entry['decision']
    m_level = match_map.get(decision['match_level'], decision['match_level'])
    line = f"Job {result_entry['job_index']} ({result_entry['job_id']}): {m_level} - {decision['reason']}"
    
    with open(txt_path, 'a') as f:
        f.write(line + "\n")


def run_pack_mode(args):
    if not args.labeled_json:
        print("Error: --labeled-json is required for pack mode")
        sys.exit(1)
        
    print(f"Packing data from {args.labeled_json}...")
    
    # Run Pack
    ensure_gitignore("data")
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    
    if not os.path.exists(args.labeled_json):
        print(f"Error: labeled json not found at {args.labeled_json}")
        sys.exit(1)
        
    with open(args.labeled_json, 'r') as f:
        data = json.load(f)
        
    cand_id = data['candidate']['candidate_id']
    cand_highlights_str = json.dumps(data['candidate']['highlights'])
    
    samples = []
    
    for job in data['jobs']:
        # We need job_text.
        job_text = job.get('job_text_context')
        
        if not job_text:
            continue # Skip if we can't reconstruct
        
        user_content = USER_PROMPT_DECISION_TEMPLATE.format(
            date=data['date'],
            candidate_id=cand_id,
            job_id=job['job_id'],
            candidate_highlights_json=cand_highlights_str,
            job_extraction_json=json.dumps(job['extraction']),
            job_text=job_text
        )
        
        
        # Golden Output
        decision = job['decision']
        if getattr(args, 'simplify_labels', False):
             if decision['match_level'] in ['SOCIAL_TRAP', 'JUNIOR_WEAK']:
                 decision = decision.copy()
                 decision['match_level'] = 'IRRELEVANT'
                 
        assistant_content = json.dumps(decision)
        
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_DECISION.format(schema=json.dumps(SCHEMA_JOB_DECISION, indent=2))},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "meta": {
                "candidate_id": cand_id,
                "job_id": job['job_id'],
                "schema_version": "v1",
                "prompt_version": data['prompt_version'],
                "label_provider": data.get('teacher', {}).get('provider'),
                "label_model": data.get('teacher', {}).get('model')
            }
        }
        samples.append(sample)
        
    # Shuffle and Split
    random.seed(args.seed)
    random.shuffle(samples)
    
    if not samples:
        print("Warning: No valid samples created (missing job_text_context?)")
        return

    split_idx = int(len(samples) * (1 - args.valid_ratio))
    train_set = samples[:split_idx]
    valid_set = samples[split_idx:]
    
    with open(args.train_out, 'w') as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")
            
    with open(args.valid_out, 'w') as f:
        for s in valid_set:
            f.write(json.dumps(s) + "\n")
            
    print(f"Packed {len(samples)} samples.")
    print(f"Train: {len(train_set)} -> {args.train_out}")
    print(f"Valid: {len(valid_set)} -> {args.valid_out}") 



async def main_async():
    parser = argparse.ArgumentParser(description="Two-Stage Job Matcher")
    parser.add_argument("--mode", choices=["label", "run", "pack"], required=True)
    
    # Global optionals
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true", help="Enable detailed HTTP/API logging")
    
    # Source Args
    parser.add_argument("--source", choices=["file", "db"], default="db")
    parser.add_argument("--db-path", default="job_search_data.db")
    parser.add_argument("--strategy", choices=["all", "mixed"], default="all", help="all: process all jobs, mixed: all relevant + N irrelevant")
    parser.add_argument("--irrelevant-count", type=int, default=30, help="Number of irrelevant jobs to sample in mixed strategy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs")
    parser.add_argument("--jobs-format", default="jsonl")
    parser.add_argument("--candidate", help="Required for label/run")
    parser.add_argument("--outdir", default="results/matcher_labels")
    
    # Gemini
    parser.add_argument("--gemini-model", default="gemini-3-pro-preview")
    parser.add_argument("--gemini-url", default="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
    parser.add_argument("--gemini-max-output-tokens", type=int, default=8192)

    # OpenAI
    parser.add_argument("--label-provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--openai-model", default="gpt-5")
    
    # Ollama
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="qwen2.5:3b-instruct")
    
    # Pack
    parser.add_argument("--labeled-json")
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--valid-out", default="data/valid.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simplify-labels", action="store_true", help="Map detailed rejection labels to IRRELEVANT")
    
    args = parser.parse_args()

    if args.mode in ['label', 'run']:
        if not args.candidate:
            print("Error: --candidate is required for label/run mode")
            sys.exit(1)
        await run_label_or_run_mode(args)
        
    elif args.mode == 'pack':
        run_pack_mode(args)

if __name__ == "__main__":
    asyncio.run(main_async())

