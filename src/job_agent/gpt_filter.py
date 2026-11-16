import logging
import pandas as pd
import os
import re
import requests as http_requests
from pypdf import PdfReader
from openai import OpenAI
from datetime import datetime
from typing import List, Tuple
from .candidate_profile import fill
from .config import DEFAULT_FILTER_MODEL, KEYWORDS, EXCLUDE_PHRASES, FALLBACK_CHAIN, MODEL_EFFORT, TRUNCATION_JOB_DESC_FILTER
from .gemini_utils import call_gemini_with_retry
from .jd_sanitizer import clean_jd

DEFAULT_PROMPT_TEMPLATE = '''
Current Date: {date}

=== CANDIDATE PROFILE ===
{profile_text}

=== JOB DESCRIPTION ===
{jobs_block}

=== INSTRUCTIONS ===
Act as a strict Executive Recruiter. Your goal is to determine if these roles are a STRATEGIC MATCH for the candidate.

Analyze each job based on these strict rules.
CRITICAL: REJECTION RULES OVERRIDE POTENTIAL MATCHES. If a job violates a rejection rule, it MUST be rejected, even if it looks like a "good opportunity" otherwise.

1. THE "SOCIAL MEDIA/GROWTH" TRAP (CRITICAL REJECTION):
   - {{profile.summary}}
   - They are NOT a Social Media Manager, Growth Marketer, or Community Manager.
   - REJECT (SOCIAL TRAP) if the core role is:
     * Community management, posting schedules, answering comments.
     * Purely analytics, growth hacking, or performance marketing.
     * "Content Strategist" roles that are actually just copywriting/blogging.
   - ACCEPT if the role is:
     * Producing content (video, audio, podcast, games, live events) for any platform.
     * Managing production operations, budgets, schedules, or creative workflows.
     * Creative operations, studio operations, or production management.
     * Program/project management in media, entertainment, or creative industries.
     * YouTube/creator partnerships, channel management.
     * Localization, dubbing, or post-production management.

1b. THE "TECH PM" TRAP (CRITICAL REJECTION):
   - This search targets production roles; engineering/software work is outside the target profile.
   - REJECT (IRRELEVANT) if the role requires:
     * Technical Program/Project Management (TPM) with engineering oversight
     * Technical skills: AWS, Kubernetes, CI/CD, SQL, Python, Java, cloud platforms, APIs, microservices
     * Engineering team management, sprint planning for software teams, release engineering
     * DevOps, SRE, Infrastructure management
   - REJECT "Product Manager" roles (different career track from Producer).
   - ACCEPT "Project Manager" or "Program Manager" ONLY if the context is clearly media, entertainment, creative, or content production — NOT software/engineering.

1c. THE "DIRECTOR/HEAD-LEVEL" CHECK:
   - Titles like "Head of Content," "VP of Production," or "Director of Content" are OVER-LEVELED for this candidate regardless of domain fit (the target level is individual-contributor, not org leadership).
   - If production-leadership: grade AT MOST "WORTH A LOOK" — never HIGH or POTENTIAL.
   - REJECT (IRRELEVANT) if marketing-leadership with no production component.
   - The CREATIVE-CRAFT leadership ladder is not the candidate's at ALL: "Creative Director," "Associate Creative Director," "Art Director," "Design Director" → REJECT (IRRELEVANT), not capped. These roles run craft/design ladders (visual concepting, copy, design systems) — a different profession from production.

1d. THE "CRAFT SPECIALIST" TRAP (CRITICAL REJECTION):
   - The target role is a PROJECT-MANAGER-style PRODUCER: planning, budgeting, scheduling, coordinating crews and vendors, and delivering productions. It is NOT a hands-on creative specialist role. Editing/animation/design tools on a profile count ONLY as supporting skills for supervising that work — "capable of" is NOT "the candidate's job". Do not map the candidate's skills onto craft roles.
   - REJECT (CRAFT SPECIALIST) if the core deliverable is personally executing a craft:
     * Motion Designer / Motion Graphics / Animator / 3D Artist
     * Videographer / Camera Operator / Shooter / Photographer
     * Video Editor as the primary job (staff editor; "edit N videos per week" as the core duty)
     * Graphic Designer / Visual Designer / Illustrator
     * Sound Engineer / Audio Engineer / Mixer
     * Copywriter / Writer as the primary job
     * AI Artist / Prompt Artist / AI Content Creator — personally generating AI images/video/assets as the core deliverable.
     * On-camera talent — Host / Presenter / Streamer / Livestream Host: the core is BEING the talent, not producing the show
     * "Content Creator" / "Shorts & Reels Creator" / TikTok-creator roles where the core is personally shooting, editing, and posting the content
     * Physical event labor — stagehand, set construction / strike crew, fabrication, builder/painter tiers
     * Non-production operations dressed in content language ("plus social content" does not make them production)
     * AI-output RATING / data-annotation / rubric-grading gigs — reviewing AI-generated content against rubrics: grading model output for a rating platform is a different profession, however media-adjacent the domain reads
   - HYBRID EXCEPTION (EVIDENCE-GATED): hybrid status requires the DESCRIPTION to show real coordination duties — schedules / budgets / vendors / crew booking / client management — as a SUBSTANTIAL half. A slash-title or the word "producer"/"production"/"post-production" alone is NOT evidence; if the day-to-day is operating the craft (edit suite, camera, OBS/streaming rig, being on camera), it is a craft REJECT. Genuine hybrids (e.g. "Videographer/Producer" with evidenced producer duties): grade the PRODUCER half — cap at "WORTH A LOOK" unless production clearly dominates the description. AI-content roles centered on DIRECTING/QC of AI output or building AI content pipelines (not personally generating assets) are hybrids too — cap at "WORTH A LOOK", never HIGH.

2. THE "LEVEL" CHECK (MOST IMPORTANT):
   - {{profile.summary}}
   - STRONG TARGET: hands-on IC production at the level stated in the candidate summary — Producer, Senior Producer (IC), Associate Producer, Lead Producer, hands-on Production Manager (process, not people). Line Producer and Associate Producer ARE relevant — Associate Producer is a legitimate target level, NOT junior.
   - BELOW-LEVEL WALL: coordinator-level roles (Production / Event / Content / Post-Production / Studio Coordinator and the like) and other entry-level roles (assistant, trainee, or a JD whose experience requirement tops out at 0-2 years) are BELOW the target level and its pay band → REJECT (JUNIOR/WEAK). The OVERQUALIFIED EXCEPTION below does NOT apply to them, and "DO NOT penalize under-leveled roles" does NOT apply to them, and the WORTH A LOOK "slightly under-leveled" / "prefer WORTH A LOOK over IRRELEVANT" guidance does NOT apply to them — a coordinator role at a well-known brand is still a coordinator role. A title that co-names a producer role ("Production Coordinator / Producer") is judged on the DESCRIPTION: if the day-to-day is supporting producers (scheduling, logistics, paperwork), it is coordinator-level.
   - DO NOT penalize under-leveled roles — a mid/associate role at a good company is the TARGET, not a step down.
   - LEADERSHIP-TITLE CAP: an over-leveled / ambiguous leadership TITLE (Principal/Lead/Executive Producer) with NO direct-report evidence = AT MOST "WORTH A LOOK" — keep VISIBLE (it may be inflated or hands-on); do NOT reject on the title alone.
   - DIRECTOR DISAMBIGUATION: "Director" in a title has THREE readings and your reasoning must NAME which one the JD shows: (a) ORG-SENIORITY Director — owns strategy/teams/org budgets, or the JD demands many years of leadership experience → REJECT (seniority/people-management wall); (b) CRAFT Director — film/video director, art director, creative director: the person who DIRECTS the creative work → REJECT (craft ladder); (c) INFLATED TITLE on a hands-on producer role — the JD reads as schedules/budgets/vendors IC work under a grand title → WORTH A LOOK, but ONLY with explicit JD evidence of the hands-on core. Neither (a) nor (b) fits the candidate; (c) must be EVIDENCED, never assumed as the default. NO-EVIDENCE DIRECTORS: a Director title whose description is EMPTY or a stub has no evidence and CANNOT reach (c) — and the general "description insufficient → ambiguous, keep visible" branch does NOT apply to Director titles: a Director with no JD evidence is REJECT (a known failure mode: Director titles passing on empty or stub descriptions, even with org-seniority named in the verdict). THE CAP NEVER APPLIES TO DIRECTOR TITLES: a 'Director' resolves ONLY through this disambiguation — readings (a) org-seniority and (b) craft are REJECT, not 'capped at WORTH A LOOK'. Writing 'this is an org-seniority Director' (owns departmental budgets, reports into senior leadership, runs a portfolio/function) and then choosing WORTH A LOOK is the exact contradiction VERDICT-BINDING forbids — resolve it as REJECT (a known failure mode: a verdict that names org-seniority for a Director title, then caps it).
   - PEOPLE-MANAGEMENT TEST: scan the DESCRIPTION for explicit direct-report language — "manage a team of / hire / headcount / direct reports / manage managers / performance reviews". If the role's CORE is people management, REJECT (IRRELEVANT) — the target level is individual-contributor with no direct reports, so it is not viable however production-y the title. An org-exec title (VP/SVP/Chief/Head of <function>) is people management by definition → also REJECT. The hands-on exception (NOT a reject) holds only when the role owns schedules/budgets/vendors with NO team to manage — "collaborate/work with the team" is NOT people management.
   - VERDICT-BINDING (hard override — applies to EVERY rule above, not just this one): your verdict MUST follow your own reasoning. If your analysis identifies ANY enumerated REJECT condition — confirmed people-management (explicit hiring/performance-reviews/succession/direct-reports), a craft-executor or creative-craft/directing-ladder role, a game-development producer, CMS/web-build work, a role with NO production component, a different profession, or a HARD JD requirement the candidate cannot meet (e.g. "15+ years in X", a mandatory platform/domain the candidate lacks) — the verdict IS REJECT. Do NOT soften a disqualifier you yourself identified into "WORTH A LOOK". WORTH A LOOK exists for exactly two cases: (1) genuinely adjacent-plausible — a production-transferable core with a SOFT gap; (2) genuinely ambiguous — the description is insufficient to decide (e.g. a leadership title with no direct-report evidence). Writing "outside the candidate's domain" / "not production-focused" / "critical gap" and then choosing WORTH A LOOK is a contradiction — resolve it as REJECT. (Hedged rejections like this are the most common way the worth-a-look tier fills with noise.)
   - BUSINESS CAP: account/client/partner management = AT MOST "WORTH A LOOK"; REJECT if no production component.
   - REJECT: Intern, unpaid, volunteer, 0-1 year experience only, one-man-band videographer roles.
   - OVERQUALIFIED EXCEPTION: Accept Junior/Entry-level titles at recognisable organisations (major studios, streamers, public broadcasters, leading tech firms, well-known universities, or sector-leading organisations in entertainment/media/events) when the domain matches. Mark as POTENTIAL MATCH with reason "overqualified but strategic fit." This exception NEVER applies to coordinator-level, assistant, trainee or 0-2-year roles — those are REJECT under the BELOW-LEVEL WALL.

3. LOCATION & INDUSTRY:
   - Must be in the candidate's home metro or home state (the candidate is in {{profile.home_location}}), or US-based Remote. REJECT (IRRELEVANT) if the ONLY location is on-site outside that area with no remote option. REJECT "remote" roles restricted to a non-US region (EMEA / India / UK / Canada / LATAM). EXCEPTION: on-site in a secondary metro that the candidate summary names as acceptable is allowed but weak (grade at most WORTH A LOOK).
   - Industry: Tech, Media, Entertainment, Gaming, Creator Economy preferred.
   - GOVERNMENT / DEFENSE WALL: roles supporting government, defense, military, or homeland-security programs (contractor-staffed federal seminars/symposia, military education centers, defense IT firms) → REJECT even when the FUNCTION reads as event/production coordination — federal-contract environments are a different market from media production and are commonly clearance-gated. "Event coordination is event coordination" is the rationalization to refuse.
   - MARKETING-EVENTS WALL: corporate-marketing event roles are a DIFFERENT PROFESSION, not a live-events production adjacency → REJECT. The family: Field Marketing (Manager/Specialist/Marketer), Marketing Events Specialist/Coordinator/Manager, tradeshow/conference-booth and sponsorship-activation roles, demand-gen event programs (webinars, executive dinners, lead capture/MQLs, booth-and-swag logistics), corporate Meeting Planner, fundraising/development-events roles, advertising coordination, and Design Program Manager / design-ops (running a design org's workflow). Their deliverable is leads, brand presence, or design throughput — not a produced show or content. A live-events/experiential background does NOT rescue these: that adjacency applies ONLY when the JD shows produced-show/content duties (run-of-show, staging, AV/broadcast production, filming, editorial output) — "it's an event role, and events are events" is the rationalization to refuse.

4. CAPABILITY MATCH (The "Bridge"):
   - Does the experience evidenced in the candidate profile (production workflows, cross-functional coordination, creative operations) map to the JD's requirements?

5. DOMAIN AWARENESS:
   Each job listing includes a "Domain Match" field indicating the candidate's strongest CV variant for that role. The domains include: producer_core, live_and_experiential, production_ops_and_management, project_and_program_management, podcast_and_audio, youtube_and_creators, games_production. A strong domain alignment boosts the match; a mismatch should temper confidence. Domain alignment alone does not override rejection rules.

=== OUTPUT FORMAT ===
Output a single line per job in this format:
Job X: [MATCH LEVEL] - [DETAILED REASONING: 2-3 sentences explaining WHY this match level was assigned. Cover: (1) which specific candidate skills/experience map to the role, (2) what makes this opportunity interesting or concerning, (3) any domain alignment or gaps. Be specific — reference actual job requirements and candidate qualifications.]

Where [MATCH LEVEL] is one of:
- HIGH MATCH (Perfect fit for skills & seniority)
- POTENTIAL MATCH (Good fit, but strictly adheres to all Rejection Rules)
- WORTH A LOOK (Adjacent or a stretch: content-heavy but light on video, editor/producer hybrid, slightly over/under-leveled, or a game-dev producer. Surfaced in a lower tier. PREFER this over IRRELEVANT whenever there is real production signal but you are not confident it is a strong fit.)
- SOCIAL TRAP (Rejected: Role is social media, growth, or community management)
- JUNIOR/WEAK (Rejected: Too junior or lacks production focus)
- IRRELEVANT (Completely wrong field or location)

Games nuance: content roles at game studios (trailers, cinematics, capture, social) can be HIGH/POTENTIAL, but a pure GAME PRODUCER shipping the game itself is at most WORTH A LOOK.
'''

# Claude-optimized Stage 2 prompt — uses XML tags for structured input sections.
# Claude models parse XML structure more reliably than === HEADER === markers.
STAGE2_PROMPT_CLAUDE = '''
Current Date: {date}

<candidate_profile>
{profile_text}
</candidate_profile>

<job_descriptions>
{jobs_block}
</job_descriptions>

<instructions>
Act as a strict Executive Recruiter. Determine if these roles are a STRATEGIC MATCH for the candidate.

CRITICAL: REJECTION RULES OVERRIDE POTENTIAL MATCHES. If a job violates a rejection rule, it MUST be rejected.

<rejection_rules>
1. SOCIAL MEDIA/GROWTH TRAP:
   {{profile.summary}}
   REJECT if the core role is community management, posting schedules, analytics/growth hacking, or copywriting/blogging disguised as "Content Strategist."
   ACCEPT if producing content (video/audio/podcast/games/live events), managing production ops/budgets/schedules, creative operations, studio operations, media program/project management, YouTube/creator partnerships, localization, or post-production.

2. TECH PM TRAP:
   This search targets production roles; engineering/software work is outside the target profile.
   REJECT if the role requires: TPM with engineering oversight, technical skills (AWS, Kubernetes, CI/CD, SQL, Python, Java, cloud, APIs, microservices), engineering team management, DevOps, SRE, or infrastructure.
   REJECT "Product Manager" roles (different career track).
   ACCEPT "Project/Program Manager" ONLY in media, entertainment, creative, or content production contexts.

2b. DIRECTOR/HEAD-LEVEL TITLE CHECK:
   Titles like "Head of Content," "VP of Production," or "Director of Content" are OVER-LEVELED for this candidate regardless of domain fit — the target level is individual-contributor, so a leadership title is not a realistic hire.
   If production-leadership (overseeing pipelines, managing producers): grade AT MOST "WORTH A LOOK" — never HIGH or POTENTIAL.
   REJECT (IRRELEVANT) if marketing-leadership: demand generation, lead nurture, sales enablement, growth/brand marketing strategy with no production component.
   The CREATIVE-CRAFT leadership ladder is not the candidate's at ALL: "Creative Director", "Associate Creative Director", "Art Director", "Design Director" → REJECT (IRRELEVANT), not capped. These run craft/design ladders (visual concepting, copy, design systems) — a different profession from production.

2c. CRAFT SPECIALIST TRAP (CRITICAL REJECTION):
   The target role is a PROJECT-MANAGER-style PRODUCER: planning, budgeting, scheduling, coordinating crews and vendors, and delivering productions. It is NOT a hands-on creative specialist role. Editing/animation/design tools on a profile count ONLY as supporting skills for supervising that work — "capable of" is NOT "the candidate's job". Do NOT map the candidate's skills onto craft-execution roles.
   REJECT (CRAFT SPECIALIST → use IRRELEVANT) if the core deliverable is personally executing a craft:
   - Motion Designer / Motion Graphics / Animator / 3D Artist
   - Videographer / Camera Operator / Shooter / Photographer
   - Video Editor as the primary job (staff editor; "edit N videos per week" as the core duty)
   - Graphic Designer / Visual Designer / Illustrator
   - Sound Engineer / Audio Engineer / Mixer
   - Copywriter / Writer as the primary job
   - AI Artist / Prompt Artist / AI Content Creator — personally generating AI images/video/assets as the core deliverable.
   - On-camera talent — Host / Presenter / Streamer / Livestream Host: the core is BEING the talent, not producing the show
   - "Content Creator" / "Shorts & Reels Creator" / TikTok-creator roles where the core is personally shooting, editing, and posting the content
   - Physical event labor — stagehand, set construction / strike crew, fabrication, builder/painter tiers
   - Non-production operations dressed in content language ("plus social content" does not make them production)
   - AI-output RATING / data-annotation / rubric-grading gigs — reviewing AI-generated content against rubrics: grading model output for a rating platform is a different profession, however media-adjacent the domain reads
   HYBRID EXCEPTION (EVIDENCE-GATED): hybrid status requires the DESCRIPTION to show real coordination duties — schedules / budgets / vendors / crew booking / client management — as a SUBSTANTIAL half. A slash-title or the word "producer"/"production"/"post-production" alone is NOT evidence; if the day-to-day is operating the craft (edit suite, camera, OBS/streaming rig, being on camera), it is a craft REJECT. Genuine hybrids (e.g. "Videographer/Producer" with evidenced producer duties): grade the PRODUCER half — cap at "WORTH A LOOK" unless production clearly dominates the description. AI-content roles centered on DIRECTING/QC of AI output or building AI content pipelines (not personally generating assets) are hybrids too — cap at "WORTH A LOOK", never HIGH.

3. LEVEL CHECK — THE MOST IMPORTANT RULE:
   {{profile.summary}}
   STRONG TARGET (HIGH MATCH eligible): hands-on INDIVIDUAL-CONTRIBUTOR production at the level stated in the candidate summary — Producer, Senior Producer (IC, no direct reports), Associate Producer, Lead Producer, hands-on Production Manager (schedules/budgets/vendors — process, not people). Line Producer and Associate Producer ARE relevant — Associate Producer is a legitimate target level, NOT junior.
   BELOW-LEVEL WALL: coordinator-level roles (Production / Event / Content / Post-Production / Studio Coordinator and the like) and other entry-level roles (assistant, trainee, or a JD whose experience requirement tops out at 0-2 years) are BELOW the target level and its pay band → REJECT (JUNIOR/WEAK). The OVERQUALIFIED EXCEPTION below does NOT apply to them, and "DO NOT penalize under-leveled roles" does NOT apply to them, and the WORTH A LOOK "slightly under-leveled" / "prefer WORTH A LOOK over IRRELEVANT" guidance does NOT apply to them — a coordinator role at a well-known brand is still a coordinator role. A title that co-names a producer role ("Production Coordinator / Producer") is judged on the DESCRIPTION: if the day-to-day is supporting producers (scheduling, logistics, paperwork), it is coordinator-level.
   DO NOT penalize "under-leveled" roles: a mid or associate role at a good company is the TARGET, not a step down — never downgrade a role merely for being "below the candidate's trajectory".
   LEADERSHIP-TITLE CAP: an over-leveled or ambiguous leadership TITLE (Principal, Lead, Executive Producer) with NO direct-report evidence in the description is AT MOST "WORTH A LOOK" — never HIGH MATCH or POTENTIAL MATCH, but keep it VISIBLE (it may be inflated or a hands-on senior role); do NOT reject on the title alone.
   DIRECTOR DISAMBIGUATION: "Director" in a title has THREE readings and your reasoning must NAME which one the JD shows: (a) ORG-SENIORITY Director — owns strategy/teams/org budgets, or the JD demands many years of leadership experience → REJECT (seniority/people-management wall); (b) CRAFT Director — film/video director, art director, creative director: the person who DIRECTS the creative work → REJECT (craft ladder); (c) INFLATED TITLE on a hands-on producer role — the JD reads as schedules/budgets/vendors IC work under a grand title → WORTH_A_LOOK, but ONLY with explicit JD evidence of the hands-on core. Neither (a) nor (b) fits the candidate; (c) must be EVIDENCED, never assumed as the default. NO-EVIDENCE DIRECTORS: a Director title whose description is EMPTY or a stub has no evidence and CANNOT reach (c) — and the general "description insufficient → ambiguous, keep visible" branch does NOT apply to Director titles: a Director with no JD evidence is REJECT (a known failure mode: Director titles passing on empty or stub descriptions, even with org-seniority named in the verdict). THE CAP NEVER APPLIES TO DIRECTOR TITLES: a 'Director' resolves ONLY through this disambiguation — readings (a) org-seniority and (b) craft are REJECT, not 'capped at WORTH A LOOK'. Writing 'this is an org-seniority Director' (owns departmental budgets, reports into senior leadership, runs a portfolio/function) and then choosing WORTH A LOOK is the exact contradiction VERDICT-BINDING forbids — resolve it as REJECT (a known failure mode: a verdict that names org-seniority for a Director title, then caps it).
   PEOPLE-MANAGEMENT TEST (run FIRST): a title can hide a people-leadership role. Scan the DESCRIPTION for explicit direct-report language — "manage a team of", "hire", "headcount", "direct reports", "people leadership", "manage managers", "performance reviews". If the role's CORE is people management, REJECT (IRRELEVANT) — the target level is individual-contributor with no direct reports, so a role built around running/hiring/reviewing a team is not viable, no matter how production-relevant the title looks. An org-level executive title (VP / SVP / EVP / Chief X Officer / Head of <function>) is people management by definition → also REJECT. A "Manager" / "Program Manager" / "Manager, <Department>" with team-ownership language is people management, NOT a hands-on Production Manager. The hands-on exception (a REAL match, not a reject) holds ONLY when the role owns SCHEDULES / BUDGETS / VENDORS / DELIVERABLES with NO team to manage — "collaborate with / work with the team" is NOT people management. Reframing "manages a team" as "production leadership" does NOT avoid the REJECT.
   VERDICT-BINDING (hard override — applies to EVERY rule in this prompt, not just this one): your verdict MUST follow your own reasoning. If your analysis identifies ANY enumerated REJECT condition — confirmed people-management (explicit hiring/performance-reviews/succession/direct-reports), a craft-executor or creative-craft/directing-ladder role, a game-development producer, CMS/web-build work, a role with NO production component, a different profession, or a HARD JD requirement the candidate cannot meet (e.g. "15+ years in X", a mandatory platform/domain the candidate lacks) — the verdict IS REJECT. Do NOT soften a disqualifier you yourself identified into WORTH_A_LOOK. WORTH_A_LOOK exists for exactly two cases: (1) genuinely adjacent-plausible — a production-transferable core with a SOFT gap; (2) genuinely ambiguous — the description is insufficient to decide (e.g. a leadership title with no direct-report evidence). Writing "outside the candidate's domain" / "not production-focused" / "critical gap" and then choosing WORTH_A_LOOK is a contradiction — resolve it as REJECT. (Hedged rejections like this are the most common way the worth-a-look tier fills with noise.)
   BUSINESS CAP: account/client/partner/customer-success management or revenue-ownership roles are AT MOST "WORTH A LOOK"; REJECT (IRRELEVANT) if no production component. Reframing a business title as "production ops" does NOT lift the cap.
   REJECT: Intern, unpaid, volunteer, 0-1 year experience only, one-man-band videographer roles.
   OVERQUALIFIED EXCEPTION: Accept Junior/Entry-level titles at recognisable organisations (major studios, streamers, public broadcasters, leading tech firms, well-known universities, or sector-leading organisations in entertainment/media/events) when the domain matches. Mark as POTENTIAL MATCH with reason "overqualified but strategic fit." This exception NEVER applies to coordinator-level, assistant, trainee or 0-2-year roles — those are REJECT under the BELOW-LEVEL WALL.

4. LOCATION: the candidate's home metro or home state (the candidate is in {{profile.home_location}}), or US-based Remote. REJECT (IRRELEVANT) if the ONLY location is on-site outside that area with no remote option, or a "remote" role restricted to a non-US region (EMEA/India/UK/Canada). EXCEPTION: on-site in a secondary metro that the candidate summary names as acceptable is allowed but weak (WORTH A LOOK at most).
5. INDUSTRY: Tech, Media, Entertainment, Gaming, Creator Economy preferred.
   GOVERNMENT / DEFENSE WALL: roles supporting government, defense, military, or homeland-security programs (contractor-staffed federal seminars/symposia, military education centers, defense IT firms) → REJECT (IRRELEVANT) even when the FUNCTION reads as event/production coordination — federal-contract environments are a different market from media production and are commonly clearance-gated. "Event coordination is event coordination" is the rationalization to refuse.
   MARKETING-EVENTS WALL: corporate-marketing event roles are a DIFFERENT PROFESSION, not a live-events production adjacency → REJECT (IRRELEVANT). The family: Field Marketing (Manager/Specialist/Marketer), Marketing Events Specialist/Coordinator/Manager, tradeshow/conference-booth and sponsorship-activation roles, demand-gen event programs (webinars, executive dinners, lead capture/MQLs, booth-and-swag logistics), corporate Meeting Planner, fundraising/development-events roles, advertising coordination, and Design Program Manager / design-ops (running a design org's workflow). Their deliverable is leads, brand presence, or design throughput — not a produced show or content. A live-events/experiential background does NOT rescue these: that adjacency applies ONLY when the JD shows produced-show/content duties (run-of-show, staging, AV/broadcast production, filming, editorial output) — "it's an event role, and events are events" is the rationalization to refuse.
6. CAPABILITY MATCH: Does the experience evidenced in the candidate profile (production workflows, cross-functional coordination, creative operations) map to the JD?
</rejection_rules>

<domain_awareness>
Each job listing includes a "Domain Match" field indicating the candidate's strongest CV variant for that role. The domains include:
- producer_core: general video/content/media production
- live_and_experiential: live events, broadcast, experiential production
- production_ops_and_management: production operations, workflow design, studio management
- project_and_program_management: cross-functional program/project management in creative industries
- podcast_and_audio: podcast and audio content production
- youtube_and_creators: YouTube channel management, creator partnerships, platform growth
- games_production: game production, sprint-based pipelines, shipping interactive products

Use the Domain Match to inform your assessment: a strong domain alignment boosts the match, while a mismatch (e.g., games domain for a podcast role) should temper confidence. Domain alignment alone does not override rejection rules.
</domain_awareness>

<tiering>
We want WIDE RECALL with honest labeling. Use a softer middle verdict so adjacent roles surface in a clearly-labeled lower tier instead of being dropped:
- WORTH A LOOK: the role is ADJACENT or a STRETCH rather than a clean fit — content-heavy but lighter on video, a hybrid editor/producer role, slightly over- or under-leveled, or a producer role at a GAMES studio that leans more game-development than content. PREFER "WORTH A LOOK" over "IRRELEVANT" whenever the role shows genuine production signal (shoots, edits, post, campaigns, content calendars, briefs, deliverables) but you are not confident it is a strong fit.
- GAMES NUANCE: content roles at game studios (trailers, cinematics, capture, social, brand) can be HIGH or POTENTIAL; but a pure GAME PRODUCER who ships the game itself (sprints, milestones, feature delivery) is at most WORTH A LOOK — never HIGH MATCH.
Keep SOCIAL TRAP / JUNIOR/WEAK / IRRELEVANT for roles that are genuinely the WRONG JOB, not merely a stretch.
</tiering>

<output_format>
Output a single line per job:
Job X: [MATCH LEVEL] - [DETAILED REASONING: 2-3 sentences explaining WHY this match level was assigned. Cover: (1) which specific candidate skills/experience map to the role, (2) what makes this opportunity interesting or concerning, (3) any domain alignment or gaps. Be specific — reference actual job requirements and candidate qualifications.]

Match levels: HIGH MATCH | POTENTIAL MATCH | WORTH A LOOK | SOCIAL TRAP | JUNIOR/WEAK | IRRELEVANT
</output_format>
</instructions>
'''

# Initialize clients lazily
_openai_client = None
_anthropic_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None # Fail gracefully if key missing but not used
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError:
            logging.error("anthropic package not installed. Run: pip install anthropic")
            return None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        # Bounded: the SDK default (600s x 3 attempts = 30 min per hung call) sat
        # behind synchronous API routes; 300s x 2 keeps a worst case under 10 min
        # and inside the outreach/prep stale-claim window (1380s).
        _anthropic_client = anthropic.Anthropic(api_key=api_key, timeout=300.0, max_retries=2)
    return _anthropic_client

def _call_ollama(model_tag: str, prompt: str, temperature: float = 0.0) -> str:
    """Call a local Ollama model via its HTTP API."""
    url = "http://localhost:11434/api/generate"
    try:
        # 60s, not 300: a Stage-1 fallback that waits 5 min PER CALL turns an
        # API outage into a multi-hour hang (one run spent 25 min in Ollama
        # timeouts). A local model that can't answer a binary YES/NO in 60s
        # isn't going to answer at all.
        resp = http_requests.post(url, json={
            "model": model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except http_requests.ConnectionError:
        logging.error("Ollama not running. Start it with: ollama serve")
        return ""
    except Exception as e:
        logging.error(f"Ollama call failed ({model_tag}): {e}")
        return ""


def is_ollama_responsive(timeout: float = 3.0) -> bool:
    """Quick probe: is a local Ollama server up and answering?

    Used at cascade start to empty the ollama/* fallback chain when the
    server is down/asleep — otherwise every Stage-1 failure burns a full
    request timeout per fallback model before giving up.
    """
    try:
        resp = http_requests.get("http://localhost:11434/api/tags", timeout=timeout)
        return resp.ok
    except Exception:
        return False

def call_llm(prompt: str, model: str = DEFAULT_FILTER_MODEL, temperature: float = 0.0,
             usage_out: dict = None, effort: str = None, _allow_fallback: bool = True) -> str:
    """
    Generic wrapper to call LLMs (Gemini, OpenAI, Anthropic, or Ollama).
    Model routing by prefix:
      - "gemini-*"  → Google Gemini API
      - "claude-*"  → Anthropic API
      - "ollama/*"  → Local Ollama server
      - everything else → OpenAI API

    usage_out: optional dict the Anthropic branch fills with real
    input_tokens/output_tokens (LangFuse observability). Other providers
    leave it untouched — callers fall back to estimates.

    effort: Claude thinking effort (low/medium/high/xhigh/max). When omitted,
    config.MODEL_EFFORT supplies a pin for models whose API default differs
    from what this pipeline was tuned on (Opus 5.5 defaults to "medium").

    A Claude REFUSAL (stop_reason "refusal") never returns partial text: the
    request is retried down FALLBACK_CHAIN[model], one level deep, and "" is
    returned if nothing answers. Refusals are model-specific classifier
    decisions, so another model is the documented remedy — except
    "reasoning_extraction" declines, which Anthropic says not to retry.
    """

    # === GEMINI ===
    if model.lower().startswith("gemini"):
        fallback = FALLBACK_CHAIN.get(model, [])
        gemini_fallback = [m for m in fallback if m.lower().startswith("gemini")]
        cross_provider_fallback = [m for m in fallback if not m.lower().startswith("gemini")]

        result = call_gemini_with_retry(
            model,
            prompt,
            temperature,
            generation_config={"max_output_tokens": 8192},
            fallback_chain=gemini_fallback
        )
        if result:
            return result

        # If all Gemini options fail, try configured cross-provider fallbacks
        # (e.g., Claude) in order.
        for fallback_model in cross_provider_fallback:
            logging.warning(f"Gemini chain failed for {model}; trying fallback model {fallback_model}")
            result = call_llm(prompt, model=fallback_model, temperature=temperature,
                              usage_out=usage_out)
            if result:
                return result
        return ""

    # === OLLAMA (local) ===
    if model.lower().startswith("ollama/"):
        model_tag = model.split("/", 1)[1]  # "ollama/qwen3:8b" → "qwen3:8b"
        return _call_ollama(model_tag, prompt, temperature)

    # === ANTHROPIC CLAUDE ===
    if model.lower().startswith("claude"):
        try:
            client = get_anthropic_client()
            if not client:
                logging.error("Anthropic client not initialized (check ANTHROPIC_API_KEY).")
                return ""

            # Newer Claude models (opus-4-7+, sonnet-5, fable-5) REJECT sampling
            # params with a 400 ("temperature is deprecated"). Allowlist the
            # legacy models that still accept it; anything newer/unknown omits —
            # that matches the API's direction of travel (probed 2026-07-16).
            # Thinking models (fable-5/sonnet-5/opus-5/opus-5-5) spend the output budget
            # on thinking blocks BEFORE any text block — at 8192 a long think
            # can starve the answer entirely ("no text blocks"). Give them
            # 16384. (opus-5 probe 2026-07-25: a 16-token request returned ONLY
            # a ThinkingBlock — same family behavior.)
            _THINKING_PREFIXES = ("claude-fable", "claude-sonnet-5", "claude-opus-5")
            # Fable's adaptive thinking is ALWAYS on and shares max_tokens with
            # the answer — give it headroom (128K max output on the 5-series).
            claude_params = {
                "model": model,
                "max_tokens": (32768 if model.startswith("claude-fable")
                               else 16384 if model.startswith(_THINKING_PREFIXES) else 8192),
                "messages": [{"role": "user", "content": prompt}],
            }
            _TEMP_OK_PREFIXES = ("claude-sonnet-4", "claude-haiku-4",
                                 "claude-opus-4-1", "claude-opus-4-5", "claude-opus-4-6")
            if model.startswith(_TEMP_OK_PREFIXES):
                claude_params["temperature"] = temperature
            _effort = effort or MODEL_EFFORT.get(model)
            if _effort:
                claude_params["output_config"] = {"effort": _effort}
            # xhigh/max think far longer, and max_tokens is a hard cap on
            # thinking + answer together: Anthropic's migration guide says to
            # start at 64K there. A budget that size must also STREAM — a
            # non-streaming call would be cut off by the 300s client timeout
            # mid-thought and retried twice, paying three times for a CV that
            # never arrives. (The SDK refuses such requests outright on a
            # default-timeout client; ours has a custom timeout, so it would
            # fail silently instead.) Streaming times out per chunk, not per
            # request, and get_final_message() returns the same Message.
            if _effort in ("xhigh", "max"):
                claude_params["max_tokens"] = 65536
            if claude_params["max_tokens"] >= 65536:
                with client.messages.stream(**claude_params) as _stream:
                    message = _stream.get_final_message()
            else:
                message = client.messages.create(**claude_params)

            if not message:
                logging.error("Anthropic returned no message.")
                return ""
            # Surface real token usage to callers that asked for it (LangFuse
            # observability) — BEFORE any early return, so a refused call's
            # spend is still counted. Anthropic-only: the other providers fall
            # back to caller-side estimates.
            _in = _out = 0
            if getattr(message, "usage", None):
                _in = int(message.usage.input_tokens or 0)
                _out = int(message.usage.output_tokens or 0)
                if usage_out is not None:
                    usage_out["input_tokens"], usage_out["output_tokens"] = _in, _out

            # Refusal is checked FIRST. Classifiers usually decline before any
            # output (empty content), which the empty-content check below would
            # otherwise log as a generic "EMPTY response" and lose the category;
            # a decline mid-answer leaves partial text that must never be
            # returned as if it were complete (a half cover letter saved as ready).
            if getattr(message, "stop_reason", None) == "refusal":
                _category = getattr(getattr(message, "stop_details", None), "category", None)
                _chain = FALLBACK_CHAIN.get(model, []) if _allow_fallback else []
                if _category == "reasoning_extraction":
                    _chain = []   # Anthropic: do not retry these on another model
                logging.warning(
                    f"call_llm({model}): stop_reason=refusal (category={_category}) — "
                    + (f"retrying on {_chain}" if _chain else "no fallback, returning empty"))
                _answer = ""
                for _fb in _chain:
                    # one level deep: the chains contain cycles (sonnet-5 <-> opus-5-5)
                    _fb_usage: dict = {}
                    _answer = call_llm(prompt, model=_fb, temperature=temperature,
                                       usage_out=_fb_usage, _allow_fallback=False)
                    _in += int(_fb_usage.get("input_tokens") or 0)   # every attempt is billed
                    _out += int(_fb_usage.get("output_tokens") or 0)
                    if _answer:
                        logging.warning(f"call_llm({model}): refusal answered by {_fb}")
                        break
                if usage_out is not None:
                    usage_out["input_tokens"], usage_out["output_tokens"] = _in, _out
                return _answer

            if not message.content:
                logging.error("Anthropic returned EMPTY response.")
                return ""
            if getattr(message, "stop_reason", None) == "max_tokens":
                # Still returned (a partial Stage-2 batch keeps the verdicts it
                # did write; the missing ones retry), but loud: for a document it
                # means the text is cut off. Never observed since this check
                # landed on 2026-09-01; xhigh now gets a 64K budget.
                logging.error(
                    f"call_llm({model}): output hit max_tokens={claude_params['max_tokens']} — "
                    "answer is TRUNCATED (thinking shares the budget)")

            # Newer models (fable-5/sonnet-5) may emit a ThinkingBlock FIRST —
            # content[0].text raised AttributeError and silently killed every
            # call. Join the TEXT blocks only.
            texts = [b.text for b in message.content
                     if getattr(b, "type", None) == "text" and getattr(b, "text", None)]
            if not texts:
                logging.error("Anthropic returned no text blocks (only thinking?).")
                return ""
            return "\n".join(texts).strip()

        except Exception as e:
            logging.error(f"call_llm FAILED ({model}): {e}")
            return ""

    # === OPENAI (default) ===
    try:
        client = get_openai_client()
        if not client:
            logging.error("OpenAI Client is NOT initialized (check OPENAI_API_KEY).")
            return ""

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        }

        # Models that don't support custom temperature
        no_temp_prefixes = ("o1-", "o3", "o4-", "gpt-5")
        if not any(model.startswith(p) for p in no_temp_prefixes):
            params["temperature"] = temperature

        # GPT-5+ and o-series use max_completion_tokens instead of max_tokens
        if model.startswith("gpt-5") or model.startswith("o"):
            params["max_completion_tokens"] = 8192
        else:
            params["max_tokens"] = 8192

        response = client.chat.completions.create(**params)

        if not response or not response.choices:
            logging.error("OpenAI returned EMPTY response.")
            return ""

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"call_llm FAILED ({model}): {e}")
        return ""


# Resolve {{profile.*}} markers once, at import.
DEFAULT_PROMPT_TEMPLATE = fill(DEFAULT_PROMPT_TEMPLATE)
STAGE2_PROMPT_CLAUDE = fill(STAGE2_PROMPT_CLAUDE)

def load_profile_text(path: str) -> str:
    """Load text from PDF profile."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logging.error(f"Failed to load profile PDF {path}: {e}")
        return ""

def batch_job_relevance_check(df: pd.DataFrame, profile_text: str, model: str = DEFAULT_FILTER_MODEL, batch_size: int = 10, progress_callback=None, custom_prompt_template: str = None, cascade_config: dict = None, db=None) -> pd.DataFrame:
    """
    Evaluate job relevance using AI filtering.

    Priority order:
      1. CascadeFilter (if cascade_config is provided and enabled)
      2. JobMatcher (structured two-stage matching)
      3. Legacy batch text-parsing (fallback)
    """
    if df.empty:
        return df

    # Coordinator / assistant / trainee titles sit below the target level:
    # decided on the title, BEFORE any LLM call, on every path.
    # The rejected rows come back with the standard reject columns so the DB
    # row says why.
    # Director / Head-of / VP / Chief titles (above the target level) likewise.
    from .utils import split_above_level_titles, split_below_level_titles
    df, below_df = split_below_level_titles(df)
    df, above_df = split_above_level_titles(df)
    result = pd.DataFrame() if df.empty else _relevance_check_inner(
        df, profile_text, model, batch_size, progress_callback, custom_prompt_template,
        cascade_config, db)
    if below_df.empty and above_df.empty:
        return result
    return pd.concat([p for p in (result, below_df, above_df) if p is not None and not p.empty],
                     ignore_index=True)


def _relevance_check_inner(df: pd.DataFrame, profile_text: str, model: str, batch_size: int,
                           progress_callback, custom_prompt_template: str, cascade_config: dict,
                           db) -> pd.DataFrame:
    # --- Cascade Filter (primary when enabled) ---
    if cascade_config and cascade_config.get("enabled"):
        try:
            from .cascade_filter import CascadeFilter
            cf = CascadeFilter(
                profile_text=profile_text,
                stage1_model=cascade_config.get("stage1_model"),
                stage2_model=cascade_config.get("stage2_model"),
                stage3_model=cascade_config.get("stage3_model"),
                stage1_prompt=cascade_config.get("stage1_prompt"),
                stage2_prompt=cascade_config.get("stage2_prompt") or custom_prompt_template,
                stage3_prompt=cascade_config.get("stage3_prompt"),
                progress_callback=progress_callback,
                stop_check=cascade_config.get("stop_check"),
                db=db,
            )
            result = cf.filter_jobs(df)
            # Write stats back onto cascade_config so the UI can read them
            cascade_config["stats"] = cf.get_stats_for_ui()
            return result
        except Exception as e:
            # Do NOT silently fall back to JobMatcher: by this point Stage 1/2
            # may have completed PAID API calls — a silent re-filter discards
            # that work and re-bills everything. Preserve stats and re-raise;
            # main.py alerts + aborts the run loudly.
            try:
                cascade_config["stats"] = cf.get_stats_for_ui()
            except Exception:
                pass
            logging.error(f"CascadeFilter failed (NOT falling back — would re-bill all grading): {e}")
            raise

    # --- JobMatcher (structured two-stage matching) ---
    try:
        from .job_matcher import JobMatcher
        matcher = JobMatcher(profile_text, model, progress_callback=progress_callback)
        return matcher.match_jobs(df)
    except Exception as e:
        logging.warning(f"JobMatcher failed, falling back to legacy filter: {e}")
        return _legacy_batch_relevance_check(df, profile_text, model, batch_size, progress_callback, custom_prompt_template)


def _legacy_batch_relevance_check(df: pd.DataFrame, profile_text: str, model: str = DEFAULT_FILTER_MODEL, batch_size: int = 10, progress_callback=None, custom_prompt_template: str = None) -> pd.DataFrame:
    """
    Legacy text-parsing batch filter. Used as fallback if JobMatcher fails.
    """
    df = df.copy()
    relevance_results = []
    reasoning_results = []

    logging.info(f"Batch evaluating {len(df)} jobs with {model} (legacy mode)...")

    iterator = range(0, len(df), batch_size)
    total_batches = (len(df) + batch_size - 1) // batch_size

    if progress_callback:
        iter_wrapper = iterator
    else:
        try:
            from rich.progress import track
            iter_wrapper = track(iterator, description="[magenta]AI Filtering...", total=total_batches)
        except ImportError:
            iter_wrapper = iterator

    for batch_idx, i in enumerate(iter_wrapper):
        if progress_callback:
            batch_progress = (batch_idx / max(total_batches, 1))
            overall_progress = 0.80 + (batch_progress * 0.10)
            progress_callback(f"AI Filtering: Batch {batch_idx + 1}/{total_batches}", overall_progress)

        batch = df.iloc[i : i + batch_size]

        jobs_block = ""
        for idx, job in enumerate(batch.to_dict("records"), 1):
            # Posting text is data: AI-directed / applicant-instruction
            # sentences are stripped before the model reads it (jd_sanitizer).
            raw_desc = job.get('description', '')
            desc = clean_jd(raw_desc if isinstance(raw_desc, str) else "")[:TRUNCATION_JOB_DESC_FILTER]
            jobs_block += f"""
Job {idx}:
Title: {job.get('title', '')}
Company: {job.get('company', '')}
Location: {job.get('location', '')}
Date: {job.get('date_posted', 'Unknown')}
Description: {desc}
---
"""

        prompt_template = custom_prompt_template if custom_prompt_template else DEFAULT_PROMPT_TEMPLATE

        try:
            prompt = prompt_template.replace("{date}", datetime.now().strftime('%Y-%m-%d')) \
                                    .replace("{profile_text}", profile_text) \
                                    .replace("{jobs_block}", jobs_block)
        except Exception as e:
            logging.error(f"Prompt formatting failed: {e}")
            prompt = DEFAULT_PROMPT_TEMPLATE.replace("{date}", datetime.now().strftime('%Y-%m-%d')) \
                                            .replace("{profile_text}", profile_text) \
                                            .replace("{jobs_block}", jobs_block)
        try:
            answer = call_llm(prompt, model=model)

            logging.debug(f"RAW LLM RESPONSE (Batch {batch_idx+1}):\n{answer}\n{'-'*20}")

            batch_verdicts = ["irrelevant"] * len(batch)
            batch_reasons = ["Parse error"] * len(batch)

            for line in answer.splitlines():
                if "job" in line.lower() and ":" in line:
                    try:
                        parts = line.split(":", 1)
                        job_num_match = re.search(r"Job\s+(\d+)", parts[0], re.IGNORECASE)
                        if job_num_match:
                            idx_in_batch = int(job_num_match.group(1)) - 1
                            if 0 <= idx_in_batch < len(batch):
                                content = parts[1].strip()
                                upper_content = content.upper()
                                if ("HIGH MATCH" in upper_content or "POTENTIAL MATCH" in upper_content
                                        or "WORTH A LOOK" in upper_content):
                                    batch_verdicts[idx_in_batch] = "relevant"
                                else:
                                    batch_verdicts[idx_in_batch] = "irrelevant"
                                batch_reasons[idx_in_batch] = content
                    except Exception:
                        pass

            relevance_results.extend(batch_verdicts)
            reasoning_results.extend(batch_reasons)

        except Exception as e:
            logging.error(f"Batch failed: {e}")
            relevance_results.extend(["irrelevant"] * len(batch))
            reasoning_results.extend([f"Error: {e}"] * len(batch))

    df["gpt_relevance"] = relevance_results
    df["gpt_reasoning"] = reasoning_results
    # Add placeholder structured columns for compatibility
    if "match_level" not in df.columns:
        df["match_level"] = df["gpt_relevance"].apply(
            lambda r: "POTENTIAL_MATCH" if r == "relevant" else "IRRELEVANT"
        )
    if "confidence" not in df.columns:
        df["confidence"] = df["gpt_relevance"].apply(lambda r: 0.7 if r == "relevant" else 0.3)
    if "reason" not in df.columns:
        df["reason"] = df["gpt_reasoning"]
    if "rejection_rule" not in df.columns:
        df["rejection_rule"] = "NONE"
    return df
