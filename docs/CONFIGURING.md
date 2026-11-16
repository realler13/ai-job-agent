# Configuring the agent: the profile file

Everything the agent knows about the person who is job hunting comes from one
YAML file. This guide walks through it section by section, using the shipped
fictional example (`profiles/example.yaml`: Maya Okonkwo, a video producer in
Seattle).

## Where the profile is read from

`src/job_agent/candidate_profile.py` loads the first file that exists:

1. the path in the `JOB_AGENT_PROFILE` environment variable
   (relative paths are resolved from the repository root),
2. `profiles/profile.yaml` — your real one, gitignored,
3. `profiles/example.yaml` — the fictional example.

```bash
cp profiles/example.yaml profiles/profile.yaml
$EDITOR profiles/profile.yaml
```

The profile is loaded once, at import time. A required key that is missing
raises a `KeyError` that names the key and the file, instead of producing a
generic document for nobody. After editing the profile, restart the API
server and any running pipeline.

Keep every section of the example in your copy, even the ones you do not use
yet. Fields marked *reserved* below are part of the schema but are not read by
the code in this release.

## `candidate`

| Key | Used for |
|---|---|
| `name` | prompts (`{{profile.name}}`); the name line that `scripts/create_base_cvs.py` looks for |
| `first_name` | the greeting in the email digest |
| `headline` | one-line description in prompts (`{{profile.headline}}`) |
| `years_experience` | the canonical years figure, written like `8+ years`. The CV fact check refuses a generated summary that states a different number |
| `pronouns` | `subject` / `object` / `possessive`, so prompts read naturally in the third person. Defaults to they / them / their |
| `email` | the contact line that `scripts/create_base_cvs.py` looks for |
| `last_name`, `phone`, `website`, `note`, `network_label` | reserved |

The header of generated CVs and cover letters, and the document file names,
come from environment variables, not from this section: `CANDIDATE_NAME`,
`CANDIDATE_EMAIL`, `CANDIDATE_PHONE`, `CANDIDATE_LOCATION` (see
`env.example`).

## `paths`

| Key | Used for |
|---|---|
| `transcript` | optional interview notes (markdown), given to the CV writer as context |
| `source_bank` | optional career source bank (markdown): the highest-precedence fact source for generated text |
| `resume`, `base_cv`, `base_cv_dir` | reserved. The resume and base CV are passed on the command line (`--resume`, `--base-cv`) or set in the web UI's settings |

Both files are optional; a missing file is skipped.

## `search`

### `search.locations`

| Key | Used for |
|---|---|
| `home` | the location sent to the job boards on the main scrape pass, and `{{profile.home_location}}` in prompts |
| `primary_metro_label` | display name of your metro; the fallback location for company and salary research |
| `primary_metro_terms` | lower-case place names that classify a posting as "in the primary metro" |
| `secondary_metro_label` | the location of the second scrape pass (switch the pass off with `SECONDARY_SECOND_PASS` in `config.py`) |
| `secondary_metro_terms` | place names that classify a posting as "in the secondary metro": still shown, but capped at the lower tier and held to the higher pay floor |
| `allowed` | the pre-filter's allow-list: a posting whose location matches none of these is dropped before any model sees it. Include your cities, your state's name and abbreviation, and `remote` / `hybrid` if you want them |
| `state_code`, `state_name`, `secondary_metro_abbrev` | reserved |

**The location rules were first tuned for California metros.** The term lists
above come from your profile, but the classifier in `config.py`
(`classify_location`) keeps its original internals: the class for the primary
metro is called `bay`, the class for the secondary metro is called `la`, and
there are helper lists for California place names that also exist elsewhere,
plus tables that treat other US states and other countries as out of area.
The example profile shows the intended way to point the rules elsewhere —
Seattle as the primary metro, Portland as the secondary — but check a handful
of real locations from your market before you trust the gate:

```bash
PYTHONPATH=src python -c "from job_agent.config import classify_location as c; print(c('Tacoma, WA'), c('Portland, OR'), c('Remote'), c('Austin, TX'))"
```

If you have no secondary metro, leave `secondary_metro_terms` as an empty
list and set `SECONDARY_SECOND_PASS = False` in `config.py`.

### Titles and exclusions

| Key | Used for |
|---|---|
| `keywords_by_category` | the job titles to search for, grouped by category. The category names are also the keys of `domains` (below), so a posting found under a category gets that category's base CV |
| `exclude_phrases` | a posting whose title contains one of these is dropped in the pre-filter |
| `negative_keywords_industrial` | one hit in the title or the description drops the posting (written for titles like "Production Manager", which also exist in factories) |
| `negative_keywords_tech` | a hit in the title drops the posting; in the description it takes two different terms |
| `stale_posting_days` | postings older than this are dropped in the pre-filter. Postings with no date pass |

Every keyword is a search query against several job boards, so the list size
drives run time and cost. Start small.

### Pay floor

`comp_floor_yearly`, `comp_floor_secondary_metro` and `comp_floor_note` are
reserved: in this release the floors are the constants `COMP_FLOOR_YEARLY`
and `COMP_FLOOR_LA_ONSITE` in `src/job_agent/comp_utils.py`. Change them
there. The rule itself: a role whose pay *ceiling* is under the floor is
removed from the feed; unstated pay is never penalised; on-site roles in the
secondary metro must clear the higher floor.

## `metrics_text`

The block of verified facts that is injected into every generation prompt.
Generators may use only the numbers written here — this is the
anti-hallucination contract, so it is worth writing carefully:

- start with the canonical years figure and say that no other figure is
  allowed;
- state which role is past and which is current, so tenses come out right;
- list numbers per employer, each with the thing it measures;
- leave out anything you could not defend in an interview.

Instead of prose you can supply a `metrics` list of entries with `value`,
`context`, `employer` and `period`; it is rendered into the same block. If
both are present, `metrics_text` wins.

## `rules`

The judgement policy, in plain English. `candidate_summary` is pasted into
the grading prompts at its `{{profile.summary}}` marker, so the models read
your words directly. The other blocks have markers too, but the shipped
cascade prompts (`config.py`, `gpt_filter.py`) spell their policy out
themselves for a production-role search: when you change one of those
blocks, change the matching prompt text as well, or put its marker into the
prompt.

| Key | Marker | What to write |
|---|---|---|
| `candidate_summary` | `{{profile.summary}}` | who the candidate is, in a short paragraph: level, field, where they can work |
| `target_roles` | `{{profile.target_roles}}` | what the top tier looks like |
| `seniority_rule` | `{{profile.seniority_rule}}` | which levels are in range, which are rejected, which are capped at the lower tier |
| `reject_rules` | `{{profile.reject_rules}}` | the traps: neighbouring professions that share your job titles |
| `craft_rule` | `{{profile.craft_rule}}` | optional: specialist roles that look adjacent but are a different job |
| `industry_rule` | `{{profile.industry_rule}}` | optional: preferred industries and industry-level walls |

Three habits from the example are worth copying:

- **Tell the judge to read the description, not the title.** Titles are
  inflated and ambiguous; duties are not.
- **End every reject rule by saying the softer neighbouring rule does not
  apply.** Otherwise the model names a disqualifier and then keeps the role
  in the lower tier anyway.
- **Keep the verdict-binding paragraph.** It tells the model that a reject
  condition it named itself is a reject.

Text in single braces (`{title}`, `{description}`) is filled per posting by
the code. Do not use single braces in your rules.

Before you trust a rewritten `rules` block, run a small search
(`--limit 20`) and read the reasons the models give.

## `application_answers`

Boilerplate for application forms (name, city, work authorisation, the
voluntary-disclosure questions). The interview-prep and application helpers
read it as a dictionary; add or remove keys freely.

## `domains`

One entry per category in `search.keywords_by_category`:

```yaml
domains:
  podcast_and_audio:
    cv: base_cv_podcast_audio.docx
    description: podcast and audio content production
```

`cv` is the base CV used for postings in that category; `description` tells
the CV writer how that variant is framed. If the file does not exist, the
generic base CV you passed with `--base-cv` is used, so you can start with a
single CV and add variants later. `producer_core` is the fallback category in
the code when a posting matches no category — keep a domain with that name,
or change `resolve_cv_domain` in `config.py`.

`scripts/create_base_cvs.py` builds the `.docx` variants from a markdown file
(`base-cv-variants.md`), and `scripts/build_cv_pool.py` builds the optional
bullet pool for the `B2` content mode from them.

## Keeping your profile private

`profiles/profile.yaml`, the resume, base CVs, the database, `results/`,
`cv/`, `cl/` and `logs/` are gitignored. The profile holds your real
employers and numbers, and the database holds your application history: check
`git status` before you push a fork.
