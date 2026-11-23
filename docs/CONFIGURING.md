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
| `primary_metro_terms` | lower-case place names that classify a posting as "in the primary metro". Write a town whose name also exists elsewhere with its state: `vancouver, wa` |
| `secondary_metro_label` | the location of the second scrape pass (switch the pass off with `SECONDARY_SECOND_PASS` in `config.py`) |
| `secondary_metro_terms` | place names that classify a posting as "in the secondary metro": still shown, but capped at the lower tier and held to the higher pay floor |
| `allowed` | your cities, your state's name and abbreviation, and `remote` / `hybrid` if you want them (the API's `/api/config` lists them). A US state named here (`oregon` / `or`) is also a state where a bare metro term counts — list the state of a secondary metro that sits across a state line |
| `state_code`, `state_name` | your home state (`wa` / `washington`). On-site postings elsewhere in it pass to the judge; other states are dropped; a bare label for it ("Washington", "WA, US") is never blocked by the location feedback loop. If both are missing, the state at the end of `home` is used |
| `secondary_metro_abbrev` | a bare abbreviation that means the secondary metro on its own (`pdx` for a Portland metro). Leave it empty when there is none; only an exact match counts, so a location that merely contains it ("PDX Airport, OR") is not read as the metro |

How a location is read (`classify_location` in `config.py`):

1. a primary-metro term → top priority (`primary`);
2. remote, unless pinned to a non-US region → actionable (`remote`);
3. a secondary-metro term, on-site → shown, capped at the lower tier (`secondary`);
4. elsewhere in your home state → passed to the judge (`home_state`);
5. another US state or another country → dropped (`reject`);
6. a bare national label or nothing → passed to the judge (`unknown`).

A term without a state (`portland`) counts unless the location names a US
state that is neither your home state nor one listed in `allowed`
("Portland, ME"), or another country ("Kirkland, QC, Canada"). A term with a
state (`vancouver, wa`) counts only when the location names that state, by
code or by name ("Vancouver, WA", "Vancouver, Washington"). Write every town
whose name also exists elsewhere in that form. In a multi-location posting
each option is read on its own, and any in-area option makes the posting
actionable. Check a handful of real locations from your market before you
trust the gate:

```bash
PYTHONPATH=src python -c "from job_agent.config import classify_location as c; print(c('Tacoma, WA'), c('Portland, OR'), c('Spokane, WA'), c('Remote'), c('Austin, TX'))"
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

`comp_floor_yearly` is the yearly pay floor; `comp_floor_secondary_metro`
is the higher floor for on-site roles in the secondary metro. A profile that
leaves one out gets the default from `src/job_agent/comp_utils.py` ($50,000
and $100,000). `comp_floor_note` is free text for you; nothing reads it. The
rule itself: a role whose pay *ceiling* is under the floor is removed from
the feed; unstated pay is never penalised; on-site roles in the secondary
metro must clear the higher floor.

### `search.apple_target_band`

Apple publishes an internal level on its job pages (IC2 entry, IC3 mid, IC4
senior, IC5 staff; MGR is the people-manager ladder). `{low: 3, high: 4}` keeps
a level inside the band, caps one level above it at "worth a look", and rejects
anything lower, two or more levels above, or on the MGR ladder. The key is
optional and has no default: leave it out and the Apple level rule is off. No
posting is then capped or rejected for its Apple level, and the level line the
judges read names no band (the digest and the web UI still show the level). A
malformed band (not whole numbers, or not `1 <= low <= high <= 9`) raises an
error that names the key.

### `search.builtin_sites`

[Built In](https://builtin.com) runs one job site per metro
(`https://www.builtinseattle.com`, `https://www.builtinnyc.com`, ...). List the
ones for your market as https base URLs:

```yaml
search:
  builtin_sites:
  - https://www.builtinseattle.com
```

Check that each address opens in a browser before you add it: not every metro
has a site. The key is optional and has no default: leave it out and the Built
In scraper is off (it makes no request and logs one line saying so). Each entry
must be an https URL on a Built In host, with no path; anything else raises an
error that names the key. A card's location is the first of your
`primary_metro_terms`, then `secondary_metro_terms` (the town of a term written
with its state), then "Remote" or "Hybrid", that the card mentions.

### `search.glassdoor_city_id`

Glassdoor ignores a typed location: a search for "Video Producer" in
"Seattle, WA" returns the nationwide results. Only Glassdoor's own location id
narrows a search to one place. To find your city's id, run a job search for
your city on glassdoor.com and look at the address of the results page: it
carries `locT=C&locId=<number>` (`locT` is the kind of place, `C` for a city;
`locId` is the id). Write them into the profile:

```yaml
search:
  glassdoor_city_id: {type: C, id: <the locId number>}
```

With the key, every Glassdoor keyword is searched twice: in your home city
(`search.locations.home`), then nationwide, the search that brings remote
roles and roles in other metros (the location rules above sort them). Without
it, Glassdoor is searched nationwide only, and nothing is logged about it. The
key is optional and has no default. `id` must be a positive whole number and
`type` (default `C`) a single letter; anything else raises an error that names
the key.

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

## `letter_template`

The fallback cover letter: it ships when the letter check cannot run, when a
posting has no text, or when too little of the model's letter survives the
checks. Optional; without it the fallback is a short, number-free letter.

| Key | What it is |
|---|---|
| `opening` | the first sentence after "I am writing to apply for ..." |
| `proofs` | `{name: sentence}`: one past-tense proof per name |
| `proofs_by_domain` | `{domain: [2-3 proof names]}`, strongest first; a domain not listed uses `producer_core`'s |
| `fallback_proof` | optional number-free sentence used when no proof survives |

Every number in these sentences must restate `metrics_text`: each sentence is
checked against it when the letter is built, and a failing one is dropped (with
a warning in the log). Name no employers here; the letter is about the work.

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
