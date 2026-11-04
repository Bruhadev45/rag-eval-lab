# Data card: legal-demo evaluation set (fictional)

29 labelled cases (45 queries) over six fictional legal-like sources, frozen
into train, dev and test splits **before any retrieval system or model was run
on them**. They exist to test software paths that a legal RAG system must get
right: version dates, conflicting sources, jurisdiction, tenant and matter
isolation, unknown metadata and abstention. **The sources are not law, the
labels are not legal advice, and nothing scored on this set measures legal
accuracy.** This set is separate from the historical 120/204-query benchmark
in `data/runs*/`, and nothing here changes those results.

## Files

| File | What it holds |
| --- | --- |
| `sources.json` | minirag's fictional fixture, a byte-identical copy of `examples/legal_rag/fixtures/sources.json` at minirag commit `7d45104` (see `manifest.json`). Its own [data card](https://github.com/Bruhadev45/minirag/blob/main/examples/legal_rag/DATA_CARD.md) covers the sources. |
| `cases.json` | The cases, the declared provision groups, and the label authors and methods. |
| `manifest.json` | The freeze: dataset version, a hash of each file, the per-split counts, and the usage policy. |

Check the set and regenerate its report (`results/legal-demo/`):

```bash
python -m rageval.legal_demo
```

## Who wrote the labels, and how

- **Author.** An AI coding assistant (the daily build agent) wrote every
  question, expected outcome, quote and note on 2026-11-04. Each case records
  its author as `ai-assistant`.
- **Method** (`fixture-reading`). Read only the text and metadata of
  `sources.json`, wrote each question, chose the expected outcome, and copied
  each quote verbatim from the section it cites.
- **Review.** No person has reviewed these labels yet, and no legal expert
  has. They are engineering labels for fictional text.
- **Machine checks.** `rageval.legal_demo` rechecks every case against the
  fixture each time it loads: every quote occurs exactly once in its section;
  every evidence passage is eligible and every trap is not; and each category's
  outcome and trap rules hold (below). The checks prove agreement with the
  fixture, not that a label is the best one.

## What a case holds

Each case has one or more paraphrased `queries` asked by one requester
(`scope`: tenant and matter) about one `jurisdiction` and `regime` on one
`as_of` date. It then gives the expected `outcome`, a short reference `answer`,
the `evidence` (source version, section, verbatim quote), the `traps` (passages
a correct system must not use), an `issue`, a `split` and a `note` explaining
the label.

| Category | Allowed outcomes | Extra rule |
| --- | --- | --- |
| `answerable` | `answer`, `clarify` | Evidence is eligible. |
| `unanswerable` | `abstain` | No evidence. |
| `wrong-regime` | `answer`, `abstain` | A trap is from another jurisdiction or regime. |
| `outdated` | `answer`, `abstain` | A trap is not yet, or no longer, in force on the as-of date. |
| `conflicting-source` | `report-conflict` | Evidence from at least two source families. |

`clarify` means the answer depends on a fact the question leaves out. The
expected response states that dependency instead of giving one number.

## The eligibility rule the labels assume

A source is eligible only if:

- the requester may see it: it is public, or it has the same tenant and, for a
  matter-scoped source, the same matter;
- its jurisdiction and regime are known and match the question;
- it is known to be in force on the as-of date.

Effective intervals are half-open: a version is in force on `effective_from`
and not on `effective_to`. `"unknown"` never matches, so the Quillmouth Port
Code, whose end date is unknown, is never eligible. Its cases therefore expect
an abstention.

This rule fails closed. If minirag's eligibility filter ends up with a
different rule, these labels need a new dataset version.

## How the splits are kept apart

- A **provision group** names the sections that answer one question, in every
  version. For example, the Act's inspection interval in v1 and v2 belongs with
  the guidance note that conflicts with it. Every section of the fixture is in
  exactly one group, and `cases.json` gives the reason for each.
- A **leakage group** is every case connected by a shared issue or a shared
  provision group, through evidence or traps. Each leakage group sits inside
  one split, so paraphrases, superseded and current versions, and conflicting
  sources never cross a split.
- No two queries in different splits may share half or more of their content
  words. This is a coarse screen for copied questions, not a paraphrase
  detector.
- `cases.json` assigns each split by hand, so that each split covers as many
  categories as the groups allow. The assignment was frozen before any system
  ran.

| Split | Cases (queries) | Leakage groups |
| --- | --- | --- |
| train | 7 (11) | penalties; Basaltine file note; inspection certificate |
| dev | 11 (15) | Orchidwake memo; record keeping; inspection fee |
| test | 11 (19) | inspection interval (Varnholt and Quillmouth); inspector qualification |

## Rules for use

From `manifest.json`:

- Tune only on train and dev: retriever weights, candidate depth, context
  limits and abstention thresholds.
- Score a frozen system configuration on test, and report every test run made.
- Any content change to either file needs a new `dataset_version` and a new
  manifest. Results across versions are not comparable.

## Limitations

- **Tiny and fictional.** 29 cases cannot support a quality claim, a
  significance test or a leaderboard. The set exercises code paths.
- **Unreviewed AI-written labels.** See above.
- **Uneven coverage.** The fixture has one conflict and one ambiguous-length
  question. Both sit in the interval group, so `conflicting-source` and
  `clarify` cases appear only in test, and dev cannot tune for them.
- **Shared family.** The Harbour Safety Act's sections answer different
  questions, so they sit in different splits: interval in test, record keeping
  in dev, penalties in train. A system tuned on dev could still pick up
  Act-specific vocabulary. `results/legal-demo/SPLITS.md` lists this overlap.
- **Traps are not exhaustive.** Each case lists the ineligible passages it
  means to test, not every passage on the topic.
- **The reference answers are short and English-only.** Matching them is a
  later scoring decision. A response can be correct in other words.
