# rag-eval-lab

**A reproducible retrieval benchmark with a negative result: on specialist legal text, a general-purpose cross-encoder reranker makes retrieval *worse* — by 8.7 to 16.3 nDCG points, p ≤ 0.0002.**

[![CI](https://github.com/Bruhadev45/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruhadev45/rag-eval-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![The reranker penalty](results/figures/reranker_penalty.png)

Every number in this repository — every table cell, figure, confidence interval, and p-value — is recomputed from committed per-query records by one command:

```bash
make reproduce
```

No API keys, no model downloads, no corpus. The analysis is deterministic (seeded bootstrap and permutation tests), so your run produces byte-identical results to CI's.

## The setup

[JurisGPT](https://github.com/Bruhadev45/Juris-GPT) is a citation-grounded RAG assistant over 47,867 Indian statutory documents, published as an IEEE paper. Its retrieval was evaluated on a **120-query human-verified benchmark** (Fleiss' κ = 0.81 among annotators) across six legal categories, in seven configurations:

| Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| Lexical (token coverage) | 0.8250 | 0.8633 | 0.8878 | 0.7849 |
| **Hybrid BM25 + lexical (RRF)** | **0.8417** | 0.7233 | **0.9531** | 0.6846 |
| Hybrid + cross-encoder | 0.8375 | 0.7733 | 0.8639 | 0.5974 |
| Dense MiniLM (subset index) | 0.4417 | 0.9250 | 0.9250 | 0.5661 |
| Dense MiniLM + cross-encoder | 0.4417 | 0.6167 | 0.7583 | 0.4032 |
| Dense MiniLM (full corpus) | 0.6792 | 0.9067 | 0.9333 | 0.6771 |
| Dense InLegalBERT | 0.4833 | 0.8883 | 0.8982 | 0.7800 |

This repository ships the raw per-query records for all seven runs (`data/runs/`, 7 × 120 rows), a wider 204-query re-run of five of them (`data/runs-ext204/`), and the analysis that turns them into claims.

## Finding 1 — the reranker penalty (significant)

Adding a MS MARCO cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of retrieval **lowered nDCG@5 in every configuration tested**:

| Comparison | Δ nDCG@5 | 95% CI | p |
|---|---|---|---|
| Hybrid → Hybrid + reranker | **−0.0871** | [−0.129, −0.046] | 0.0002 |
| Dense → Dense + reranker | **−0.1629** | [−0.187, −0.139] | 0.0001 |

Paired bootstrap CIs and two-sided sign-flip permutation tests, 10,000 resamples, both configurations answering the same 120 queries.

The damage is not an outlier effect. Per query, the reranker **hurt 75 of 120 queries and helped 36**:

![Per-query deltas](results/figures/per_query_deltas.png)

Why this matters: "add a reranker" is default advice in RAG engineering. On a specialist corpus it transferred negatively — a model trained on web-search relevance does not know that *Section 149* answering a board-composition question matters more than a passage that merely shares vocabulary. Retrieval quality on specialist corpora is **corpus-bound, not retriever-bound**. Measure before you stack components.

## Finding 2 — what hybrid fusion actually buys (and what it doesn't)

Fusing BM25 with lexical retrieval via weighted Reciprocal Rank Fusion (the extraction of which became [ragfuse](https://github.com/Bruhadev45/ragfuse)) against the lexical baseline:

| Metric | Δ | 95% CI | p | Verdict |
|---|---|---|---|---|
| MRR | +0.0653 | [+0.023, +0.111] | 0.0025 | **significant** |
| Recall@5 | +0.0167 | [−0.013, +0.046] | 0.44 | *not significant* |

So the honest sentence is narrower than the flattering one: on this benchmark, hybrid fusion **reliably puts the first relevant statute higher** (MRR), while its recall improvement is within noise at n=120. Both facts are asserted as tests in `tests/test_analysis.py` — if the data or the analysis ever stops supporting them, CI fails. Read the next section before quoting the MRR win: it does not survive on the wider query set.

![Recall vs MRR](results/figures/recall_vs_mrr.png)

## Does the reranker penalty replicate out of domain?

Mostly yes — and the replication also costs this repository one of its other claims.

The same harness ran five of these configurations on a wider **204-query** set, shipped here as `data/runs-ext204/`. Be precise about what that set is: it is the original 120 human-verified queries **plus 84 auto-generated ones** over document types the benchmark never covered — raw statutes, case law, FAQ and clause text. The 84 were generated from the corpus, not labelled by annotators, so they are weaker labels over wider ground, not a second human-verified benchmark. The analysis therefore reports them as a separate held-out subset, because those 84 queries are the only genuinely out-of-sample evidence here.

![Reranker penalty replication](results/figures/ext204_replication.png)

**The penalty replicates.** Δ nDCG@5 from adding the cross-encoder, in every query set:

| Configuration | 120 human-verified | 204 extended | 84 held out |
|---|---|---|---|
| Hybrid + cross-encoder | −0.0871 (p = 0.0002) | −0.0904 (p = 0.0001) | −0.0951 (p = 0.0086) |
| Dense + cross-encoder | −0.1629 (p = 0.0001) | −0.1419 (p = 0.0001) | −0.1119 (p = 0.0001) |

All four extended-set comparisons are negative and significant, with effect sizes inside the original confidence intervals. On queries the original benchmark never asked, over document types it never covered, a general-purpose cross-encoder still makes this retrieval worse.

**But the damage arrives differently.** On the 120, the reranker's harm showed up in MRR (−0.0892, p = 0.0006) while recall@5 was untouched (−0.0042, p = 1.0): it demoted the *first* relevant statute. On the extended set that flips — MRR is no longer significant (−0.0477, p = 0.074) while recall@5 drops significantly (−0.0711, p = 0.0012): it pushes relevant documents out of the top 5 entirely. Same verdict, different mechanism, which is a reason to quote the nDCG number rather than any single-rank metric.

And per query on the held-out 84, the reranker **hurt 42 and helped 42** — a dead-even split whose mean is still clearly negative. It does not damage more queries out of domain; it damages them harder than it helps.

**Finding 2 does not survive.** Hybrid fusion's MRR win over the lexical baseline (+0.0653, p = 0.0025 on the 120) vanishes on the extended set (−0.0184, p = 0.38) and *reverses* on the held-out 84 (−0.1379, p = 0.0002). One plausible reading: queries auto-generated from document text reward exact token overlap, which is precisely what the lexical retriever optimises — so the query generator may be doing the work, not the retriever. Either way, the honest statement is now narrower than the one in Finding 2: hybrid fusion's ranking gain is **benchmark-bound**, demonstrated on the 120 human-verified queries and not beyond them. The test `test_hybrid_mrr_gain_does_not_replicate` keeps this section from quietly drifting back to the flattering version.

## Reproduce it

```bash
git clone https://github.com/Bruhadev45/rag-eval-lab
cd rag-eval-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make reproduce   # rewrites results/ from data/runs/ — diff should be empty
make test        # pins the published findings to the committed data
```

`make reproduce` overwrites `results/results.json`, `results/TABLES.md`, and the three figures in place. After a fresh clone the git diff is empty — that's the reproducibility check, and CI enforces it on every push.

## How the analysis works

```
data/runs/*.json          7 configs × 120 per-query records (the ground truth)
data/runs-ext204/*.json   5 configs × 204 per-query records (the extended set)
data/benchmark_queries.json   the 120 queries and their expected sources
src/rageval/loading.py    loads runs, validates stored aggregates against
                          recomputed means, aligns query ids across configs
src/rageval/stats.py      paired bootstrap CI + sign-flip permutation test,
                          seeded, stdlib-only
src/rageval/ext204.py     the extended benchmark: all-204 and held-out-84
                          comparisons, and the replication figure
src/rageval/figstyle.py   one palette and axis style for every figure
src/rageval/report.py     recomputes every table, figure, and results.json
tests/test_analysis.py    pins the published findings to the data
tests/test_ext204.py      pins the replication findings to the data
```

Design choices worth naming:

- **Aggregates are never trusted.** `loading.py` recomputes every mean from per-query rows and raises if it disagrees with the stored value beyond rounding. The inputs police themselves.
- **All comparisons are paired.** Both configs answered the same queries, so the unit of analysis is the per-query difference — far more sensitive than comparing independent means, and the honest choice on a fixed query set.
- **The findings are tests.** `test_reranker_penalty_on_hybrid_is_8_7_points` is a real test name. A refactor that moved the number would fail, not silently rewrite the README.

## Provenance and limitations

- Runs were produced by the JurisGPT evaluation harness (repo: `data/eval/results/` there — `eval_*.json` for the 120, `ext204_eval_*.json` for the 204) in July 2026; this repository is the analysis layer over their committed outputs, so the retrieval systems themselves are not re-executed here.
- n=120 with human-verified labels is respectable for a domain benchmark and still small: effects under ~3 points routinely land inside the CI, as the recall result shows.
- Relevance is binary. Graded labels would sharpen nDCG.
- The 84 held-out queries are auto-generated from the corpus, not human-verified. They widen the document types under test, not the labelling standard — which is why they are reported separately rather than pooled into the headline.
- One corpus, one language pair (English-language Indian statutes). The reranker result is a warning about *unmeasured transfer*, not a claim that rerankers are bad.

## License

MIT — see [LICENSE](LICENSE). The benchmark queries and per-query records come from the author's own JurisGPT evaluation.
