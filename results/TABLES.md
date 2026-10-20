# Results

Every number below is recomputed from `data/runs/*.json` by
`python -m rageval.report`. If you edit anything by hand, the next run
will overwrite it.

## Aggregate metrics (120 queries)

| Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| Lexical (token coverage) | 0.8250 | 0.8633 | 0.8878 | 0.7849 |
| Hybrid BM25 + lexical (RRF) | 0.8417 | 0.7233 | 0.9531 | 0.6846 |
| Hybrid + cross-encoder | 0.8375 | 0.7733 | 0.8639 | 0.5974 |
| Dense MiniLM (subset index) | 0.4417 | 0.9250 | 0.9250 | 0.5661 |
| Dense MiniLM + cross-encoder | 0.4417 | 0.6167 | 0.7583 | 0.4032 |
| Dense MiniLM (full corpus) | 0.6792 | 0.9067 | 0.9333 | 0.6771 |
| Dense InLegalBERT | 0.4833 | 0.8883 | 0.8982 | 0.7800 |

## Paired comparisons

Paired bootstrap 95% CI and two-sided sign-flip permutation p-value,
10,000 resamples each, seed 20260831. Positive Δ means the variant
beat its baseline.

| Baseline → Variant | Metric | Δ | 95% CI | p | Significant |
|---|---|---|---|---|---|
| Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | recall_at_5 | +0.0167 | [-0.0125, +0.0458] | 0.4380 | no |
| Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | precision_at_5 | -0.1400 | [-0.1925, -0.0911] | 0.0001 | **yes** |
| Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | mrr | +0.0653 | [+0.0225, +0.1106] | 0.0025 | **yes** |
| Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | ndcg_at_5 | -0.1003 | [-0.1435, -0.0577] | 0.0001 | **yes** |
| Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | recall_at_5 | -0.0042 | [-0.0458, +0.0333] | 1.0000 | no |
| Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | precision_at_5 | +0.0500 | [-0.0122, +0.1113] | 0.1199 | no |
| Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | mrr | -0.0892 | [-0.1444, -0.0369] | 0.0006 | **yes** |
| Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | ndcg_at_5 | -0.0871 | [-0.1287, -0.0458] | 0.0002 | **yes** |
| Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | recall_at_5 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | precision_at_5 | -0.3083 | [-0.3817, -0.2367] | 0.0001 | **yes** |
| Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | mrr | -0.1667 | [-0.2333, -0.1000] | 0.0001 | **yes** |
| Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | ndcg_at_5 | -0.1629 | [-0.1869, -0.1393] | 0.0001 | **yes** |
| Dense MiniLM (subset index) → Dense MiniLM (full corpus) | recall_at_5 | +0.2375 | [+0.1667, +0.3083] | 0.0001 | **yes** |
| Dense MiniLM (subset index) → Dense MiniLM (full corpus) | precision_at_5 | -0.0183 | [-0.0583, +0.0233] | 0.4373 | no |
| Dense MiniLM (subset index) → Dense MiniLM (full corpus) | mrr | +0.0083 | [-0.0354, +0.0521] | 0.7858 | no |
| Dense MiniLM (subset index) → Dense MiniLM (full corpus) | ndcg_at_5 | +0.1110 | [+0.0736, +0.1493] | 0.0001 | **yes** |
| Dense MiniLM (full corpus) → Dense InLegalBERT | recall_at_5 | -0.1958 | [-0.2750, -0.1125] | 0.0001 | **yes** |
| Dense MiniLM (full corpus) → Dense InLegalBERT | precision_at_5 | -0.0183 | [-0.0750, +0.0350] | 0.5588 | no |
| Dense MiniLM (full corpus) → Dense InLegalBERT | mrr | -0.0351 | [-0.0918, +0.0187] | 0.2410 | no |
| Dense MiniLM (full corpus) → Dense InLegalBERT | ndcg_at_5 | +0.1029 | [+0.0470, +0.1567] | 0.0005 | **yes** |
| Dense MiniLM (full corpus) → Hybrid BM25 + lexical (RRF) | recall_at_5 | +0.1625 | [+0.1083, +0.2250] | 0.0001 | **yes** |
| Dense MiniLM (full corpus) → Hybrid BM25 + lexical (RRF) | precision_at_5 | -0.1833 | [-0.2376, -0.1297] | 0.0001 | **yes** |
| Dense MiniLM (full corpus) → Hybrid BM25 + lexical (RRF) | mrr | +0.0197 | [-0.0133, +0.0549] | 0.2690 | no |
| Dense MiniLM (full corpus) → Hybrid BM25 + lexical (RRF) | ndcg_at_5 | +0.0074 | [-0.0379, +0.0535] | 0.7556 | no |

## Extended benchmark (204 queries)

The 204-query set contains the original 120 plus 84
auto-generated queries. The auto-generated ones were not verified by
annotators, so treat them as weaker labels over wider ground.

### Aggregate metrics (204 queries)

| Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| Lexical (token coverage) | 0.8971 | 0.8314 | 0.9301 | 0.7706 |
| Hybrid BM25 + lexical (RRF) | 0.9069 | 0.6538 | 0.9117 | 0.6141 |
| Hybrid + cross-encoder | 0.8358 | 0.6924 | 0.8640 | 0.5237 |
| Dense MiniLM (subset index) | 0.4559 | 0.8578 | 0.8578 | 0.5186 |
| Dense MiniLM + cross-encoder | 0.4559 | 0.5922 | 0.7010 | 0.3767 |

### Paired comparisons on the extended set

`ext204_all` is all 204 queries; `ext204_held_out` is only the
84 queries absent from the original benchmark.

| Query set | Baseline → Variant | Metric | Δ | 95% CI | p | Significant |
|---|---|---|---|---|---|---|
| ext204_all (204) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | recall_at_5 | +0.0098 | [-0.0074, +0.0294] | 0.4390 | no |
| ext204_all (204) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | precision_at_5 | -0.1776 | [-0.2203, -0.1357] | 0.0001 | **yes** |
| ext204_all (204) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | mrr | -0.0184 | [-0.0587, +0.0225] | 0.3840 | no |
| ext204_all (204) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | ndcg_at_5 | -0.1565 | [-0.1924, -0.1220] | 0.0001 | **yes** |
| ext204_all (204) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | recall_at_5 | -0.0711 | [-0.1127, -0.0294] | 0.0012 | **yes** |
| ext204_all (204) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | precision_at_5 | +0.0386 | [-0.0161, +0.0935] | 0.1656 | no |
| ext204_all (204) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | mrr | -0.0477 | [-0.0980, +0.0032] | 0.0736 | no |
| ext204_all (204) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | ndcg_at_5 | -0.0904 | [-0.1275, -0.0526] | 0.0001 | **yes** |
| ext204_all (204) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | recall_at_5 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| ext204_all (204) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | precision_at_5 | -0.2657 | [-0.3196, -0.2118] | 0.0001 | **yes** |
| ext204_all (204) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | mrr | -0.1569 | [-0.2059, -0.1078] | 0.0001 | **yes** |
| ext204_all (204) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | ndcg_at_5 | -0.1419 | [-0.1614, -0.1222] | 0.0001 | **yes** |
| ext204_held_out (84) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | recall_at_5 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| ext204_held_out (84) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | precision_at_5 | -0.2314 | [-0.3026, -0.1615] | 0.0001 | **yes** |
| ext204_held_out (84) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | mrr | -0.1379 | [-0.2069, -0.0718] | 0.0002 | **yes** |
| ext204_held_out (84) | Lexical (token coverage) → Hybrid BM25 + lexical (RRF) | ndcg_at_5 | -0.2369 | [-0.2904, -0.1814] | 0.0001 | **yes** |
| ext204_held_out (84) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | recall_at_5 | -0.1667 | [-0.2500, -0.0952] | 0.0001 | **yes** |
| ext204_held_out (84) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | precision_at_5 | +0.0224 | [-0.0760, +0.1187] | 0.6496 | no |
| ext204_held_out (84) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | mrr | +0.0115 | [-0.0823, +0.1030] | 0.8190 | no |
| ext204_held_out (84) | Hybrid BM25 + lexical (RRF) → Hybrid + cross-encoder | ndcg_at_5 | -0.0951 | [-0.1644, -0.0279] | 0.0086 | **yes** |
| ext204_held_out (84) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | recall_at_5 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| ext204_held_out (84) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | precision_at_5 | -0.2048 | [-0.2881, -0.1286] | 0.0001 | **yes** |
| ext204_held_out (84) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | mrr | -0.1429 | [-0.2143, -0.0714] | 0.0005 | **yes** |
| ext204_held_out (84) | Dense MiniLM (subset index) → Dense MiniLM + cross-encoder | ndcg_at_5 | -0.1119 | [-0.1459, -0.0791] | 0.0001 | **yes** |

## Confidence calibration (120 queries)

Correctness is recall@5. ECE reads each label as the centre of one of
four equal-width bins (insufficient 0.125, low 0.375, medium 0.625,
high 0.875). Concordance needs no such reading: the probability the
label ranks the better-retrieved query of a pair higher (0.5 = chance),
with a label-shuffling permutation p-value, 10,000 resamples, seed 20260831.
Non-decreasing ignores levels with fewer than 5 queries.

### Recall@5 by confidence label

| Configuration | insufficient | low | medium | high |
|---|---|---|---|---|
| Lexical (token coverage) | — | 0.0000 (n=1) | 0.9062 (n=16) | 0.8204 (n=103) |
| Hybrid BM25 + lexical (RRF) | — | 0.8095 (n=42) | 0.8333 (n=30) | 0.8750 (n=48) |
| Hybrid + cross-encoder | — | 0.7083 (n=12) | 0.8438 (n=96) | 0.9167 (n=12) |
| Dense MiniLM (subset index) | — | — | 0.4417 (n=120) | — |
| Dense MiniLM + cross-encoder | — | 0.5000 (n=3) | 0.4391 (n=115) | 0.5000 (n=2) |
| Dense MiniLM (full corpus) | 0.5625 (n=64) | — | 0.8091 (n=55) | 1.0000 (n=1) |
| Dense InLegalBERT | 0.7143 (n=14) | 0.0000 (n=13) | 0.6731 (n=26) | 0.4552 (n=67) |

### Calibration and discrimination

| Configuration | Levels used | Non-decreasing | ECE | Overconfidence | Concordance | p |
|---|---|---|---|---|---|---|
| Lexical (token coverage) | 3 | no | 0.0875 | +0.0125 | 0.4714 | 0.3695 |
| Hybrid BM25 + lexical (RRF) | 3 | yes | 0.2042 | -0.2042 | 0.5604 | 0.2568 |
| Hybrid + cross-encoder | 3 | yes | 0.2125 | -0.2125 | 0.5607 | 0.1205 |
| Dense MiniLM (subset index) | 1 | — | 0.1833 | +0.1833 | 0.5000 | 1.0000 |
| Dense MiniLM + cross-encoder | 3 | — | 0.1875 | +0.1812 | 0.5000 | 1.0000 |
| Dense MiniLM (full corpus) | 3 | yes | 0.3187 | -0.3187 | **0.6548** | 0.0001 |
| Dense InLegalBERT | 4 | no | 0.3542 | +0.1958 | 0.4794 | 0.6003 |
