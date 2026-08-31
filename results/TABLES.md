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
