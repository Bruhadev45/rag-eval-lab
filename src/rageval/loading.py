"""Load and validate the committed evaluation runs.

Every analysis in this repository starts from the per-query records in
``data/runs/*.json``. Aggregates are always recomputed from those records --
never trusted from the file -- so any inconsistency between the two is caught
here rather than propagated into a table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

METRICS = ("recall_at_5", "precision_at_5", "mrr", "ndcg_at_5")

# Presentation order: lexical family, then dense family, weakest to strongest
# claim. Every table and figure uses this order so they read consistently.
CONFIG_ORDER = (
    "baseline_lexical",
    "hybrid_bm25",
    "hybrid_bm25_rerank",
    "dense_minilm",
    "dense_minilm_rerank",
    "dense_minilm_full",
    "dense_inlegalbert",
)

LABELS = {
    "baseline_lexical": "Lexical (token coverage)",
    "hybrid_bm25": "Hybrid BM25 + lexical (RRF)",
    "hybrid_bm25_rerank": "Hybrid + cross-encoder",
    "dense_minilm": "Dense MiniLM (subset index)",
    "dense_minilm_rerank": "Dense MiniLM + cross-encoder",
    "dense_minilm_full": "Dense MiniLM (full corpus)",
    "dense_inlegalbert": "Dense InLegalBERT",
}


@dataclass(frozen=True)
class Run:
    """One evaluation run: a config name and its 120 per-query records."""

    name: str
    description: str
    timestamp: str
    per_query: tuple[dict, ...]

    def metric(self, metric: str) -> list[float]:
        """Per-query values for one metric, in benchmark query order."""
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
        return [float(row[metric]) for row in self.per_query]

    def mean(self, metric: str) -> float:
        values = self.metric(metric)
        return sum(values) / len(values)


def load_runs(runs_dir: Path) -> dict[str, Run]:
    """Load every run, validate it, and align per-query rows across configs.

    Raises:
        FileNotFoundError: If an expected run file is missing.
        ValueError: If a run has the wrong query count, query ids that do not
            match the other runs, or a stored aggregate that disagrees with
            the recomputed mean by more than rounding error.
    """
    runs: dict[str, Run] = {}
    reference_ids: list[str] | None = None

    for name in CONFIG_ORDER:
        path = runs_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"expected run file is missing: {path}")
        raw = json.loads(path.read_text())

        per_query = sorted(raw["per_query"], key=lambda row: row["query_id"])
        ids = [row["query_id"] for row in per_query]

        if len(per_query) != raw["total_queries"]:
            raise ValueError(
                f"{name}: file claims {raw['total_queries']} queries "
                f"but carries {len(per_query)} per-query rows"
            )
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(
                f"{name}: query ids do not match the other runs; "
                "paired tests would be comparing different queries"
            )

        run = Run(
            name=name,
            description=raw["config"].get("description", ""),
            timestamp=raw["timestamp"],
            per_query=tuple(per_query),
        )

        # The stored aggregate must be reproducible from the rows we ship.
        for metric in METRICS:
            stored = float(raw["aggregate"][metric])
            recomputed = run.mean(metric)
            if abs(stored - recomputed) > 5e-4:
                raise ValueError(
                    f"{name}: stored {metric}={stored:.4f} but per-query rows "
                    f"give {recomputed:.4f}; the file is inconsistent"
                )

        runs[name] = run

    return runs
