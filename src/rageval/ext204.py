"""The extended 204-query benchmark: does the reranker penalty replicate?

``data/runs-ext204/`` holds five of the seven configurations re-run on a wider
query set of 204 queries. That set is a *superset* of the 120-query benchmark:
the same 120 human-verified queries plus 84 auto-generated ones over document
types the original benchmark never touched (raw statutes, case law, FAQ and
clause text). So it buys two things, and this module computes both separately:

- **ext204_all** (204 queries) -- the headline effect on the wider set.
- **ext204_held_out** (84 queries) -- the 84 queries that are *not* in the
  original benchmark. This is the only genuinely out-of-sample evidence here,
  and it is the honest one to quote for replication.

The 84 held-out queries were generated from the corpus rather than verified by
annotators, so their labels are weaker than the 120's. That caveat belongs next
to every number this module produces.
"""

from __future__ import annotations

from pathlib import Path

from .figstyle import BLUE, INK, INK2, MUTED, ORANGE, apply_style, strip_spines
from .loading import EXT204_CONFIG_ORDER, LABELS, METRICS, Run, load_runs
from .stats import paired_comparison

# Only the comparisons both benchmarks can make: the two reranker pairs (the
# repository's headline claim) and hybrid fusion against the lexical baseline.
EXT204_COMPARISONS = (
    ("baseline_lexical", "hybrid_bm25"),
    ("hybrid_bm25", "hybrid_bm25_rerank"),
    ("dense_minilm", "dense_minilm_rerank"),
)

# The pairs the replication figure shows, top to bottom.
FIGURE_PAIRS = (
    ("hybrid_bm25", "hybrid_bm25_rerank", "Hybrid"),
    ("dense_minilm", "dense_minilm_rerank", "Dense MiniLM"),
)

ALL = "ext204_all"
HELD_OUT = "ext204_held_out"


def held_out_ids(ext_runs: dict[str, Run], runs_120: dict[str, Run]) -> tuple[str, ...]:
    """Query ids present in the extended benchmark but not in the 120."""
    original = set(next(iter(runs_120.values())).query_ids)
    return tuple(qid for qid in next(iter(ext_runs.values())).query_ids if qid not in original)


def _comparison_rows(runs: dict[str, Run], query_set: str) -> list[dict]:
    rows = []
    for baseline, variant in EXT204_COMPARISONS:
        for metric in METRICS:
            result = paired_comparison(
                runs[baseline].metric(metric),
                runs[variant].metric(metric),
                metric=metric,
            )
            rows.append(
                {
                    "query_set": query_set,
                    "baseline": baseline,
                    "variant": variant,
                    "metric": metric,
                    "baseline_mean": round(result.mean_a, 4),
                    "variant_mean": round(result.mean_b, 4),
                    "delta": round(result.delta, 4),
                    "ci95": [round(result.ci_low, 4), round(result.ci_high, 4)],
                    "p_value": round(result.p_value, 5),
                    "significant": result.significant,
                    "n_queries": result.n_queries,
                }
            )
    return rows


def build_ext204(runs_dir: Path, runs_120: dict[str, Run]) -> dict:
    """Load the extended runs and compute every ext204 number.

    Args:
        runs_dir: ``data/runs-ext204``.
        runs_120: The already-loaded 120-query runs, used to identify which
            extended queries are held out.

    Returns:
        A dict ready to serialise into ``results/results.json``.
    """
    ext_runs = load_runs(runs_dir, EXT204_CONFIG_ORDER)
    held = held_out_ids(ext_runs, runs_120)
    held_runs = {name: run.subset(held) for name, run in ext_runs.items()}

    polarity = {}
    for query_set, runs in ((ALL, ext_runs), (HELD_OUT, held_runs)):
        deltas = [
            b - a
            for a, b in zip(
                runs["hybrid_bm25"].metric("ndcg_at_5"),
                runs["hybrid_bm25_rerank"].metric("ndcg_at_5"),
            )
        ]
        polarity[query_set] = {
            "hurt": sum(1 for d in deltas if d < 0),
            "helped": sum(1 for d in deltas if d > 0),
            "unchanged": sum(1 for d in deltas if d == 0),
        }

    return {
        "description": (
            "Extended benchmark: the 120 human-verified queries plus 84 "
            "auto-generated queries over statute, case, FAQ and clause text. "
            "The held-out subset is the 84 that are not in the original 120."
        ),
        "query_sets": {
            ALL: len(ext_runs["hybrid_bm25"].per_query),
            HELD_OUT: len(held),
        },
        "aggregates": {
            name: {metric: round(ext_runs[name].mean(metric), 4) for metric in METRICS}
            for name in EXT204_CONFIG_ORDER
        },
        "reranker_polarity": polarity,
        "paired_comparisons": _comparison_rows(ext_runs, ALL)
        + _comparison_rows(held_runs, HELD_OUT),
    }


def write_ext204_tables(ext204: dict) -> str:
    """The ext204 section of ``results/TABLES.md``."""
    sets = ext204["query_sets"]
    lines = [
        "",
        f"## Extended benchmark ({sets[ALL]} queries)",
        "",
        f"The {sets[ALL]}-query set contains the original 120 plus {sets[HELD_OUT]}",
        "auto-generated queries. The auto-generated ones were not verified by",
        "annotators, so treat them as weaker labels over wider ground.",
        "",
        f"### Aggregate metrics ({sets[ALL]} queries)",
        "",
        "| Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 |",
        "|---|---|---|---|---|",
    ]
    for name in EXT204_CONFIG_ORDER:
        a = ext204["aggregates"][name]
        lines.append(
            f"| {LABELS[name]} | {a['recall_at_5']:.4f} | {a['precision_at_5']:.4f} "
            f"| {a['mrr']:.4f} | {a['ndcg_at_5']:.4f} |"
        )

    lines += [
        "",
        "### Paired comparisons on the extended set",
        "",
        f"`{ALL}` is all {sets[ALL]} queries; `{HELD_OUT}` is only the",
        f"{sets[HELD_OUT]} queries absent from the original benchmark.",
        "",
        "| Query set | Baseline → Variant | Metric | Δ | 95% CI | p | Significant |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ext204["paired_comparisons"]:
        ci = f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
        mark = "**yes**" if row["significant"] else "no"
        lines.append(
            f"| {row['query_set']} ({row['n_queries']}) "
            f"| {LABELS[row['baseline']]} → {LABELS[row['variant']]} "
            f"| {row['metric']} | {row['delta']:+.4f} | {ci} | {row['p_value']:.4f} | {mark} |"
        )
    return "\n".join(lines) + "\n"


def _ndcg_row(comparisons: list[dict], baseline: str, variant: str, query_set: str | None) -> dict:
    for row in comparisons:
        if (
            row["baseline"] == baseline
            and row["variant"] == variant
            and row["metric"] == "ndcg_at_5"
            and row.get("query_set") == query_set
        ):
            return row
    raise KeyError(f"no ndcg_at_5 row for {baseline}->{variant} on {query_set}")


def draw_ext204_figure(ext204: dict, comparisons_120: list[dict], out_path: Path) -> None:
    """Forest plot: Δ nDCG@5 from the cross-encoder on each query set.

    One row per (configuration, query set). Blue is the original 120-query
    benchmark, orange the extended set -- so replication reads as the orange
    markers landing where the blue one did, on the same side of zero.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()

    sets = ext204["query_sets"]
    entries = []  # (label, row, is_original)
    for baseline, variant, short in FIGURE_PAIRS:
        entries.append(
            (f"{short} · 120 verified", _ndcg_row(comparisons_120, baseline, variant, None), True)
        )
        entries.append(
            (
                f"{short} · {sets[ALL]} extended",
                _ndcg_row(ext204["paired_comparisons"], baseline, variant, ALL),
                False,
            )
        )
        entries.append(
            (
                f"{short} · {sets[HELD_OUT]} held out",
                _ndcg_row(ext204["paired_comparisons"], baseline, variant, HELD_OUT),
                False,
            )
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    ys = list(range(len(entries)))[::-1]  # first entry at the top
    for y, (label, row, is_original) in zip(ys, entries):
        color = BLUE if is_original else ORANGE
        low, high = row["ci95"]
        ax.plot([low, high], [y, y], color=color, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([row["delta"]], [y], s=70, color=color, zorder=3)
        ax.annotate(
            f"{row['delta']:+.3f}  p={row['p_value']:.4f}",
            xy=(high, y),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=INK2,
        )
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in entries], fontsize=9)
    ax.set_xlabel("Δ nDCG@5 from adding the cross-encoder (95% CI)")
    ax.set_xlim(-0.23, 0.10)
    ax.set_ylim(-0.9, len(entries) - 0.4)

    extended = [row for _, row, is_original in entries if not is_original]
    significant = sum(1 for row in extended if row["significant"])
    direction = "all negative" if all(row["delta"] < 0 for row in extended) else "mixed in sign"
    ax.set_title(
        "The reranker penalty on the extended benchmark\n"
        f"{significant}/{len(extended)} extended comparisons significant, {direction}",
        color=INK,
        fontsize=11.5,
        loc="left",
    )
    ax.legend(
        handles=[
            plt.Line2D([], [], color=BLUE, marker="o", linewidth=2.2, label="120 human-verified"),
            plt.Line2D([], [], color=ORANGE, marker="o", linewidth=2.2, label="Extended benchmark"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    strip_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
