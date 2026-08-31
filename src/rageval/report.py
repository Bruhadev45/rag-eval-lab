"""Recompute every number in the repository from the per-query records.

Run as a module::

    python -m rageval.report

Outputs, all overwritten in place so ``git diff`` shows any drift:

- ``results/results.json``   -- aggregates and paired comparisons, machine-readable
- ``results/TABLES.md``      -- the same numbers as Markdown tables
- ``results/figures/*.png``  -- the headline figures
"""

from __future__ import annotations

import json
from pathlib import Path

from .loading import CONFIG_ORDER, LABELS, METRICS, load_runs
from .stats import PairedComparison, paired_comparison

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "runs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# The comparisons the repository exists to make. Each is (baseline, variant):
# positive delta means the variant helped, negative means it hurt.
COMPARISONS = (
    ("baseline_lexical", "hybrid_bm25"),
    ("hybrid_bm25", "hybrid_bm25_rerank"),
    ("dense_minilm", "dense_minilm_rerank"),
    ("dense_minilm", "dense_minilm_full"),
    ("dense_minilm_full", "dense_inlegalbert"),
    ("dense_minilm_full", "hybrid_bm25"),
)


def build_aggregates(runs) -> dict:
    return {
        name: {metric: round(runs[name].mean(metric), 4) for metric in METRICS}
        for name in CONFIG_ORDER
    }


def build_comparisons(runs) -> list[dict]:
    rows = []
    for baseline, variant in COMPARISONS:
        for metric in METRICS:
            result: PairedComparison = paired_comparison(
                runs[baseline].metric(metric),
                runs[variant].metric(metric),
                metric=metric,
            )
            rows.append(
                {
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


def write_tables(aggregates: dict, comparisons: list[dict]) -> str:
    lines = ["# Results", ""]
    lines += [
        "Every number below is recomputed from `data/runs/*.json` by",
        "`python -m rageval.report`. If you edit anything by hand, the next run",
        "will overwrite it.",
        "",
        "## Aggregate metrics (120 queries)",
        "",
        "| Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 |",
        "|---|---|---|---|---|",
    ]
    for name in CONFIG_ORDER:
        a = aggregates[name]
        lines.append(
            f"| {LABELS[name]} | {a['recall_at_5']:.4f} | {a['precision_at_5']:.4f} "
            f"| {a['mrr']:.4f} | {a['ndcg_at_5']:.4f} |"
        )

    lines += [
        "",
        "## Paired comparisons",
        "",
        "Paired bootstrap 95% CI and two-sided sign-flip permutation p-value,",
        "10,000 resamples each, seed 20260831. Positive Δ means the variant",
        "beat its baseline.",
        "",
        "| Baseline → Variant | Metric | Δ | 95% CI | p | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for row in comparisons:
        ci = f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
        mark = "**yes**" if row["significant"] else "no"
        lines.append(
            f"| {LABELS[row['baseline']]} → {LABELS[row['variant']]} "
            f"| {row['metric']} | {row['delta']:+.4f} | {ci} | {row['p_value']:.4f} | {mark} |"
        )
    return "\n".join(lines) + "\n"


def draw_figures(runs, aggregates: dict) -> None:
    """Draw the three headline figures.

    Palette and mark specs follow a validated data-viz method: categorical
    slots blue #2a78d6 / orange #eb6834 (adjacent-pair CVD deltaE 24.7, all
    checks pass on a light surface), diverging poles blue/red for the
    per-query polarity strip, text in ink tokens rather than series colors,
    recessive grid, no top/right spines.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE = "#2a78d6"    # categorical slot 1: the un-reranked configuration
    ORANGE = "#eb6834"  # categorical slot 2: the reranked configuration
    RED = "#e34948"     # diverging negative pole: queries the reranker hurt
    INK = "#0b0b0b"
    INK2 = "#52514e"
    MUTED = "#8a8983"
    SURFACE = "#ffffff"

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.edgecolor": MUTED,
            "font.size": 10,
        }
    )

    def strip_spines(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.8)

    # Figure 1 -- grouped bars: nDCG@5 with and without the cross-encoder.
    pairs = [("hybrid_bm25", "hybrid_bm25_rerank"), ("dense_minilm", "dense_minilm_rerank")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    x = range(len(pairs))
    without = [aggregates[a]["ndcg_at_5"] for a, _ in pairs]
    with_rr = [aggregates[b]["ndcg_at_5"] for _, b in pairs]
    width = 0.34  # slim bars with a visible gap between the pair
    bars_a = ax.bar([i - width / 2 - 0.01 for i in x], without, width, color=BLUE,
                    label="Without reranker")
    bars_b = ax.bar([i + width / 2 + 0.01 for i in x], with_rr, width, color=ORANGE,
                    label="With MS MARCO cross-encoder")
    for bar in list(bars_a) + list(bars_b):  # direct value labels, ink not series color
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    for i, (a, b) in enumerate(pairs):
        drop = (aggregates[b]["ndcg_at_5"] - aggregates[a]["ndcg_at_5"]) * 100
        ax.annotate(f"{drop:+.1f} pts", xy=(i, max(without[i], with_rr[i])),
                    xytext=(0, 22), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Hybrid BM25 + lexical", "Dense MiniLM"])
    ax.set_ylabel("nDCG@5")
    ax.set_ylim(0, 0.85)
    ax.set_title("A general-purpose cross-encoder reranker hurts legal retrieval",
                 color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    strip_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reranker_penalty.png")
    plt.close(fig)

    # Figure 2 -- scatter: recall vs MRR, one emphasized entity, ink labels.
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for name in CONFIG_ORDER:
        a = aggregates[name]
        highlight = name == "hybrid_bm25"
        ax.scatter(a["recall_at_5"], a["mrr"],
                   s=120 if highlight else 64,
                   color=BLUE if highlight else MUTED,
                   zorder=3)
        ax.annotate(LABELS[name], xy=(a["recall_at_5"], a["mrr"]),
                    xytext=(7, -3), textcoords="offset points",
                    fontsize=8.5, color=INK if highlight else INK2)
    ax.set_xlabel("Recall@5")
    ax.set_ylabel("MRR")
    ax.set_title("Hybrid fusion holds the top-right corner", color=INK, fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    strip_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "recall_vs_mrr.png")
    plt.close(fig)

    # Figure 3 -- diverging strip: per-query reranker deltas around zero.
    deltas = sorted(
        b - a
        for a, b in zip(
            runs["hybrid_bm25"].metric("ndcg_at_5"),
            runs["hybrid_bm25_rerank"].metric("ndcg_at_5"),
        )
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    colors = [RED if d < 0 else BLUE for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors, width=0.9)
    ax.axhline(0, color=INK2, linewidth=0.8)
    hurt = sum(1 for d in deltas if d < 0)
    helped = sum(1 for d in deltas if d > 0)
    unchanged = len(deltas) - hurt - helped
    ax.set_title(
        f"Reranker effect per query, hybrid config \u2014 "
        f"hurt {hurt} \u00b7 helped {helped} \u00b7 unchanged {unchanged}",
        color=INK, fontsize=11, loc="left",
    )
    ax.set_xlabel("Query, sorted by delta")
    ax.set_ylabel("\u0394 nDCG@5")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    strip_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_query_deltas.png")
    plt.close(fig)


def main() -> None:
    runs = load_runs(RUNS_DIR)
    aggregates = build_aggregates(runs)
    comparisons = build_comparisons(runs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / "results.json").write_text(
        json.dumps(
            {
                "benchmark": "JurisGPT 120-query human-verified benchmark (Fleiss' κ = 0.81)",
                "seed": 20260831,
                "resamples": 10_000,
                "aggregates": aggregates,
                "paired_comparisons": comparisons,
            },
            indent=2,
        )
        + "\n"
    )
    (RESULTS_DIR / "TABLES.md").write_text(write_tables(aggregates, comparisons))
    draw_figures(runs, aggregates)

    print("wrote results/results.json")
    print("wrote results/TABLES.md")
    print("wrote results/figures/{reranker_penalty,recall_vs_mrr,per_query_deltas}.png")


if __name__ == "__main__":
    main()
