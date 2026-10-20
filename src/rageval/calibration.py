"""Is the per-query confidence label worth anything?

Every run record carries a ``confidence`` label -- ``insufficient``, ``low``,
``medium`` or ``high`` -- assigned by the JurisGPT harness from retrieval
similarity thresholds before any answer is generated. This module asks whether
that label tracks *retrieval correctness*, measured as the query's recall@5
(the fraction of its expected sources found in the top 5).

Two questions, kept apart because they fail independently:

- **Calibration** -- does the label mean what it says? The label is ordinal, not
  a probability, so each level is read as the centre of one of four equal-width
  confidence bins (:data:`NOMINAL`). Expected calibration error (ECE) is then the
  size-weighted gap between that nominal value and the observed mean recall@5.
  ECE depends on this reading; the next number does not.
- **Discrimination** -- does a higher label at least go with better retrieval?
  :func:`concordance` is the probability that, of two queries where one
  retrieved more of its expected sources, the label ranks that one higher
  (label ties count half). 0.5 is a coin flip. A seeded label-shuffling
  permutation test says whether it differs from 0.5 beyond noise.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from .figstyle import BLUE, INK, INK2, MUTED, ORANGE, apply_style, strip_spines
from .loading import CONFIG_ORDER, LABELS, Run

# Ascending: a later level claims more confidence than an earlier one.
LEVELS = ("insufficient", "low", "medium", "high")

# The probability each label is read as claiming: four equal-width bins on
# [0, 1], one per level, each label at its bin's centre.
NOMINAL = {"insufficient": 0.125, "low": 0.375, "medium": 0.625, "high": 0.875}

CORRECTNESS = "recall_at_5"

# Levels holding fewer queries than this are left out of the non-decreasing
# verdict, so one query in a sparse level cannot decide it.
MIN_CELL = 5

# The two configs the reliability diagram draws: the shipped hybrid pipeline,
# and the only config whose label ranks queries better than chance.
FIGURE_CONFIGS = (("hybrid_bm25", BLUE), ("dense_minilm_full", ORANGE))


@dataclass(frozen=True)
class LevelStats:
    """The queries one run labelled with one confidence level."""

    level: str
    n: int
    nominal: float
    mean_recall: float


def _labels(run: Run) -> list[str]:
    labels = []
    for row in run.per_query:
        label = row.get("confidence")
        if label is None:
            raise ValueError(f"{run.name}: query {row.get('query_id')} has no confidence label")
        if label not in NOMINAL:
            raise ValueError(f"{run.name}: unknown confidence label {label!r}; expected {LEVELS}")
        labels.append(label)
    return labels


def reliability(run: Run) -> tuple[LevelStats, ...]:
    """Mean recall@5 per confidence level, ascending, unused levels omitted.

    Raises:
        ValueError: If a query has no label or a label outside :data:`LEVELS`.
    """
    labels = _labels(run)
    recalls = run.metric(CORRECTNESS)
    stats = []
    for level in LEVELS:
        values = [r for label, r in zip(labels, recalls) if label == level]
        if values:
            stats.append(LevelStats(level, len(values), NOMINAL[level], sum(values) / len(values)))
    return tuple(stats)


def expected_calibration_error(levels: Sequence[LevelStats]) -> float:
    """Size-weighted mean of ``|observed recall - nominal confidence|``."""
    total = sum(s.n for s in levels)
    if total == 0:
        raise ValueError("cannot compute calibration error over no queries")
    return sum(s.n * abs(s.mean_recall - s.nominal) for s in levels) / total


def overconfidence(levels: Sequence[LevelStats]) -> float:
    """Mean claimed confidence minus mean recall@5: > 0 means overconfident.

    Summed per level like :func:`expected_calibration_error`, so the two agree
    exactly in magnitude when every level's gap has the same sign.
    """
    total = sum(s.n for s in levels)
    if total == 0:
        raise ValueError("cannot compute overconfidence over no queries")
    return sum(s.n * (s.nominal - s.mean_recall) for s in levels) / total


def non_decreasing(levels: Sequence[LevelStats]) -> bool | None:
    """Whether recall never falls as the label rises.

    Only levels holding at least :data:`MIN_CELL` queries count. Returns None
    when fewer than two levels qualify, since there is nothing to order.
    """
    means = [s.mean_recall for s in levels if s.n >= MIN_CELL]
    if len(means) < 2:
        return None
    return all(a <= b for a, b in pairwise(means))


def concordance(labels: Sequence[int], outcomes: Sequence[float]) -> float:
    """P(the label ranks the better-retrieved query of a pair higher).

    Args:
        labels: Ordinal label per query (higher = more confident).
        outcomes: Correctness per query, same order.

    Raises:
        ValueError: On mismatched lengths, or if every outcome is equal (no
            pair has a better-retrieved member, so the measure is undefined).
    """
    if len(labels) != len(outcomes):
        raise ValueError(f"labels and outcomes differ in length: {len(labels)} vs {len(outcomes)}")
    # Count (outcome, label) cells once; pairs are products of cell sizes.
    cells = Counter(zip(outcomes, labels))
    agreeing = 0.0
    pairs = 0
    for (y_hi, c_hi), n_hi in cells.items():
        for (y_lo, c_lo), n_lo in cells.items():
            if y_hi <= y_lo:
                continue
            pairs += n_hi * n_lo
            if c_hi > c_lo:
                agreeing += n_hi * n_lo
            elif c_hi == c_lo:
                agreeing += 0.5 * n_hi * n_lo
    if pairs == 0:
        raise ValueError("concordance is undefined when every query has the same outcome")
    return agreeing / pairs


def concordance_test(
    labels: Sequence[int],
    outcomes: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 20260831,
) -> tuple[float, float]:
    """Observed concordance and its two-sided permutation p-value against 0.5.

    Under the null the label is unrelated to correctness, so shuffling labels
    across queries leaves the distribution unchanged.
    """
    observed = concordance(labels, outcomes)
    rng = random.Random(seed)
    shuffled = list(labels)  # a working copy; the caller's list is untouched
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        rng.shuffle(shuffled)
        if abs(concordance(shuffled, outcomes) - 0.5) >= abs(observed - 0.5) - 1e-12:
            at_least_as_extreme += 1
    return observed, (at_least_as_extreme + 1) / (n_resamples + 1)


def _config_row(run: Run) -> dict:
    levels = reliability(run)
    ranks = [LEVELS.index(label) for label in _labels(run)]
    recalls = run.metric(CORRECTNESS)
    observed, p_value = concordance_test(ranks, recalls)
    return {
        "levels": [
            {
                "level": s.level,
                "n": s.n,
                "nominal": s.nominal,
                "mean_recall_at_5": round(s.mean_recall, 4),
            }
            for s in levels
        ],
        "levels_used": len(levels),
        "non_decreasing": non_decreasing(levels),
        "ece": round(expected_calibration_error(levels), 4),
        "overconfidence": round(overconfidence(levels), 4),
        "concordance": round(observed, 4),
        "concordance_p": round(p_value, 5),
        "significant": p_value < 0.05,
    }


def build_calibration(runs: dict[str, Run]) -> dict:
    """Every calibration number, ready to serialise into ``results.json``."""
    return {
        "correctness": CORRECTNESS,
        "nominal_confidence": dict(NOMINAL),
        "configs": {name: _config_row(runs[name]) for name in CONFIG_ORDER},
    }


def write_calibration_tables(calibration: dict) -> str:
    """The calibration section of ``results/TABLES.md``."""
    lines = [
        "",
        "## Confidence calibration (120 queries)",
        "",
        "Correctness is recall@5. ECE reads each label as the centre of one of",
        "four equal-width bins (insufficient 0.125, low 0.375, medium 0.625,",
        "high 0.875). Concordance needs no such reading: the probability the",
        "label ranks the better-retrieved query of a pair higher (0.5 = chance),",
        "with a label-shuffling permutation p-value, 10,000 resamples, seed 20260831.",
        f"Non-decreasing ignores levels with fewer than {MIN_CELL} queries.",
        "",
        "### Recall@5 by confidence label",
        "",
        "| Configuration | " + " | ".join(LEVELS) + " |",
        "|---|" + "---|" * len(LEVELS),
    ]
    for name, row in calibration["configs"].items():
        by_level = {s["level"]: s for s in row["levels"]}
        cells = [
            f"{by_level[lv]['mean_recall_at_5']:.4f} (n={by_level[lv]['n']})"
            if lv in by_level
            else "—"
            for lv in LEVELS
        ]
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Calibration and discrimination",
        "",
        "| Configuration | Levels used | Non-decreasing | ECE | Overconfidence | Concordance | p |",
        "|---|---|---|---|---|---|---|",
    ]
    verdict = {True: "yes", False: "no", None: "—"}
    for name, row in calibration["configs"].items():
        mark = "**" if row["significant"] else ""
        lines.append(
            f"| {LABELS[name]} | {row['levels_used']} | {verdict[row['non_decreasing']]} "
            f"| {row['ece']:.4f} | {row['overconfidence']:+.4f} "
            f"| {mark}{row['concordance']:.4f}{mark} | {row['concordance_p']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def draw_calibration_figure(calibration: dict, out_path: Path) -> None:
    """Reliability diagram (left) beside per-config concordance (right).

    Left: observed recall@5 per label against the label's nominal confidence,
    marker area proportional to query count; the diagonal is perfect
    calibration. Right: concordance for every config against the 0.5 line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    configs = calibration["configs"]
    fig, (rel, disc) = plt.subplots(
        1, 2, figsize=(11, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1.15]}
    )

    rel.plot([0, 1], [0, 1], color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
    rel.annotate("perfectly calibrated", xy=(0.66, 0.6), rotation=38, fontsize=8.5, color=MUTED)
    for name, color in FIGURE_CONFIGS:
        levels = configs[name]["levels"]
        xs = [s["nominal"] for s in levels]
        ys = [s["mean_recall_at_5"] for s in levels]
        rel.plot(xs, ys, color=color, linewidth=1.6, zorder=2, label=LABELS[name])
        rel.scatter(xs, ys, s=[12 + 2.5 * s["n"] for s in levels], color=color, zorder=3)
        for s in levels:
            rel.annotate(
                f"n={s['n']}",
                xy=(s["nominal"], s["mean_recall_at_5"]),
                # Blue above-left, orange below-right: the two lines cross.
                xytext=(-7, 7) if color == BLUE else (7, -13),
                textcoords="offset points",
                ha="right" if color == BLUE else "left",
                fontsize=8,
                color=INK2,
            )
    rel.set_xticks([NOMINAL[level] for level in LEVELS])
    rel.set_xticklabels(LEVELS, fontsize=9)
    rel.set_xlim(0, 1)
    rel.set_ylim(0, 1.02)
    rel.set_xlabel("Confidence label, at its nominal bin centre")
    rel.set_ylabel("Observed recall@5")
    rel.set_title("Reliability: what each label delivers", color=INK, fontsize=11, loc="left")
    rel.legend(frameon=False, loc="lower right", fontsize=8.5)
    rel.grid(alpha=0.25, linewidth=0.5)
    rel.set_axisbelow(True)
    strip_spines(rel)

    highlight = dict(FIGURE_CONFIGS)
    ys = list(range(len(configs)))[::-1]  # first config at the top
    for y, (name, row) in zip(ys, configs.items()):
        color = highlight.get(name, MUTED)
        disc.scatter([row["concordance"]], [y], s=70, color=color, zorder=3)
        disc.annotate(
            f"{row['concordance']:.3f}  p={row['concordance_p']:.4f}",
            xy=(0.70, y),  # one label column, clear of every dot and the 0.5 line
            va="center",
            fontsize=8.5,
            color=INK if row["significant"] else INK2,
            fontweight="bold" if row["significant"] else "normal",
        )
    disc.axvline(0.5, color=MUTED, linewidth=0.9, zorder=1)
    disc.set_yticks(ys)
    disc.set_yticklabels([LABELS[name] for name in configs], fontsize=9)
    disc.set_xlim(0.35, 0.85)
    disc.set_xlabel("Concordance with recall@5 (0.5 = chance)")
    significant = sum(1 for row in configs.values() if row["significant"])
    disc.set_title(
        f"Discrimination: {significant} of {len(configs)} labels beat chance",
        color=INK,
        fontsize=11,
        loc="left",
    )
    disc.grid(axis="x", alpha=0.25, linewidth=0.5)
    disc.set_axisbelow(True)
    strip_spines(disc)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
