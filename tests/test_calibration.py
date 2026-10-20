"""Tests for the confidence-label calibration analysis.

Same contract as ``test_analysis.py``: the arithmetic is unit-tested on small
hand-checkable inputs, and the numbers quoted in the README's "Is the
confidence label worth anything?" section are pinned to the committed data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rageval.calibration import (
    LEVELS,
    MIN_CELL,
    NOMINAL,
    LevelStats,
    build_calibration,
    concordance,
    concordance_test,
    expected_calibration_error,
    non_decreasing,
    overconfidence,
    reliability,
)
from rageval.loading import CONFIG_ORDER, Run, load_runs

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"


def _run(rows: list[tuple[str, float]]) -> Run:
    return Run(
        name="toy",
        description="",
        timestamp="",
        per_query=tuple(
            {"query_id": f"Q{i:03d}", "confidence": label, "recall_at_5": recall}
            for i, (label, recall) in enumerate(rows)
        ),
    )


@pytest.fixture(scope="module")
def calibration():
    return build_calibration(load_runs(RUNS_DIR))


class TestReliability:
    def test_groups_by_label_in_ascending_order(self) -> None:
        run = _run([("high", 1.0), ("low", 0.0), ("high", 0.5), ("low", 1.0)])
        assert reliability(run) == (
            LevelStats(level="low", n=2, nominal=NOMINAL["low"], mean_recall=0.5),
            LevelStats(level="high", n=2, nominal=NOMINAL["high"], mean_recall=0.75),
        )

    def test_unused_levels_are_omitted(self) -> None:
        levels = reliability(_run([("medium", 1.0)] * 3))
        assert [s.level for s in levels] == ["medium"]

    def test_unknown_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown confidence label 'certain'"):
            reliability(_run([("certain", 1.0)]))

    def test_missing_label_is_rejected(self) -> None:
        run = Run(name="toy", description="", timestamp="", per_query=({"recall_at_5": 1.0},))
        with pytest.raises(ValueError, match="no confidence label"):
            reliability(run)

    def test_nominal_confidence_rises_with_the_label(self) -> None:
        values = [NOMINAL[level] for level in LEVELS]
        assert values == sorted(values)
        assert all(0.0 < v < 1.0 for v in values)


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_labels_score_zero(self) -> None:
        levels = (
            LevelStats("low", 4, 0.375, 0.375),
            LevelStats("high", 4, 0.875, 0.875),
        )
        assert expected_calibration_error(levels) == pytest.approx(0.0)

    def test_gaps_are_weighted_by_level_size(self) -> None:
        levels = (
            LevelStats("low", 3, 0.375, 0.875),  # gap 0.5
            LevelStats("high", 1, 0.875, 0.875),  # gap 0
        )
        assert expected_calibration_error(levels) == pytest.approx(0.375)

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no queries"):
            expected_calibration_error(())


class TestOverconfidence:
    def test_sign_says_which_way_the_label_errs(self) -> None:
        assert overconfidence((LevelStats("high", 2, 0.875, 0.5),)) > 0
        assert overconfidence((LevelStats("low", 2, 0.375, 1.0),)) < 0

    def test_equals_minus_ece_when_every_level_is_underconfident(self) -> None:
        levels = (
            LevelStats("low", 3, 0.375, 0.875),
            LevelStats("high", 5, 0.875, 0.9),
        )
        assert overconfidence(levels) == -expected_calibration_error(levels)

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no queries"):
            overconfidence(())


class TestNonDecreasing:
    def test_rising_recall_is_non_decreasing(self) -> None:
        levels = (LevelStats("low", 10, 0.375, 0.5), LevelStats("high", 10, 0.875, 0.5))
        assert non_decreasing(levels) is True

    def test_a_drop_between_full_levels_is_not(self) -> None:
        levels = (LevelStats("low", 10, 0.375, 0.9), LevelStats("high", 10, 0.875, 0.5))
        assert non_decreasing(levels) is False

    def test_sparse_levels_cannot_decide_the_verdict(self) -> None:
        levels = (
            LevelStats("low", MIN_CELL - 1, 0.375, 1.0),  # would break the order
            LevelStats("medium", MIN_CELL, 0.625, 0.5),
            LevelStats("high", MIN_CELL, 0.875, 0.6),
        )
        assert non_decreasing(levels) is True

    def test_fewer_than_two_full_levels_gives_no_verdict(self) -> None:
        levels = (LevelStats("low", 1, 0.375, 0.0), LevelStats("high", 50, 0.875, 0.8))
        assert non_decreasing(levels) is None


class TestConcordance:
    def test_perfect_ranking_scores_one(self) -> None:
        assert concordance([0, 1, 2, 3], [0.0, 0.25, 0.5, 1.0]) == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self) -> None:
        assert concordance([3, 2, 1, 0], [0.0, 0.25, 0.5, 1.0]) == pytest.approx(0.0)

    def test_a_constant_label_scores_one_half(self) -> None:
        assert concordance([2, 2, 2, 2], [0.0, 0.5, 1.0, 1.0]) == pytest.approx(0.5)

    def test_ties_in_the_outcome_are_not_pairs(self) -> None:
        # Only the four (0.0, 1.0) pairs count: 3 concordant + 1 tied.
        assert concordance([0, 1, 1, 2], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.875)

    def test_constant_outcome_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same outcome"):
            concordance([0, 1, 2], [1.0, 1.0, 1.0])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            concordance([0, 1], [1.0])


class TestConcordanceTest:
    def test_a_strong_ranking_is_significant(self) -> None:
        labels = [0] * 20 + [3] * 20
        outcomes = [0.0] * 20 + [1.0] * 20
        observed, p = concordance_test(labels, outcomes, n_resamples=2000)
        assert observed == pytest.approx(1.0)
        assert p < 0.01

    def test_a_constant_label_has_p_one(self) -> None:
        _, p = concordance_test([1] * 10, [0.0, 1.0] * 5, n_resamples=500)
        assert p == pytest.approx(1.0)

    def test_same_seed_reproduces_exactly(self) -> None:
        labels = [0, 1, 2, 3, 1, 2] * 5
        outcomes = [0.0, 0.5, 1.0, 0.5, 1.0, 0.0] * 5
        assert concordance_test(labels, outcomes) == concordance_test(labels, outcomes)

    def test_does_not_mutate_its_input(self) -> None:
        labels = [0, 1, 2, 3] * 5
        before = list(labels)
        concordance_test(labels, [0.0, 0.5, 1.0, 1.0] * 5, n_resamples=200)
        assert labels == before


def _levels(calibration: dict, name: str) -> dict:
    return {row["level"]: row for row in calibration["configs"][name]["levels"]}


class TestPublishedFindings:
    """The README's calibration claims, asserted against the committed data."""

    def test_every_config_is_analysed(self, calibration) -> None:
        assert tuple(calibration["configs"]) == CONFIG_ORDER

    def test_hybrid_low_label_still_retrieves_81_percent(self, calibration) -> None:
        levels = _levels(calibration, "hybrid_bm25")
        assert levels["low"]["n"] == 42
        assert levels["low"]["mean_recall_at_5"] == pytest.approx(0.8095, abs=5e-4)
        assert levels["medium"]["mean_recall_at_5"] == pytest.approx(0.8333, abs=5e-4)
        assert levels["high"]["n"] == 48
        assert levels["high"]["mean_recall_at_5"] == pytest.approx(0.875, abs=5e-4)

    def test_hybrid_label_is_ordered_but_not_significant(self, calibration) -> None:
        hybrid = calibration["configs"]["hybrid_bm25"]
        assert hybrid["non_decreasing"] is True
        assert hybrid["concordance"] == pytest.approx(0.5604, abs=5e-4)
        assert hybrid["concordance_p"] == pytest.approx(0.2568, abs=5e-4)
        assert not hybrid["significant"]

    def test_hybrid_label_is_underconfident_by_20_points(self, calibration) -> None:
        hybrid = calibration["configs"]["hybrid_bm25"]
        assert hybrid["ece"] == pytest.approx(0.2042, abs=5e-4)
        assert hybrid["overconfidence"] == -hybrid["ece"]

    def test_ece_spans_lexical_to_inlegalbert(self, calibration) -> None:
        eces = {name: row["ece"] for name, row in calibration["configs"].items()}
        assert min(eces, key=eces.get) == "baseline_lexical"
        assert eces["baseline_lexical"] == pytest.approx(0.0875, abs=5e-4)
        assert max(eces, key=eces.get) == "dense_inlegalbert"
        assert eces["dense_inlegalbert"] == pytest.approx(0.3542, abs=5e-4)

    def test_only_the_full_corpus_dense_label_ranks_queries(self, calibration) -> None:
        significant = [name for name, row in calibration["configs"].items() if row["significant"]]
        assert significant == ["dense_minilm_full"]
        full = calibration["configs"]["dense_minilm_full"]
        assert full["concordance"] == pytest.approx(0.6548, abs=5e-4)
        # At the 10,000-resample floor, and inside a Bonferroni bound for 7 tests.
        assert full["concordance_p"] == pytest.approx(1 / 10_001, abs=1e-6)
        assert full["concordance_p"] < 0.05 / len(CONFIG_ORDER)

    def test_full_corpus_insufficient_label_still_retrieves_56_percent(self, calibration) -> None:
        insufficient = _levels(calibration, "dense_minilm_full")["insufficient"]
        assert insufficient["n"] == 64
        assert insufficient["mean_recall_at_5"] == pytest.approx(0.5625, abs=5e-4)

    def test_lexical_and_inlegalbert_labels_do_no_better_than_chance(self, calibration) -> None:
        """Below 0.5, but not significantly: no signal, not a reversed one."""
        for name, expected, p in (
            ("baseline_lexical", 0.4714, 0.3695),
            ("dense_inlegalbert", 0.4794, 0.6003),
        ):
            row = calibration["configs"][name]
            assert row["concordance"] == pytest.approx(expected, abs=5e-4)
            assert row["concordance_p"] == pytest.approx(p, abs=5e-4)
            assert not row["significant"]
            assert row["non_decreasing"] is False

    def test_lexical_says_high_on_103_queries(self, calibration) -> None:
        high = _levels(calibration, "baseline_lexical")["high"]
        assert high["n"] == 103
        assert high["mean_recall_at_5"] == pytest.approx(0.8204, abs=5e-4)

    def test_inlegalbert_insufficient_out_retrieves_its_high(self, calibration) -> None:
        levels = _levels(calibration, "dense_inlegalbert")
        assert levels["insufficient"]["mean_recall_at_5"] == pytest.approx(0.7143, abs=5e-4)
        assert levels["high"]["mean_recall_at_5"] == pytest.approx(0.4552, abs=5e-4)

    def test_subset_minilm_labels_every_query_medium(self, calibration) -> None:
        minilm = calibration["configs"]["dense_minilm"]
        assert [(row["level"], row["n"]) for row in minilm["levels"]] == [("medium", 120)]
        assert minilm["non_decreasing"] is None
        assert minilm["concordance"] == pytest.approx(0.5)
        assert minilm["concordance_p"] == pytest.approx(1.0)
