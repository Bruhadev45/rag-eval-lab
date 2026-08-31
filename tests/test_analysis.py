"""Tests that pin the repository's claims to its data.

These are not unit tests of arithmetic alone -- several assert the *published
findings* against the committed per-query records, so a data or code change
that moved a headline number would fail CI rather than silently rewriting
history.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rageval.loading import CONFIG_ORDER, METRICS, load_runs
from rageval.stats import paired_comparison

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"


@pytest.fixture(scope="module")
def runs():
    return load_runs(RUNS_DIR)


class TestLoading:
    def test_all_seven_configs_load(self, runs) -> None:
        assert set(runs) == set(CONFIG_ORDER)

    def test_every_run_has_120_aligned_queries(self, runs) -> None:
        ids = None
        for run in runs.values():
            assert len(run.per_query) == 120
            these = [row["query_id"] for row in run.per_query]
            if ids is None:
                ids = these
            assert these == ids

    def test_metric_values_are_in_unit_range(self, runs) -> None:
        for run in runs.values():
            for metric in METRICS:
                assert all(0.0 <= v <= 1.0 for v in run.metric(metric))

    def test_unknown_metric_is_rejected(self, runs) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            runs["hybrid_bm25"].metric("f1")


class TestPairedComparison:
    def test_identical_lists_give_zero_delta_and_high_p(self) -> None:
        values = [0.2, 0.5, 0.9, 0.4] * 10
        result = paired_comparison(values, list(values), n_resamples=2000)
        assert result.delta == pytest.approx(0.0)
        assert not result.significant

    def test_constant_shift_is_detected(self) -> None:
        base = [0.5] * 40
        better = [0.6] * 40
        result = paired_comparison(base, better, n_resamples=2000)
        assert result.delta == pytest.approx(0.1)
        assert result.significant
        assert result.ci_low > 0

    def test_direction_is_b_minus_a(self) -> None:
        result = paired_comparison([1.0] * 10, [0.0] * 10, n_resamples=500)
        assert result.delta == pytest.approx(-1.0)

    def test_same_seed_reproduces_exactly(self) -> None:
        a = [0.1, 0.4, 0.7, 0.2, 0.9] * 8
        b = [0.2, 0.3, 0.8, 0.4, 0.7] * 8
        first = paired_comparison(a, b)
        second = paired_comparison(a, b)
        assert first == second

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            paired_comparison([1.0], [1.0, 2.0])

    def test_empty_lists_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            paired_comparison([], [])


class TestPublishedFindings:
    """The claims in the README, asserted against the committed data."""

    def test_reranker_penalty_on_hybrid_is_8_7_points(self, runs) -> None:
        result = paired_comparison(
            runs["hybrid_bm25"].metric("ndcg_at_5"),
            runs["hybrid_bm25_rerank"].metric("ndcg_at_5"),
        )
        assert result.delta == pytest.approx(-0.0871, abs=5e-4)
        assert result.significant

    def test_reranker_penalty_on_dense_is_16_3_points(self, runs) -> None:
        result = paired_comparison(
            runs["dense_minilm"].metric("ndcg_at_5"),
            runs["dense_minilm_rerank"].metric("ndcg_at_5"),
        )
        assert result.delta == pytest.approx(-0.1629, abs=5e-4)
        assert result.significant

    def test_hybrid_mrr_gain_over_lexical_is_significant(self, runs) -> None:
        result = paired_comparison(
            runs["baseline_lexical"].metric("mrr"),
            runs["hybrid_bm25"].metric("mrr"),
        )
        assert result.delta > 0
        assert result.significant

    def test_hybrid_recall_gain_is_not_significant(self, runs) -> None:
        """The honest one: the recall improvement is within noise, and the
        repository must keep saying so."""
        result = paired_comparison(
            runs["baseline_lexical"].metric("recall_at_5"),
            runs["hybrid_bm25"].metric("recall_at_5"),
        )
        assert result.delta > 0
        assert not result.significant

    def test_reranker_hurts_a_majority_of_queries(self, runs) -> None:
        deltas = [
            b - a
            for a, b in zip(
                runs["hybrid_bm25"].metric("ndcg_at_5"),
                runs["hybrid_bm25_rerank"].metric("ndcg_at_5"),
            )
        ]
        hurt = sum(1 for d in deltas if d < 0)
        helped = sum(1 for d in deltas if d > 0)
        assert hurt == 75
        assert helped == 36
