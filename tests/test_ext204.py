"""Tests for the extended 204-query benchmark.

Same contract as ``test_analysis.py``: the loading rules are unit-tested, and
the numbers quoted in the README's "Does the reranker penalty replicate out of
domain?" section are pinned to the committed per-query records.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rageval.ext204 import ALL, EXT204_COMPARISONS, HELD_OUT, build_ext204, held_out_ids
from rageval.loading import EXT204_CONFIG_ORDER, METRICS, load_runs

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "runs"
EXT204_RUNS_DIR = ROOT / "data" / "runs-ext204"


@pytest.fixture(scope="module")
def runs_120():
    return load_runs(RUNS_DIR)


@pytest.fixture(scope="module")
def ext_runs():
    return load_runs(EXT204_RUNS_DIR, EXT204_CONFIG_ORDER)


@pytest.fixture(scope="module")
def ext204(runs_120, ext_runs):
    return build_ext204(EXT204_RUNS_DIR, runs_120)


class TestLoading:
    def test_all_five_configs_load(self, ext_runs) -> None:
        assert set(ext_runs) == set(EXT204_CONFIG_ORDER)

    def test_every_run_has_204_aligned_queries(self, ext_runs) -> None:
        ids = None
        for run in ext_runs.values():
            assert len(run.per_query) == 204
            if ids is None:
                ids = run.query_ids
            assert run.query_ids == ids

    def test_metric_values_are_in_unit_range(self, ext_runs) -> None:
        for run in ext_runs.values():
            for metric in METRICS:
                assert all(0.0 <= v <= 1.0 for v in run.metric(metric))

    def test_extended_set_contains_the_original_120(self, ext_runs, runs_120) -> None:
        original = set(runs_120["hybrid_bm25"].query_ids)
        assert original <= set(ext_runs["hybrid_bm25"].query_ids)

    def test_held_out_is_the_84_new_queries(self, ext_runs, runs_120) -> None:
        held = held_out_ids(ext_runs, runs_120)
        assert len(held) == 84
        assert not set(held) & set(runs_120["hybrid_bm25"].query_ids)


class TestSubset:
    def test_subset_keeps_only_the_requested_queries(self, ext_runs) -> None:
        run = ext_runs["hybrid_bm25"]
        wanted = run.query_ids[:10]
        subset = run.subset(wanted)
        assert subset.query_ids == wanted
        assert subset.name == run.name

    def test_subset_does_not_mutate_the_original(self, ext_runs) -> None:
        run = ext_runs["hybrid_bm25"]
        before = len(run.per_query)
        run.subset(run.query_ids[:5])
        assert len(run.per_query) == before

    def test_subset_rejects_unknown_query_ids(self, ext_runs) -> None:
        with pytest.raises(ValueError, match="not present in this run"):
            ext_runs["hybrid_bm25"].subset(["NOPE-001"])


def _ndcg(ext204: dict, baseline: str, variant: str, query_set: str) -> dict:
    return next(
        row
        for row in ext204["paired_comparisons"]
        if row["baseline"] == baseline
        and row["variant"] == variant
        and row["metric"] == "ndcg_at_5"
        and row["query_set"] == query_set
    )


class TestPublishedFindings:
    """The README's replication claims, asserted against the committed data."""

    def test_reranker_penalty_replicates_on_all_204(self, ext204) -> None:
        assert _ndcg(ext204, "hybrid_bm25", "hybrid_bm25_rerank", ALL)["delta"] == pytest.approx(
            -0.0904, abs=5e-4
        )
        assert _ndcg(ext204, "hybrid_bm25", "hybrid_bm25_rerank", ALL)["significant"]
        assert _ndcg(ext204, "dense_minilm", "dense_minilm_rerank", ALL)["delta"] == pytest.approx(
            -0.1419, abs=5e-4
        )
        assert _ndcg(ext204, "dense_minilm", "dense_minilm_rerank", ALL)["significant"]

    def test_reranker_penalty_replicates_on_the_84_held_out(self, ext204) -> None:
        hybrid = _ndcg(ext204, "hybrid_bm25", "hybrid_bm25_rerank", HELD_OUT)
        dense = _ndcg(ext204, "dense_minilm", "dense_minilm_rerank", HELD_OUT)
        assert hybrid["delta"] == pytest.approx(-0.0951, abs=5e-4)
        assert hybrid["significant"]
        assert dense["delta"] == pytest.approx(-0.1119, abs=5e-4)
        assert dense["significant"]
        assert hybrid["n_queries"] == 84

    def test_every_extended_ndcg_reranker_delta_is_negative(self, ext204) -> None:
        """The direction the figure's title asserts."""
        for query_set in (ALL, HELD_OUT):
            for baseline, variant in EXT204_COMPARISONS[1:]:
                assert _ndcg(ext204, baseline, variant, query_set)["delta"] < 0

    def test_hybrid_mrr_gain_does_not_replicate(self, ext204) -> None:
        """The honest one: on the wider set hybrid fusion's MRR win over the
        lexical baseline disappears, and on the held-out queries it reverses."""
        rows = {
            row["query_set"]: row
            for row in ext204["paired_comparisons"]
            if row["baseline"] == "baseline_lexical"
            and row["variant"] == "hybrid_bm25"
            and row["metric"] == "mrr"
        }
        assert rows[ALL]["delta"] < 0
        assert not rows[ALL]["significant"]
        assert rows[HELD_OUT]["delta"] < 0
        assert rows[HELD_OUT]["significant"]

    def test_reranker_splits_the_held_out_queries_evenly(self, ext204) -> None:
        """It does not hurt *more* queries out of domain -- it hurts them
        harder. Both halves are 42, yet the mean delta is negative."""
        polarity = ext204["reranker_polarity"][HELD_OUT]
        assert polarity["hurt"] == 42
        assert polarity["helped"] == 42
        assert polarity["unchanged"] == 0

    def test_query_set_sizes(self, ext204) -> None:
        assert ext204["query_sets"] == {ALL: 204, HELD_OUT: 84}
