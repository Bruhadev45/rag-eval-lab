"""Held-out retrieval ablation: predeclared variants, scored on the test split.

``python -m rageval.legal_demo.heldout`` re-runs the dev tuning of
:mod:`~rageval.legal_demo.live`, refuses to go on unless it still selects the
frozen configuration :data:`FROZEN_SHA`, then scores every variant in
:data:`VARIANTS` on dev and on test and writes ``results/legal-demo/live/``.
Nothing is chosen from test numbers. Only ``bm25``'s depth and context were
selected on dev; the other variants and their settings (the hybrid weight, the
reranker, the child split, the edges) were declared before test was first run,
not tuned. The command is deterministic, and each rerun (as the tests do)
re-scores test with the same result; the outputs list what changed between runs.

All variants keep ``context_k = 2`` passages from at most ``depth = 3``
candidates, the frozen dev selection, so they differ only in how they rank:

- ``bm25`` (the frozen dev selection) and ``tfidf``: one lexical retriever.
- ``hybrid``: weighted RRF of both through ragfuse, bm25 weight 0.5.
- ``rerank``: the hybrid candidates reordered by ragfuse's ``rerank`` with a
  *lexical* score, the share of distinct query words a passage contains. No
  cross-encoder or other model was run.
- ``parent``: BM25 over child passages (each line and sentence of a section;
  a heading's number such as ``2.`` splits off as its own child), each section
  scored by its best child.
- ``graph``: BM25 seeds, each followed by the sections its explicit reference
  edges name (``data/legal-demo/references.json``, a byte-identical copy of
  minirag's file), one hop, inside the eligible pool only.

Every kept id is checked again against the case's eligible pool and
:func:`~rageval.legal_demo.model.ineligibility`. There is no learned-dense row,
because no embedding model was run.

Uncertainty is paired by **case**, since a case's paraphrases are not
independent: per-case means, an exact two-sided sign-flip permutation test and
a seeded case bootstrap 95% interval (``rageval.stats``) for each variant minus
``bm25``. Headline scores are means over queries, so they differ from the
case-level mean differences. Test has only seven cases with evidence: an
interval of ``[0, 0]`` means no case differed, not that the difference is known.
A missed query is labelled ``at-ceiling`` when its recall equals the most that
``context_k`` passages can reach, and ``ranking`` otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Final, TypeAlias

from ..stats import paired_comparison
from .live import (
    METRIC_DECIMALS,
    OUT_DIR,
    LiveRunError,
    RunConfig,
    hybrid_trace,
    provenance,
    retrieve,
    score_record,
    tune,
)
from .model import CATEGORIES, DATA_DIR, Case, Dataset, DatasetError, ineligibility
from .report import ROOT
from .retrievers import bm25, eligible_passages, tokens
from .validate import load_dataset

try:
    from ragfuse import rerank
except ImportError as exc:  # the owner's ragfuse, never the PyPI package of that name
    raise ImportError(
        "rageval.legal_demo.heldout needs the owner's ragfuse: pip install -e '.[live]'"
    ) from exc

FROZEN_SHA: Final = "sha256:a38309eefe854f5046713659d44ea5b6d961e9dfb44e605dacae8ee53cc59ae1"
FROZEN: Final = RunConfig("bm25", 3, 2)
HYBRID: Final = RunConfig("hybrid", 3, 2, 0.5)
HYBRID_CANDIDATES: Final = RunConfig("hybrid", 3, 3, 0.5)  # rerank reorders all depth-3 hits
METRICS: Final = ("recall", "rr", "ndcg")
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 70
MAX_EXACT_CASES: Final = 16  # 2**16 sign patterns
REFERENCES: Final = DATA_DIR / "references.json"
REFERENCES_UPSTREAM: Final = {
    "repository": "https://github.com/Bruhadev45/minirag",
    "path": "examples/legal_rag/fixtures/references.json",
    "commit": "9c9d5d8c07d3d9f2ef11049d904cd83283474690",
    "file_sha256": "18348fc818da971d45edc8a428fca6427ce7ee5423eba60f6ce9cc8d0193d1fd",
}
# What changed between test runs. Kept by hand: the code cannot see its own history.
TEST_RUN_LOG: Final = (
    "First test run: the six variants as declared, before any test number was seen.",
    (
        "Later runs, all after review: diagnostics (graph reach, miss kinds), the shared"
        " rageval.stats bootstrap, refactors and wording. Every variant keeps the same"
        " passages for every query as in the first run."
    ),
)
MAX_REFERENCE_BYTES: Final = 65_536
_CHILD_SPLIT: Final = re.compile(r"(?<=[.;:])\s+|\n+")

Edge: TypeAlias = "tuple[str, str, str | None]"  # citing label, family, section or None
Retrieve: TypeAlias = Callable[[Dataset, Case, str, tuple[Edge, ...]], tuple[str, ...]]


def load_references(root: Path, dataset: Dataset) -> tuple[Edge, ...]:
    """The explicit reference edges, checked against the frozen fixture.

    Refuses a file that is not byte-identical to :data:`REFERENCES_UPSTREAM`, a
    citing section that is missing or does not contain its quoted words, and a
    target family or section the fixture does not have. The offsets in the file
    index minirag's whole record text, so here only the words are checked.
    """
    path = root / REFERENCES
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_REFERENCE_BYTES + 1)
    except OSError as exc:
        raise DatasetError(f"cannot read {REFERENCES} ({type(exc).__name__})") from exc
    if len(raw) > MAX_REFERENCE_BYTES:
        raise DatasetError(f"{REFERENCES} is larger than {MAX_REFERENCE_BYTES} bytes")
    if hashlib.sha256(raw).hexdigest() != REFERENCES_UPSTREAM["file_sha256"]:
        raise DatasetError(f"{REFERENCES} is not the frozen copy of minirag's file")
    try:
        return _parse_references(json.loads(raw)["references"], dataset)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"{REFERENCES} is malformed ({type(exc).__name__})") from None


def _parse_references(entries: list[dict[str, str]], dataset: Dataset) -> tuple[Edge, ...]:
    sections = {
        (source.family_id, sid) for source in dataset.sources.values() for sid in source.sections
    }
    edges = []
    for entry in entries:
        source_id, _, section_id = entry["from"].rpartition("#")
        source = dataset.sources.get(source_id)
        family, _, target = entry["to"].partition("#")
        if source is None or entry["quote"] not in source.sections.get(section_id, ""):
            raise DatasetError(f"reference from {entry['from']} does not match the fixture")
        if not any(f == family and (not target or s == target) for f, s in sections):
            raise DatasetError(f"reference to {entry['to']} names nothing in the fixture")
        edges.append((entry["from"], family, target or None))
    return tuple(edges)


def _checked(dataset: Dataset, case: Case, kept: Sequence[str]) -> tuple[str, ...]:
    pool = eligible_passages(dataset, case)
    for pid in kept:
        source = dataset.sources.get(pid.rpartition("#")[0])
        if pid not in pool or source is None or ineligibility(source, case):
            raise LiveRunError("a variant returned a passage outside the eligible pool")
    if len(set(kept)) != len(kept):
        raise LiveRunError("a variant returned the same passage twice")
    return tuple(kept)


def _config(config: RunConfig) -> Retrieve:
    return lambda dataset, case, query, _edges: retrieve(dataset, case, query, config)


def _rerank(dataset: Dataset, case: Case, query: str, _edges: tuple[Edge, ...]) -> tuple[str, ...]:
    pool = eligible_passages(dataset, case)
    trace = hybrid_trace(dataset, case, query, HYBRID_CANDIDATES)
    asked = set(tokens(query))
    coverage = {
        key: len(asked & set(tokens(pool[key]))) / len(asked) if asked else 0.0
        for key in trace.keys
    }
    kept = rerank(trace, coverage, eligible=set(pool), top_k=HYBRID.context_k).keys
    return _checked(dataset, case, kept)


def _parent(dataset: Dataset, case: Case, query: str, _edges: tuple[Edge, ...]) -> tuple[str, ...]:
    pool = eligible_passages(dataset, case)
    children = {
        f"{pid}/{n}": child
        for pid, text in pool.items()
        for n, child in enumerate(c for c in _CHILD_SPLIT.split(text) if c.strip())
    }
    parents: list[str] = []
    for child_id, _ in bm25(children, query, len(children)):
        parent = child_id.rpartition("/")[0]
        if parent not in parents:
            parents.append(parent)
    return _checked(dataset, case, parents[: FROZEN.context_k])


def graph_walk(
    dataset: Dataset, case: Case, query: str, edges: tuple[Edge, ...]
) -> tuple[list[str], list[str]]:
    """BM25 seeds, each followed by its eligible reference targets; and the targets reached.

    One hop, in seed order, then edge order, then pool order. A target outside
    the case's eligible pool is never reached, so it is neither kept nor counted.
    """
    pool = eligible_passages(dataset, case)
    walk: list[str] = []
    reached: list[str] = []
    for seed in retrieve(dataset, case, query, RunConfig("bm25", FROZEN.depth, FROZEN.depth)):
        targets = [
            pid
            for citing, family, section in edges
            if citing == seed
            for pid in pool
            if dataset.sources[pid.rpartition("#")[0]].family_id == family
            and (section is None or pid.rpartition("#")[2] == section)
        ]
        for pid in targets:
            if pid not in reached:
                reached.append(pid)
        for pid in (seed, *targets):
            if pid not in walk:
                walk.append(pid)
    return walk, reached


def _graph(dataset: Dataset, case: Case, query: str, edges: tuple[Edge, ...]) -> tuple[str, ...]:
    return _checked(dataset, case, graph_walk(dataset, case, query, edges)[0][: FROZEN.context_k])


def graph_reach(dataset: Dataset, split: str, edges: tuple[Edge, ...]) -> dict[str, int]:
    """How often an edge fired on ``split``.

    Counts queries whose seeds reach a target, and queries whose kept context
    differs from plain BM25's because of one. A target that is also a seed adds
    nothing, so it does not count as added.
    """
    reached = added = 0
    for case in (c for c in dataset.cases if c.split == split):
        for query in case.queries:
            walk, targets = graph_walk(dataset, case, query, edges)
            reached += bool(targets)
            plain = retrieve(dataset, case, query, FROZEN)
            added += tuple(walk[: FROZEN.context_k]) != plain
    return {"queries_reaching_a_target": reached, "queries_where_a_target_changed_context": added}


VARIANTS: Final[dict[str, tuple[str, Retrieve]]] = {
    "bm25": ("BM25, the frozen dev selection", _config(FROZEN)),
    "tfidf": ("TF-IDF cosine", _config(RunConfig("tfidf", 3, 2))),
    "hybrid": ("weighted RRF via ragfuse, bm25 weight 0.5", _config(HYBRID)),
    "rerank": ("hybrid, reranked by query-word coverage (lexical)", _rerank),
    "parent": ("BM25 over lines and sentences, sections by best child", _parent),
    "graph": ("BM25 seeds plus their explicit reference targets, one hop", _graph),
}


def score(
    dataset: Dataset, split: str, name: str, edges: tuple[Edge, ...], *, held_out: bool = False
) -> dict[str, Any]:
    """Every query of ``split`` through variant ``name``; test needs ``held_out``."""
    if split == "test" and not held_out:
        raise LiveRunError("test is held out: score it only through run_ablation")
    if split not in ("dev", "test"):
        raise LiveRunError("variants are scored on dev and test only")
    _, fetch = VARIANTS[name]
    records = []
    for case in (c for c in dataset.cases if c.split == split):
        for qid, query in zip(case.query_ids, case.queries, strict=True):
            kept = fetch(dataset, case, query, edges)
            record = score_record(dataset, case, qid, kept, FROZEN.context_k)
            records.append({"case": case.id, **record})
    return _summary(records)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if "recall" in r]
    unscored = [r for r in records if "recall" not in r]
    means = {m: _mean([r[m] for r in scored]) for m in METRICS}
    by_category = {
        cat: _mean([r["recall"] for r in scored if r["category"] == cat])
        for cat in CATEGORIES
        if any(r["category"] == cat for r in scored)
    }
    misses = [
        {
            "query_id": r["query_id"],
            "recall": r["recall"],
            "ceiling": ceiling(r),
            "kind": "at-ceiling" if r["recall"] >= ceiling(r) else "ranking",
            "missing": sorted(set(r["relevant"]) - set(r["retrieved"])),
        }
        for r in scored
        if set(r["relevant"]) - set(r["retrieved"])
    ]
    return {
        "queries": len(records),
        "scored": len(scored),
        **means,
        "recall_by_category": by_category,
        "misses": misses,
        "no_evidence": len(unscored),
        "no_evidence_retrieved_something": sum(1 for r in unscored if r["retrieved"]),
        "per_query": records,
    }


def ceiling(record: dict[str, Any]) -> float:
    """The most recall ``context_k`` passages can reach for this query, as rounded for output."""
    return round(min(1.0, FROZEN.context_k / len(record["relevant"])), METRIC_DECIMALS)


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), METRIC_DECIMALS) if values else None


def case_means(run: dict[str, Any], metric: str) -> dict[str, float]:
    """``{case id: mean of metric over its scored queries}``."""
    by_case: dict[str, list[float]] = {}
    for record in run["per_query"]:
        if metric in record:
            by_case.setdefault(record["case"], []).append(record[metric])
    return {case: sum(v) / len(v) for case, v in sorted(by_case.items())}


def paired(base: dict[str, float], other: dict[str, float]) -> dict[str, Any]:
    """Case-paired difference ``other - base``: mean, exact sign-flip p, bootstrap 95% CI.

    The interval is :func:`rageval.stats.paired_comparison`'s paired bootstrap,
    run over cases. Its p-value is Monte Carlo; with this few cases every sign
    pattern can be enumerated, so the exact two-sided p is reported instead.
    """
    if base.keys() != other.keys() or not base:
        raise LiveRunError("paired statistics need the same, non-empty set of cases")
    if len(base) > MAX_EXACT_CASES:
        raise LiveRunError(f"exact sign-flip enumeration is capped at {MAX_EXACT_CASES} cases")
    diffs = [other[c] - base[c] for c in base]
    observed = abs(sum(diffs))
    patterns = 2 ** len(diffs)
    extreme = sum(
        1
        for signs in product((1, -1), repeat=len(diffs))
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
    )
    boot = paired_comparison(
        list(base.values()),
        list(other.values()),
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r = METRIC_DECIMALS
    return {
        "cases": len(diffs),
        "mean_diff": round(boot.delta, r),
        "helped": sum(1 for d in diffs if d > 1e-12),
        "hurt": sum(1 for d in diffs if d < -1e-12),
        "sign_flip_p": round(extreme / patterns, r),
        "bootstrap_ci95": [round(boot.ci_low, r), round(boot.ci_high, r)],
    }


def run_ablation(root: Path) -> dict[str, Any]:
    """Re-check the frozen dev selection, then score every variant on dev and test."""
    dataset = load_dataset(root)
    selected = tune(dataset, "dev")["selected"]
    if selected["config_sha256"] != FROZEN_SHA or FROZEN.sha256 != FROZEN_SHA:
        raise LiveRunError("dev tuning no longer selects the frozen configuration; re-freeze it")
    edges = load_references(root, dataset)
    runs = {
        split: {name: score(dataset, split, name, edges, held_out=True) for name in VARIANTS}
        for split in ("dev", "test")
    }
    reach = {split: graph_reach(dataset, split, edges) for split in ("dev", "test")}
    comparisons = {
        name: {
            metric: paired(
                case_means(runs["test"]["bm25"], metric), case_means(runs["test"][name], metric)
            )
            for metric in METRICS
        }
        for name in VARIANTS
        if name != "bm25"
    }
    references_sha = "sha256:" + REFERENCES_UPSTREAM["file_sha256"]
    return {
        "provenance": {
            **provenance(dataset),
            "references": {"sha256": references_sha, "upstream": REFERENCES_UPSTREAM},
        },
        "frozen": {"config": asdict(FROZEN), "config_sha256": FROZEN_SHA},
        "variants": {name: description for name, (description, _) in VARIANTS.items()},
        "test_runs": TEST_RUN_LOG,
        "uncertainty": {
            "unit": "case (paraphrases of one case are averaged first)",
            "sign_flip": "exact, two-sided, over every sign pattern of the case differences",
            "bootstrap": f"rageval.stats paired bootstrap, {BOOTSTRAP_RESAMPLES} case resamples,"
            + f" seed {BOOTSTRAP_SEED}",
        },
        "runs": runs,
        "graph_reach": reach,
        "test_vs_bm25": comparisons,
    }


def render_markdown(result: dict[str, Any]) -> str:
    """A short, human-readable view of the held-out ablation."""
    test, dev = result["runs"]["test"], result["runs"]["dev"]
    first = test["bm25"]
    lines = [
        "# Legal demo: held-out retrieval ablation",
        "",
        "Fictional data, not law and not legal advice. Generated by",
        "`python -m rageval.legal_demo.heldout`; edits here are overwritten.",
        f"Each variant ran on the {first['queries']} test queries ({first['scored']} with evidence,",
        f"from {len(case_means(first, 'recall'))} cases), at depth {FROZEN.depth} and context"
        + f" {FROZEN.context_k}.",
        f"Only `bm25`'s depth and context were selected on dev (`{FROZEN_SHA}`);",
        "the other variants' settings were declared before test was first run, not tuned.",
        "The command is deterministic: every rerun re-scores test with the same result.",
        "",
        "Lexical only: no embedding, cross-encoder or language model was run, so there is",
        "no learned-dense row. Seven test cases cannot separate small differences; these",
        "numbers check the variants' code paths on fictional data, not retrieval quality.",
        "",
        f"- References: `{result['provenance']['references']['sha256']}` (minirag "
        + f"`{REFERENCES_UPSTREAM['commit'][:7]}`)",
        f"- Uncertainty: by {result['uncertainty']['unit']}; {result['uncertainty']['bootstrap']}",
        "",
        "## Test scores (means over queries)",
        "",
        "| Variant | Recall | MRR | nDCG | Dev recall | Dev MRR | No-evidence queries with hits |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, description in result["variants"].items():
        t, d = test[name], dev[name]
        lines.append(
            f"| `{name}`: {description} | {t['recall']} | {t['rr']} | {t['ndcg']} | "
            f"{d['recall']} | {d['rr']} | {t['no_evidence_retrieved_something']} of "
            f"{t['no_evidence']} |"
        )
    lines += [
        "",
        "A no-evidence query with hits only shows that a lexical retriever found shared",
        "words; abstention is not scored here (Day 73).",
        "",
        "## Paired against `bm25` on test, by case",
        "",
        "Means over cases, so they differ from the query means above. `[0.0, 0.0]` means",
        "no case differed, not that the difference is known to be zero.",
        "",
        "| Variant | Metric | Mean diff | Helped / hurt cases | Sign-flip p | Bootstrap 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for name, metrics in result["test_vs_bm25"].items():
        for metric, stats in metrics.items():
            low, high = stats["bootstrap_ci95"]
            lines.append(
                f"| `{name}` | {metric} | {stats['mean_diff']} | {stats['helped']} / "
                f"{stats['hurt']} | {stats['sign_flip_p']} | [{low}, {high}] |"
            )
    categories = sorted({c for run in test.values() for c in run["recall_by_category"]})
    lines += [
        "",
        "## Test recall by category, and misses",
        "",
        "A miss is `at-ceiling` when its recall is the most two passages can reach (a",
        "conflicting-source case cites three sections), and `ranking` otherwise.",
        "",
        "| Variant | " + " | ".join(categories) + " | Missed queries (kind) |",
        "|---|" + "---|" * len(categories) + "---|",
    ]
    for name, run in test.items():
        cells = [str(run["recall_by_category"].get(c, "-")) for c in categories]
        missed = ", ".join(f"{m['query_id']} ({m['kind']})" for m in run["misses"]) or "none"
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {missed} |")
    reach = result["graph_reach"]["test"]
    lines += [
        "",
        f"Graph edges reached a target on {reach['queries_reaching_a_target']} of "
        + f"{first['queries']} test queries and changed the kept context on "
        + f"{reach['queries_where_a_target_changed_context']}. A family-wide edge reaches every",
        "eligible section of that family, kept in pool order, not by relevance.",
        "",
        "## Changes between test runs",
        "",
    ]
    lines += [f"- {entry}" for entry in result["test_runs"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rageval.legal_demo.heldout",
        description="Score the frozen legal-demo retrieval variants once on the test split.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    root = parser.parse_args(argv).root
    try:
        result = run_ablation(root)
    except (DatasetError, LiveRunError) as exc:
        raise SystemExit(f"rageval.legal_demo.heldout: error: {exc}") from exc
    outputs = {
        "test-ablation.json": json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        "TEST_ABLATION.md": render_markdown(result),
    }
    out_dir = root / OUT_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in outputs.items():
            (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise SystemExit(
            f"rageval.legal_demo.heldout: error: cannot write {OUT_DIR} ({type(exc).__name__})"
        ) from exc
    print(f"scored {len(VARIANTS)} frozen variants on dev and test")
    for name in outputs:
        print(f"wrote {OUT_DIR}/{name}")


if __name__ == "__main__":
    main()
