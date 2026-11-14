"""Live retrieval runs on the frozen legal-demo set, and tuning on dev only.

``python -m rageval.legal_demo.live`` loads and re-checks the frozen set, runs
every configuration in :data:`GRID` on the dev split, picks one by a declared
rule and writes ``results/legal-demo/live/``. Nothing here reads a result
file: each run executes :mod:`~rageval.legal_demo.retrievers` on the queries.

- **Eligibility first.** A case's pool holds only eligible sections. Hybrid
  fusion goes through ragfuse's ``fuse(..., eligible=pool)``, which refuses
  any other id, and every returned id is checked again against the pool and
  :func:`~rageval.legal_demo.model.ineligibility` before it is scored.
- **Dev only.** :func:`tune` refuses the test split, and :func:`evaluate`
  refuses it unless ``held_out=True`` is passed for a frozen configuration.
  This module never passes it; held-out scoring is a separate step.
- **Provenance.** Each output records hashes of the harness code, the frozen
  data and every configuration, the ragfuse version and commit, and that no
  model was run. The outputs carry no timestamps or paths, so a rerun of the
  same code on the same data writes the same bytes.

Scores are mean recall, reciprocal rank and nDCG at ``context_k`` over queries
whose case cites evidence. Queries with no evidence (expected abstentions) are
counted, with how many still retrieved something; retrieval cannot score them.
Dev has 15 queries over at most a handful of eligible sections per case, so
these numbers exercise the code; they do not measure retrieval quality.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Final

from ragfuse import fuse, ndcg_at_k, recall_at_k, reciprocal_rank

from .model import SPLITS, Case, Dataset, DatasetError, canonical_sha256, ineligibility
from .report import ROOT
from .retrievers import RANKERS, RETRIEVERS, Ranking, eligible_passages
from .validate import load_dataset

OUT_DIR: Final = Path("results") / "legal-demo" / "live"
SYSTEMS: Final = (*RETRIEVERS, "hybrid")
RRF_K: Final = 60
MAX_DEPTH: Final = 100
TUNING_SPLITS: Final = ("train", "dev")
CODE_FILES: Final = ("live.py", "retrievers.py", "model.py")
METRIC_DECIMALS: Final = 4
MODEL: Final = {
    "kind": "none",
    "note": "BM25 and TF-IDF cosine are lexical; no embedding or language model was run.",
}


class LiveRunError(ValueError):
    """A run was asked to do something the evaluation rules forbid."""


@dataclass(frozen=True)
class RunConfig:
    """One retrieval system: what runs, how deep, and how much context is kept.

    ``bm25_weight`` is set only for ``hybrid``, where TF-IDF gets ``1 - bm25_weight``
    in weighted RRF (``k = 60``). ``context_k`` passages are kept and scored.
    """

    system: str
    depth: int
    context_k: int
    bm25_weight: float | None = None

    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise LiveRunError(f"system must be one of {SYSTEMS}")
        for name in ("depth", "context_k"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= MAX_DEPTH:
                raise LiveRunError(f"{name} must be an int from 1 to {MAX_DEPTH}")
        if self.context_k > self.depth:
            raise LiveRunError("context_k cannot exceed depth")
        weight = self.bm25_weight
        if (self.system == "hybrid") != (weight is not None):
            raise LiveRunError("bm25_weight is required for hybrid and only for hybrid")
        if weight is not None and (type(weight) is not float or not 0 < weight < 1):
            raise LiveRunError("bm25_weight must be a float strictly between 0 and 1")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


def _grid() -> tuple[RunConfig, ...]:
    shapes = [(d, c) for d in (3, 5, 10) for c in (1, 2, 3, 5) if c <= d]
    configs = [RunConfig(s, d, c) for s in RETRIEVERS for d, c in shapes]
    configs += [RunConfig("hybrid", d, c, w) for w in (0.25, 0.5, 0.75) for d, c in shapes]
    return tuple(configs)


GRID: Final = _grid()


def retrieve(dataset: Dataset, case: Case, query: str, config: RunConfig) -> tuple[str, ...]:
    """The ``context_k`` passage ids ``config`` keeps for ``query``, best first."""
    pool = eligible_passages(dataset, case)
    lists: dict[str, Ranking] = {
        name: RANKERS[name](pool, query, config.depth) for name in RETRIEVERS
    }
    if config.system == "hybrid":
        weight = config.bm25_weight
        assert weight is not None  # RunConfig guarantees it for hybrid
        trace = fuse(
            lists,
            method="rrf",
            k=RRF_K,
            weights={"bm25": weight, "tfidf": 1 - weight},
            eligible=set(pool),
            top_k=config.context_k,
        )
        kept = trace.keys
    else:
        kept = tuple(pid for pid, _ in lists[config.system][: config.context_k])
    for pid in kept:
        source_id = pid.partition("#")[0]
        if pid not in pool or ineligibility(dataset.sources[source_id], case):
            raise LiveRunError("a retriever returned a passage outside the eligible pool")
    return kept


def _query_record(dataset: Dataset, case: Case, qid: str, query: str, config: RunConfig) -> dict:
    kept = retrieve(dataset, case, query, config)
    relevant = sorted({ref.label for ref in case.evidence})
    record = {
        "query_id": qid,
        "category": case.category,
        "outcome": case.outcome,
        "eligible": len(eligible_passages(dataset, case)),
        "relevant": relevant,
        "retrieved": list(kept),
    }
    if relevant:
        k = config.context_k
        record["recall"] = round(recall_at_k(kept, relevant, k), METRIC_DECIMALS)
        record["rr"] = round(reciprocal_rank(kept, relevant), METRIC_DECIMALS)
        record["ndcg"] = round(ndcg_at_k(kept, relevant, k), METRIC_DECIMALS)
    return record


def evaluate(dataset: Dataset, config: RunConfig, split: str, *, held_out: bool = False) -> dict:
    """Run ``config`` on every query of ``split``; refuses test unless ``held_out``."""
    if split not in SPLITS:
        raise LiveRunError(f"split must be one of {SPLITS}")
    if split == "test" and not held_out:
        raise LiveRunError("test is held out: score it only with a frozen config and held_out=True")
    records = [
        _query_record(dataset, case, qid, query, config)
        for case in dataset.cases
        if case.split == split
        for qid, query in zip(case.query_ids, case.queries)
    ]
    scored = [r for r in records if "recall" in r]
    unscored = [r for r in records if "recall" not in r]

    def mean(name: str) -> float | None:
        values = [r[name] for r in scored]
        return round(sum(values) / len(values), METRIC_DECIMALS) if values else None

    return {
        "split": split,
        "config": asdict(config),
        "config_sha256": config.sha256,
        "queries": len(records),
        "scored": len(scored),
        "recall": mean("recall"),
        "mrr": mean("rr"),
        "ndcg": mean("ndcg"),
        "no_evidence": len(unscored),
        "no_evidence_retrieved_something": sum(1 for r in unscored if r["retrieved"]),
        "per_query": records,
    }


def _selection_key(item: tuple[int, dict]) -> tuple:
    index, run = item
    config = run["config"]
    return (
        -(run["recall"] or 0.0),
        -(run["mrr"] or 0.0),
        config["context_k"],
        config["depth"],
        index,
    )


def tune(dataset: Dataset, split: str = "dev", grid: Sequence[RunConfig] = GRID) -> dict:
    """Evaluate ``grid`` on a tuning split and select one configuration.

    Rule, declared before any run: highest mean recall at ``context_k``, then
    highest MRR, then the smaller context, then the smaller depth, then grid order.
    """
    if split not in TUNING_SPLITS:
        raise LiveRunError(f"tuning may use only {TUNING_SPLITS}; test is held out")
    if not grid:
        raise LiveRunError("the grid is empty")
    runs = [evaluate(dataset, config, split) for config in grid]
    _, best = min(enumerate(runs), key=_selection_key)
    return {
        "split": split,
        "rule": "max recall@context_k, then MRR, then smaller context_k, then smaller depth",
        "selected": {"config": best["config"], "config_sha256": best["config_sha256"]},
        "selected_run": best,
        "grid": [{k: v for k, v in run.items() if k != "per_query"} for run in runs],
    }


def provenance(dataset: Dataset) -> dict:
    """Hashes of the harness code and frozen data, the ragfuse build, and the model used."""
    here = Path(__file__).resolve().parent
    files = {
        name: canonical_sha256((here / name).read_bytes().replace(b"\r\n", b"\n").decode("utf-8"))
        for name in CODE_FILES
    }
    return {
        "code": {"files": files, "ragfuse": _ragfuse_build()},
        "data": {
            "dataset_version": dataset.version,
            "fixture_sha256": dataset.fixture_sha256,
            "cases_sha256": dataset.cases_sha256,
        },
        "model": MODEL,
    }


def _ragfuse_build() -> dict:
    """ragfuse's version and, for a Git install, its commit; never a local path."""
    dist = metadata.distribution("ragfuse")
    commit = "unknown"
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id", "unknown")
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"version": dist.version, "commit": commit}


def render_markdown(result: dict) -> str:
    """A short, human-readable view of the tuning result."""
    selected = result["tuning"]["selected_run"]
    ragfuse = result["provenance"]["code"]["ragfuse"]
    ties = sum(
        1
        for run in result["tuning"]["grid"]
        if (run["recall"], run["mrr"]) == (selected["recall"], selected["mrr"])
    )
    lines = [
        "# Legal demo: live retrieval runs, dev tuning",
        "",
        "Fictional data, not law and not legal advice. Generated by",
        "`python -m rageval.legal_demo.live`; edits here are overwritten.",
        f"Every row ran the retrievers on the {selected['queries']} dev queries. Test was not run.",
        "",
        "Lexical retrievers only (BM25, TF-IDF cosine and their weighted RRF via ragfuse);",
        "no embedding or language model was run, so there is no learned-dense row.",
        "With a few eligible sections per case these numbers check the code, not quality.",
        "",
        f"- Code: `{result['provenance']['code']['files']['live.py']}` (live.py)",
        f"- ragfuse: {ragfuse['version']} at commit `{ragfuse['commit']}`",
        f"- Fixture: `{result['provenance']['data']['fixture_sha256']}`",
        f"- Cases: `{result['provenance']['data']['cases_sha256']}`",
        f"- Rule: {result['tuning']['rule']}",
        f"- Selected: `{json.dumps(selected['config'])}` (`{selected['config_sha256']}`)",
        f"- {ties} of {len(result['tuning']['grid'])} configurations tie with it on recall"
        + " and MRR, so the tie-breaks, not the data, chose among them.",
        "",
        "| System | bm25 weight | Depth | Context | Recall | MRR | nDCG | No-evidence queries with hits |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in result["tuning"]["grid"]:
        c = run["config"]
        weight = "-" if c["bm25_weight"] is None else c["bm25_weight"]
        hits = f"{run['no_evidence_retrieved_something']} of {run['no_evidence']}"
        lines.append(
            f"| {c['system']} | {weight} | {c['depth']} | {c['context_k']} | "
            f"{run['recall']} | {run['mrr']} | {run['ndcg']} | {hits} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rageval.legal_demo.live",
        description="Run the lexical retrievers on the frozen legal-demo dev split and tune.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    root = parser.parse_args(argv).root
    try:
        dataset = load_dataset(root)
    except DatasetError as exc:
        raise SystemExit(f"rageval.legal_demo.live: error: {exc}") from exc
    result = {"provenance": provenance(dataset), "tuning": tune(dataset, "dev")}
    outputs = {
        "dev-tuning.json": json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        "DEV_TUNING.md": render_markdown(result),
    }
    out_dir = root / OUT_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in outputs.items():
            (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise SystemExit(
            f"rageval.legal_demo.live: error: cannot write {OUT_DIR} ({type(exc).__name__})"
        ) from exc
    selected = result["tuning"]["selected"]
    print(f"ran {len(GRID)} configurations on dev; selected {json.dumps(selected['config'])}")
    for name in outputs:
        print(f"wrote {OUT_DIR}/{name}")


if __name__ == "__main__":
    main()
