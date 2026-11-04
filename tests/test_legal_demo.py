"""Tests for the legal-demo evaluation set and its frozen splits.

The committed files are pinned (counts, categories, groups, report bytes), the
eligibility rule is unit-tested on the fixture, and each consistency check is
shown to fire by tampering with a copy of the data. Tampered copies are
re-frozen first, unless the test is about the freeze, so that only the
targeted check can fail.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from rageval.legal_demo import SPLITS, DatasetError, canonical_sha256, ineligibility, load_dataset
from rageval.legal_demo.model import Case, closest_cross_split, leakage_groups
from rageval.legal_demo.report import main

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "legal-demo"
FILES = ("sources.json", "cases.json", "manifest.json")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(REPO)


def _case(**overrides) -> Case:
    fields = {
        "id": "ld-900", "split": "dev", "issue": "toy", "category": "answerable",
        "queries": ("q",), "tenant": "public", "matter": None, "jurisdiction": "fic-varnholt",
        "regime": "maritime-safety", "as_of": date(2024, 6, 1), "outcome": "answer",
        "answer": "a", "evidence": (), "traps": (), "author": "ai-assistant",
        "method": "fixture-reading", "note": "n",
    }  # fmt: skip
    return Case(**{**fields, **overrides})


def _variant(tmp_path: Path, edit=None, *, refreeze: bool = True) -> Path:
    """Copy data/legal-demo, apply ``edit(docs)`` to the parsed files, re-freeze, write."""
    shutil.copytree(DATA, tmp_path / "data" / "legal-demo")
    docs = {name: json.loads((DATA / name).read_text(encoding="utf-8")) for name in FILES}
    if edit:
        edit(docs)
    if refreeze:
        manifest, cases = docs["manifest.json"], docs["cases.json"]["cases"]
        manifest["fixture"]["sha256"] = canonical_sha256(docs["sources.json"])
        manifest["cases"]["sha256"] = canonical_sha256(docs["cases.json"])
        manifest["splits"] = {
            s: {
                "cases": sum(1 for c in cases if c.get("split") == s),
                "queries": sum(len(c["queries"]) for c in cases if c.get("split") == s),
            }
            for s in SPLITS
        }
    for name, doc in docs.items():
        (tmp_path / "data" / "legal-demo" / name).write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


def _by_id(docs: dict, case_id: str) -> dict:
    return next(c for c in docs["cases.json"]["cases"] if c["id"] == case_id)


# --- the committed set ------------------------------------------------------


def test_committed_set_is_frozen_with_29_cases_and_45_queries(dataset) -> None:
    assert dataset.version == 1
    counts = {s: (sum(c.split == s for c in dataset.cases), sum(len(c.queries) for c in dataset.cases if c.split == s)) for s in SPLITS}  # fmt: skip
    assert counts == {"train": (7, 11), "dev": (11, 15), "test": (11, 19)}


def test_every_category_is_present_and_test_holds_all_five(dataset) -> None:
    by_split = {s: {c.category for c in dataset.cases if c.split == s} for s in SPLITS}
    assert len(by_split["test"]) == 5
    # The fixture has one conflict, so conflicting-source cases fit in one split only.
    assert "conflicting-source" not in by_split["dev"] | by_split["train"]


def test_no_leakage_group_spans_two_splits(dataset) -> None:
    assert len(dataset.groups) == 8
    assert all(len({c.split for c in group}) == 1 for group in dataset.groups)


def test_every_trap_is_ineligible_and_every_evidence_eligible(dataset) -> None:
    for case in dataset.cases:
        assert all(not ineligibility(dataset.sources[r.source_id], case) for r in case.evidence)
        assert all(ineligibility(dataset.sources[r.source_id], case) for r in case.traps)


def test_closest_cross_split_queries_stay_under_the_limit(dataset) -> None:
    score, pair = dataset.closest_cross_split
    assert pair == ("ld-006.q1", "ld-029.q1")
    assert round(score, 4) == 0.4444


# --- the eligibility rule ---------------------------------------------------


@pytest.mark.parametrize(
    ("source_id", "overrides", "reasons"),
    [
        ("fic-harbour-safety-act@v2", {}, ()),
        ("fic-harbour-safety-act@v2", {"as_of": date(2023, 7, 1)}, ()),
        ("fic-harbour-safety-act@v1", {"as_of": date(2023, 7, 1)}, ("no-longer-in-force",)),
        ("fic-harbour-safety-act@v1", {"as_of": date(2023, 6, 30)}, ()),
        ("fic-harbour-safety-act@v1", {"as_of": date(2018, 12, 31)}, ("not-yet-in-force",)),
        ("fic-quillmouth-port-code@v1", {"jurisdiction": "fic-quillmouth"}, ("unknown-in-force",)),
        ("fic-quillmouth-port-code@v1", {}, ("other-jurisdiction", "unknown-in-force")),
        ("fic-harbour-safety-act@v2", {"regime": "fic-other"}, ("other-regime",)),
        ("fic-orchid-fleet-memo@v1", {}, ("other-tenant",)),
        ("fic-orchid-fleet-memo@v1", {"tenant": "tenant-orchid"}, ("other-matter",)),
        (
            "fic-orchid-fleet-memo@v1",
            {"tenant": "tenant-orchid", "matter": "matter-orchid-001"},
            (),
        ),
        (
            "fic-basalt-file-note@v1",
            {"tenant": "tenant-basalt"},
            ("unknown-jurisdiction", "unknown-regime", "unknown-in-force"),
        ),
    ],
)
def test_ineligibility_fails_closed(dataset, source_id, overrides, reasons) -> None:
    assert ineligibility(dataset.sources[source_id], _case(**overrides)) == reasons


def test_leakage_groups_join_transitively_and_sort() -> None:
    cases = [_case(id=f"ld-90{n}") for n in range(4)]
    links = {"ld-900": ["x"], "ld-901": ["y"], "ld-902": ["x", "y"], "ld-903": ["z"]}
    groups = leakage_groups(cases, lambda c: links[c.id])
    assert [[c.id for c in g] for g in groups] == [["ld-900", "ld-901", "ld-902"], ["ld-903"]]


def test_closest_cross_split_ignores_pairs_inside_one_split() -> None:
    cases = [
        _case(id="ld-900", split="dev", queries=("hull inspection fee", "hull inspection fee now")),
        _case(id="ld-901", split="test", queries=("berthing fine",)),
    ]
    assert closest_cross_split(cases) == (0.0, ("", ""))


# --- every check fires on a tampered copy -----------------------------------


def _set(case_id: str, path: tuple, value):
    def edit(docs):
        target = _by_id(docs, case_id)
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


TAMPERS = {
    "quote": (_set("ld-001", ("expected", "evidence", 0, "quote"), "every 6 months"), "does not occur exactly once"),
    "evidence-not-eligible": (_set("ld-001", ("as_of",), "2022-01-01"), "is not eligible (not-yet-in-force)"),
    "eligible-trap": (_set("ld-022", ("traps",), [{"source_id": "fic-harbour-safety-act@v2", "section_id": "s1"}]), "so it is not a trap"),
    "leakage": (_set("ld-002", ("split",), "dev"), "leakage: cases ld-001"),
    "outcome": (_set("ld-011", ("expected", "outcome"), "answer"), "does not allow outcome answer"),
    "abstain-with-evidence": (_set("ld-008", ("expected", "evidence"), [{"source_id": "fic-harbour-safety-act@v2", "section_id": "s2", "quote": "5 years"}]), "an abstention cites no evidence"),
    "answer-without-evidence": (_set("ld-017", ("expected", "evidence"), []), "outcome answer needs evidence"),
    "one-family-conflict": (_set("ld-005", ("expected", "evidence"), [{"source_id": "fic-harbour-inspection-guidance@v1", "section_id": "s2", "quote": "This note is guidance."}]), "two source families"),
    "wrong-regime-trap": (_set("ld-020", ("traps",), []), "needs a trap that fails other-jurisdiction or other-regime"),
    "outdated-trap": (_set("ld-021", ("traps",), []), "needs a trap that fails not-yet-in-force"),
    "unknown-source": (_set("ld-022", ("traps",), [{"source_id": "fic-nowhere@v1", "section_id": "s1"}]), "is repeated or not in the fixture"),
    "repeated-ref": (_set("ld-017", ("traps",), [{"source_id": "fic-harbour-safety-act@v2", "section_id": "s2"}]), "is repeated or not in the fixture"),
    "public-matter": (_set("ld-022", ("scope", "matter"), "matter-orchid-001"), "a public requester has no matter"),
    "answer-text": (_set("ld-017", ("expected", "answer"), None), "'answer' must be null"),
    "author": (_set("ld-022", ("label", "author"), "someone"), "author or method is not declared"),
    "note": (_set("ld-022", ("note",), " "), "'note' is empty"),
    "query-space": (_set("ld-022", ("queries",), [" padded"]), "every query must be trimmed"),
    "query-twice": (_set("ld-022", ("queries",), ["same", "same"]), "a query is listed twice"),
    "unknown-regime": (_set("ld-022", ("regime",), "unknown"), "regime cannot be unknown"),
    "near-duplicate": (_set("ld-029", ("queries",), ["How often must a 40-metre vessel that berths in a Varnholt harbour pass a hull inspection?"]), "content words"),
    "real-jurisdiction": (_set("ld-022", ("jurisdiction",), "in"), "not a valid jurisdiction"),
    "date-format": (_set("ld-022", ("as_of",), "2024-6-1"), "not a YYYY-MM-DD date"),
    "extra-key": (_set("ld-022", ("extra",), 1), "unexpected keys ['extra']"),
    "bool-version": (lambda d: d["cases.json"].update(dataset_version=True), "'dataset_version' has the wrong type"),
    "zero-version": (lambda d: d["cases.json"].update(dataset_version=0), "must be at least 1"),
    "empty-split": (lambda d: [c.update(split="dev") for c in d["cases.json"]["cases"] if c["split"] == "train"], "split train has no cases"),
    "duplicate-id": (_set("ld-029", ("id",), "ld-028"), "ld-028: duplicate id"),
    "unlisted-provision": (lambda d: d["cases.json"]["provision_groups"].pop("basalt-file-note"), "provisions in no provision group"),
    "provision-twice": (lambda d: d["cases.json"]["provision_groups"]["basalt-file-note"]["provisions"].append("fic-harbour-safety-act#s3"), "is in two provision groups"),
    "fixture-text": (lambda d: d["sources.json"]["sources"][0].update(text="changed"), "content_hash does not match"),
    "manifest-version": (lambda d: d["manifest.json"].update(dataset_version=2), "pins dataset_version 2"),
    "manifest-dataset": (lambda d: d["manifest.json"].update(dataset="other"), "not a legal-demo manifest"),
    "manifest-schema": (lambda d: d["manifest.json"].update(schema_version=2), "'schema_version' must be 1"),
    "manifest-policy": (lambda d: d["manifest.json"].update(policy=[]), "'policy' must list non-empty statements"),
    "manifest-cases": (lambda d: d["manifest.json"]["cases"].update(extra=1), "unexpected keys ['extra']"),
    "cases-schema": (lambda d: d["cases.json"].update(schema="other"), "not a version-1 rageval-legal-demo-cases file"),
    "not-fictional": (lambda d: d["cases.json"].update(fictional=False), "'fictional' must be true"),
    "no-cases": (lambda d: d["cases.json"].update(cases=[]), "cases.json holds no usable case"),
    "no-authors": (lambda d: d["cases.json"].update(label_authors={}), "must map ids to non-empty descriptions"),
    "group-name": (lambda d: d["cases.json"]["provision_groups"].update({"Bad Name": {"why": "w", "provisions": []}}), "provision group name is not a slug"),
    "group-why": (lambda d: d["cases.json"]["provision_groups"]["basalt-file-note"].update(why=""), "'why' is empty"),
    "group-unknown-provision": (lambda d: d["cases.json"]["provision_groups"]["basalt-file-note"]["provisions"].append("fic-x#s9"), "a provision is not in the fixture"),
    "split-name": (_set("ld-022", ("split",), "holdout"), "'split' must be one of train, dev, test"),
    "no-queries": (_set("ld-022", ("queries",), []), "'queries' must be a non-empty list of strings"),
    "missing-field": (lambda d: d["sources.json"]["sources"][0].pop("regime"), "missing 'regime'"),
    "section-not-object": (lambda d: d["sources.json"]["sources"][5].update(sections=["x"]), "expected an object holding 'start'"),
    "section-span": (lambda d: d["sources.json"]["sources"][5]["sections"][0].update(end=10_000), "section span falls outside the text"),
    "section-twice": (lambda d: d["sources.json"]["sources"][4]["sections"][1].update(section_id="s1"), "duplicate section_id"),
    "source-twice": (lambda d: d["sources.json"]["sources"][1].update(source_id="fic-harbour-safety-act@v1"), "duplicate source_id"),
    "no-sources": (lambda d: d["sources.json"].update(sources=[]), "sources.json holds no usable source"),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_each_check_rejects_its_tamper(tmp_path, name) -> None:
    edit, message = TAMPERS[name]
    with pytest.raises(DatasetError, match="problem") as error:
        load_dataset(_variant(tmp_path, edit))
    assert message in str(error.value)


def test_unrefrozen_edit_is_not_the_frozen_version(tmp_path) -> None:
    root = _variant(tmp_path, _set("ld-001", ("note",), "edited"), refreeze=False)
    with pytest.raises(DatasetError, match="cases.json is not the frozen version"):
        load_dataset(root)


def test_manifest_counts_must_match(tmp_path) -> None:
    def edit(docs):
        docs["manifest.json"]["splits"]["test"]["queries"] = 18

    with pytest.raises(DatasetError, match="split counts do not match"):
        load_dataset(_variant(tmp_path, edit, refreeze=False))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"a": 1, "a": 2}', "duplicate key 'a'"),
        ("{not json", "not valid UTF-8 JSON"),
        ("[" * 100_000, "not valid UTF-8 JSON"),
    ],
)
def test_unreadable_json_is_a_dataset_error(tmp_path, text, message) -> None:
    root = _variant(tmp_path)
    (root / "data" / "legal-demo" / "cases.json").write_text(text)
    with pytest.raises(DatasetError, match=message):
        load_dataset(root)


@pytest.mark.parametrize(
    ("name", "doc", "message"),
    [
        ("sources.json", {"schema": "other"}, "not a version-1 minirag-legal-fixture"),
        ("cases.json", [], "cases.json: expected an object"),
        ("manifest.json", {}, "manifest.json: missing keys"),
    ],
)
def test_wrong_file_shapes_are_reported(tmp_path, name, doc, message) -> None:
    root = _variant(tmp_path)
    (root / "data" / "legal-demo" / name).write_text(json.dumps(doc))
    with pytest.raises(DatasetError, match=message):
        load_dataset(root)


# --- the command ------------------------------------------------------------


def test_cli_rewrites_the_committed_report_byte_for_byte(tmp_path) -> None:
    shutil.copytree(DATA, tmp_path / "data" / "legal-demo")
    proc = subprocess.run(
        [sys.executable, "-m", "rageval.legal_demo", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "cases (queries) train 7 (11), dev 11 (15), test 11 (19)" in proc.stdout
    for name in ("splits.json", "SPLITS.md"):
        written = (tmp_path / "results" / "legal-demo" / name).read_bytes()
        assert written == (REPO / "results" / "legal-demo" / name).read_bytes()


def test_cli_error_exits_cleanly_without_writing(tmp_path) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["--root", str(tmp_path)])
    assert str(stop.value).startswith("rageval.legal_demo: error: missing file:")
    assert not (tmp_path / "results").exists()
