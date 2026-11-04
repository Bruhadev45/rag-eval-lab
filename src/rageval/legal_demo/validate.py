"""Load the legal-demo set from disk and check it against the fixture and the freeze.

The set lives in ``data/legal-demo/``, apart from the historical 120/204-query
benchmark, and has three files:

- ``sources.json`` -- minirag's fictional legal fixture, copied unchanged.
- ``cases.json`` -- questions with an expected outcome, verbatim evidence
  quotes, traps (sources a correct system must not use), the label author and
  method, and one split per case.
- ``manifest.json`` -- the freeze: dataset version, a hash of each file, and
  the number of cases and queries in each split.

``load_dataset`` re-derives from the fixture every label claim it can check
and raises ``DatasetError`` listing the problems found. The checks are
structural. They show a label agrees with the fixture, not that it is a good
label, and nothing here measures retrieval quality or legal accuracy.

Leakage groups join cases that share an issue or touch the same provision
group (``family#section`` sets declared in ``cases.json``). A provision group
covers every version of a section, so a superseded rule and its replacement
always sit in one split. Every leakage group must sit inside one split.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .model import (
    CATEGORIES,
    CATEGORY_OUTCOMES,
    CATEGORY_TRAP_REASONS,
    DATA_DIR,
    MAX_CROSS_SPLIT_JACCARD,
    MAX_QUERY_CHARS,
    OUTCOMES,
    PUBLIC,
    SPLITS,
    UNKNOWN,
    Case,
    Dataset,
    DatasetError,
    Ref,
    Source,
    canonical_sha256,
    closest_cross_split,
    ineligibility,
    leakage_groups,
    split_counts,
)

# Bounds on input size: the near-duplicate screen compares every pair of queries.
MAX_FILE_BYTES = 1_000_000
MAX_QUERIES = 2_000
MAX_PROBLEMS_SHOWN = 50

_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_PATTERNS = {  # ASCII only: [0-9], never \d, which also matches other scripts' digits
    "case_id": re.compile(r"ld-[0-9]{3}"),
    "slug": re.compile(_SLUG),
    "section_id": re.compile(r"s[1-9][0-9]*"),
    "tenant": re.compile(rf"public|tenant-{_SLUG}"),
    "matter": re.compile(rf"matter-{_SLUG}"),
    "jurisdiction": re.compile(rf"fic-{_SLUG}"),
    "source_jurisdiction": re.compile(rf"unknown|fic-{_SLUG}"),
    "date": re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}"),
}
_CASES_KEYS = {
    "schema", "schema_version", "dataset_version", "fictional", "notice",
    "label_authors", "label_methods", "provision_groups", "cases",
}  # fmt: skip
_CASE_KEYS = {
    "id", "split", "issue", "category", "queries", "scope", "jurisdiction",
    "regime", "as_of", "expected", "traps", "label", "note",
}  # fmt: skip
_MANIFEST_KEYS = {
    "schema", "schema_version", "dataset", "dataset_version", "frozen_on",
    "fixture", "cases", "splits", "policy",
}  # fmt: skip


def load_dataset(root: Path) -> Dataset:
    """Load ``<root>/data/legal-demo``, check it, and return it.

    Raises:
        DatasetError: A file is missing or malformed, a label disagrees with the
            fixture, a leakage group spans splits, two queries in different
            splits are near-duplicates, or the files are not the frozen version
            the manifest pins.
    """
    data_dir = root / DATA_DIR
    (raw_sources, fixture_hash), (raw_cases, cases_hash), (manifest, _) = (
        _read_json(data_dir / name) for name in ("sources.json", "cases.json", "manifest.json")
    )
    problems: list[str] = []
    sources = _parse_sources(raw_sources, problems)
    version, cases, owner = _parse_cases(raw_cases, sources, problems)
    if sum(len(case.queries) for case in cases) > MAX_QUERIES:
        problems.append(f"more than {MAX_QUERIES} queries")
    _raise_if(problems)
    groups = leakage_groups(cases, _links(sources, owner))
    _check_splits(cases, groups, problems)
    closest = closest_cross_split(cases)
    if closest[0] >= MAX_CROSS_SPLIT_JACCARD:
        problems.append(
            f"queries {closest[1][0]} and {closest[1][1]} sit in different splits but "
            f"share {closest[0]:.2f} of their content words (limit {MAX_CROSS_SPLIT_JACCARD})"
        )
    hashes = (fixture_hash, cases_hash)
    _check_manifest(manifest, version, hashes, cases, problems)
    _raise_if(problems)
    return Dataset(
        version=version,
        sources=MappingProxyType(sources),
        cases=cases,
        provision_group_of=MappingProxyType(owner),
        groups=groups,
        fixture_sha256=hashes[0],
        cases_sha256=hashes[1],
        closest_cross_split=closest,
    )


# --- parsing ------------------------------------------------------------------


def _raise_if(problems: list[str]) -> None:
    if problems:
        shown = [f"- {problem}" for problem in problems[:MAX_PROBLEMS_SHOWN]]
        if len(problems) > MAX_PROBLEMS_SHOWN:
            shown.append(f"- and {len(problems) - MAX_PROBLEMS_SHOWN} more")
        raise DatasetError(f"{len(problems)} problem(s) in {DATA_DIR}:\n" + "\n".join(shown))


def _read_json(path: Path) -> tuple[object, str]:
    """The parsed file and its canonical hash. Every failure is a DatasetError."""

    def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DatasetError(f"{path.name}: duplicate key {key[:40]!r}")
            result[key] = value
        return result

    def no_constant(name: str) -> object:
        raise DatasetError(f"{path.name}: {name} is not allowed")

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise DatasetError(f"{path.name} is larger than {MAX_FILE_BYTES} bytes")
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, object_pairs_hook=unique_keys, parse_constant=no_constant)
        # Hashing encodes to UTF-8, so a lone surrogate escape fails here too.
        return value, canonical_sha256(value)
    except DatasetError:
        raise
    except FileNotFoundError:
        raise DatasetError(f"missing file: {path}") from None
    except OSError as exc:
        raise DatasetError(f"{path.name}: cannot be read ({type(exc).__name__})") from None
    except (ValueError, RecursionError) as exc:  # includes UnicodeDecode/EncodeError
        raise DatasetError(f"{path.name}: not valid UTF-8 JSON ({type(exc).__name__})") from None


def _get(obj: object, key: str, kind: type | tuple[type, ...]) -> Any:
    if not isinstance(obj, dict):
        raise TypeError(f"expected an object holding {key!r}")
    if key not in obj:
        raise ValueError(f"missing {key!r}")
    value = obj[key]
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise ValueError(f"{key!r} has the wrong type")
    return value


def _match(obj: object, key: str, pattern: str) -> str:
    value = _get(obj, key, str)
    if not _PATTERNS[pattern].fullmatch(value):
        raise ValueError(f"{key!r} is not a valid {pattern}")
    return value


def _choice(obj: object, key: str, allowed: tuple[str, ...]) -> str:
    value = _get(obj, key, str)
    if value not in allowed:
        raise ValueError(f"{key!r} must be one of {', '.join(allowed)}")
    return value


def _exact_keys(obj: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError("expected an object")
    missing, extra = sorted(keys - obj.keys()), sorted(obj.keys() - keys)
    if missing or extra:
        raise ValueError(
            f"missing keys {missing[:5]}, unexpected keys {[k[:40] for k in extra[:5]]}"
        )
    return obj


def _date(obj: object, key: str) -> date:
    value = _get(obj, key, str)
    if not _PATTERNS["date"].fullmatch(value):
        raise ValueError(f"{key!r} is not a YYYY-MM-DD date")
    return date.fromisoformat(value)


def _source_date(obj: object, key: str) -> date | str | None:
    """A fixture date: YYYY-MM-DD, "unknown", or null (no end date)."""
    value = _get(obj, key, (str, type(None)))
    return value if value is None or value == UNKNOWN else _date(obj, key)


def _parse_sources(raw: object, problems: list[str]) -> dict[str, Source]:
    head = isinstance(raw, dict) and raw.get("schema") == "minirag-legal-fixture"
    if not (head and _is_one(raw.get("schema_version")) and raw.get("fictional") is True):
        problems.append("sources.json is not a version-1 minirag-legal-fixture marked fictional")
        return {}
    sources: dict[str, Source] = {}
    records = raw.get("sources")
    for index, record in enumerate(records if isinstance(records, list) else []):
        try:
            source = _parse_source(record)
        except (TypeError, ValueError) as exc:
            problems.append(f"sources.json record {index}: {exc}")
            continue
        if source.source_id in sources:
            problems.append(f"sources.json record {index}: duplicate source_id")
        sources[source.source_id] = source
    if not sources:
        problems.append("sources.json holds no usable source")
    return sources


def _parse_source(record: object) -> Source:
    text = _get(record, "text", str)
    digest = hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()
    if _get(record, "content_hash", str) != f"sha256:{digest}":
        raise ValueError("content_hash does not match the text")
    family = _match(record, "family_id", "slug")
    source_id = _get(record, "source_id", str)
    if not re.fullmatch(rf"{re.escape(family)}@v[1-9][0-9]*", source_id):
        raise ValueError("'source_id' is not <family_id>@v<version>")
    tenant = _match(record, "tenant", "tenant")
    matter = _get(record, "matter", (str, type(None)))
    if matter is not None:
        matter = _match(record, "matter", "matter")
        if tenant == PUBLIC:
            raise ValueError("a public source has no matter")
    start, end = _source_date(record, "effective_from"), _source_date(record, "effective_to")
    if start is None:
        raise ValueError("'effective_from' cannot be null")
    if isinstance(start, date) and isinstance(end, date) and end <= start:
        raise ValueError("'effective_to' must be after 'effective_from'")
    return Source(
        source_id=source_id,
        family_id=family,
        tenant=tenant,
        matter=matter,
        jurisdiction=_match(record, "jurisdiction", "source_jurisdiction"),
        regime=_match(record, "regime", "slug"),
        effective_from=start,
        effective_to=end,
        sections=MappingProxyType(_sections(record, text)),
    )


def _sections(record: object, text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for section in _get(record, "sections", list):
        start, end = _get(section, "start", int), _get(section, "end", int)
        if not 0 <= start < end <= len(text):
            raise ValueError("a section span falls outside the text")
        section_id = _match(section, "section_id", "section_id")
        if section_id in sections:
            raise ValueError("duplicate section_id")
        sections[section_id] = text[start:end]
    return sections


def _parse_cases(
    raw: object, sources: Mapping[str, Source], problems: list[str]
) -> tuple[int, tuple[Case, ...], dict[str, str]]:
    try:
        _exact_keys(raw, _CASES_KEYS)
        if raw["schema"] != "rageval-legal-demo-cases" or not _is_one(raw["schema_version"]):
            raise ValueError("not a version-1 rageval-legal-demo-cases file")
        if raw["fictional"] is not True:
            raise ValueError("'fictional' must be true")
        version = _get(raw, "dataset_version", int)
        if version < 1:
            raise ValueError("'dataset_version' must be at least 1")
        authors, methods = (_text_map(raw, key) for key in ("label_authors", "label_methods"))
        owner = _provision_groups(_get(raw, "provision_groups", dict), sources)
        records = _get(raw, "cases", list)
    except (TypeError, ValueError) as exc:
        problems.append(f"cases.json: {exc}")
        return 0, (), {}
    cases: list[Case] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        where = record.get("id") if isinstance(record, dict) else None
        try:
            case = _parse_case(record)
        except (TypeError, ValueError) as exc:
            label = where[:20] if isinstance(where, str) else f"#{index}"
            problems.append(f"case {label}: {exc}")
            continue
        if case.id in seen_ids:
            problems.append(f"case {case.id}: duplicate id")
        seen_ids.add(case.id)
        problems.extend(_check_case(case, sources, authors, methods))
        cases.append(case)
    if not cases:
        problems.append("cases.json holds no usable case")
    return version, tuple(cases), owner


def _text_map(raw: dict, key: str) -> dict[str, str]:
    value = _get(raw, key, dict)
    if not value or not all(isinstance(v, str) and v.strip() for v in value.values()):
        raise ValueError(f"{key!r} must map ids to non-empty descriptions")
    return value


def _provision_groups(raw: dict, sources: Mapping[str, Source]) -> dict[str, str]:
    known = {f"{s.family_id}#{section}" for s in sources.values() for section in s.sections}
    owner: dict[str, str] = {}
    for name, spec in raw.items():
        if not _PATTERNS["slug"].fullmatch(name):
            raise ValueError("a provision group name is not a slug")
        _exact_keys(spec, {"why", "provisions"})
        if not isinstance(spec["why"], str) or not spec["why"].strip():
            raise ValueError(f"provision group {name}: 'why' is empty")
        for provision in _get(spec, "provisions", list):
            if not isinstance(provision, str) or provision not in known:
                raise ValueError(f"provision group {name}: a provision is not in the fixture")
            if provision in owner:
                raise ValueError(f"{provision} is in two provision groups")
            owner[provision] = name
    if known - owner.keys():
        raise ValueError(f"provisions in no provision group: {sorted(known - owner.keys())}")
    return owner


def _parse_case(record: object) -> Case:
    _exact_keys(record, _CASE_KEYS)
    scope = _exact_keys(_get(record, "scope", dict), {"tenant", "matter"})
    expected = _exact_keys(_get(record, "expected", dict), {"outcome", "answer", "evidence"})
    label = _exact_keys(_get(record, "label", dict), {"author", "method"})
    queries = _get(record, "queries", list)
    if not queries or not all(isinstance(q, str) for q in queries):
        raise ValueError("'queries' must be a non-empty list of strings")
    return Case(
        id=_match(record, "id", "case_id"),
        split=_choice(record, "split", SPLITS),
        issue=_match(record, "issue", "slug"),
        category=_choice(record, "category", CATEGORIES),
        queries=tuple(queries),
        tenant=_match(scope, "tenant", "tenant"),
        matter=None if scope["matter"] is None else _match(scope, "matter", "matter"),
        jurisdiction=_match(record, "jurisdiction", "jurisdiction"),
        regime=_match(record, "regime", "slug"),
        as_of=_date(record, "as_of"),
        outcome=_choice(expected, "outcome", OUTCOMES),
        answer=_get(expected, "answer", (str, type(None))),
        evidence=tuple(_ref(item, quote=True) for item in _get(expected, "evidence", list)),
        traps=tuple(_ref(item, quote=False) for item in _get(record, "traps", list)),
        author=_get(label, "author", str),
        method=_get(label, "method", str),
        note=_get(record, "note", str),
    )


def _ref(item: object, *, quote: bool) -> Ref:
    keys = {"source_id", "section_id", "quote"} if quote else {"source_id", "section_id"}
    _exact_keys(item, keys)
    return Ref(
        source_id=_get(item, "source_id", str),
        section_id=_get(item, "section_id", str),
        quote=_get(item, "quote", str) if quote else None,
    )


# --- consistency checks -----------------------------------------------------


def _check_case(
    case: Case,
    sources: Mapping[str, Source],
    authors: Mapping[str, str],
    methods: Mapping[str, str],
) -> list[str]:
    problems: list[str] = []

    def problem(message: str) -> None:
        problems.append(f"case {case.id}: {message}")

    if case.outcome not in CATEGORY_OUTCOMES[case.category]:
        problem(f"category {case.category} does not allow outcome {case.outcome}")
    if any(
        not q or q != q.strip() or not q.isprintable() or len(q) > MAX_QUERY_CHARS
        for q in case.queries
    ):
        problem(f"every query must be one trimmed printable line of 1-{MAX_QUERY_CHARS} characters")
    if len(set(case.queries)) != len(case.queries):
        problem("a query is listed twice")
    if case.tenant == PUBLIC and case.matter is not None:
        problem("a public requester has no matter")
    if case.regime == UNKNOWN:
        problem("the asked regime cannot be unknown")
    if (case.outcome == "abstain") != (case.answer is None) or (
        case.answer is not None and not case.answer.strip()
    ):
        problem("'answer' must be null for an abstention and non-empty text otherwise")
    if case.author not in authors or case.method not in methods:
        problem("the label author or method is not declared")
    if not case.note.strip():
        problem("'note' is empty")
    problems.extend(f"case {case.id}: {message}" for message in _check_refs(case, sources))
    return problems


def _check_refs(case: Case, sources: Mapping[str, Source]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    trap_reasons: set[str] = set()
    families: set[str] = set()
    for kind, refs in (("evidence", case.evidence), ("trap", case.traps)):
        for ref in refs:
            source = sources.get(ref.source_id)
            if ref.label in seen or source is None or ref.section_id not in source.sections:
                problems.append(f"{kind} {ref.label[:80]!r} is repeated or not in the fixture")
                continue
            seen.add(ref.label)
            reasons = ineligibility(source, case)
            if kind == "trap":
                trap_reasons.update(reasons)
                if not reasons:
                    problems.append(f"trap {ref.label} is eligible, so it is not a trap")
                continue
            families.add(source.family_id)
            if reasons:
                problems.append(f"evidence {ref.label} is not eligible ({', '.join(reasons)})")
            if source.sections[ref.section_id].count(ref.quote) != 1:
                problems.append(f"the quote from {ref.label} does not occur exactly once there")
    if case.outcome == "abstain" and case.evidence:
        problems.append("an abstention cites no evidence")
    if case.outcome in ("answer", "clarify") and not case.evidence:
        problems.append(f"outcome {case.outcome} needs evidence")
    if case.outcome == "report-conflict" and len(families) < 2:
        problems.append("a conflict needs evidence from at least two source families")
    wanted = CATEGORY_TRAP_REASONS.get(case.category, ())
    if wanted and not trap_reasons.intersection(wanted):
        problems.append(f"category {case.category} needs a trap that fails {' or '.join(wanted)}")
    return problems


def _links(sources: Mapping[str, Source], owner: Mapping[str, str]) -> Callable[[Case], list[str]]:
    def links(case: Case) -> list[str]:
        refs = case.evidence + case.traps
        provisions = {f"{sources[r.source_id].family_id}#{r.section_id}" for r in refs}
        return [f"issue:{case.issue}", *sorted(f"provisions:{owner[p]}" for p in provisions)]

    return links


def _check_splits(
    cases: tuple[Case, ...], groups: tuple[tuple[Case, ...], ...], problems: list[str]
) -> None:
    for group in groups:
        splits = sorted({case.split for case in group})
        if len(splits) > 1:
            ids = ", ".join(case.id for case in group)
            problems.append(
                f"leakage: cases {ids} share an issue or provision group "
                f"but sit in {', '.join(splits)}"
            )
    for split in SPLITS:
        if not any(case.split == split for case in cases):
            problems.append(f"split {split} has no cases")


def _check_manifest(
    manifest: object,
    version: int,
    hashes: tuple[str, str],
    cases: tuple[Case, ...],
    problems: list[str],
) -> None:
    try:
        _exact_keys(manifest, _MANIFEST_KEYS)
        if (
            manifest["schema"] != "rageval-legal-demo-manifest"
            or manifest["dataset"] != "legal-demo"
        ):
            raise ValueError("not a legal-demo manifest")
        if not _is_one(manifest["schema_version"]):
            raise ValueError("'schema_version' must be 1")
        pinned_version = _get(manifest, "dataset_version", int)
        _date(manifest, "frozen_on")
        fixture = _exact_keys(_get(manifest, "fixture", dict), {"sha256", "upstream"})
        _get(fixture, "upstream", dict)
        frozen_cases = _exact_keys(_get(manifest, "cases", dict), {"sha256"})
        pinned = (_get(fixture, "sha256", str), _get(frozen_cases, "sha256", str))
        policy = _get(manifest, "policy", list)
        if not policy or not all(isinstance(line, str) and line.strip() for line in policy):
            raise ValueError("'policy' must list non-empty statements")
    except (TypeError, ValueError) as exc:
        problems.append(f"manifest.json: {exc}")
        return
    if pinned_version != version:
        problems.append(
            f"manifest.json pins dataset_version {pinned_version}, cases.json is {version}"
        )
    for name, pin, actual in zip(("sources.json", "cases.json"), pinned, hashes):
        if pin != actual:
            problems.append(
                f"{name} is not the frozen version (its hash is now {actual}). Changing it "
                "needs a new dataset_version and a new manifest, and results across "
                "versions are not comparable."
            )
    if not _same_counts(manifest["splits"], split_counts(cases)):
        problems.append("manifest.json split counts do not match cases.json")


def _same_counts(pinned: object, actual: dict[str, dict[str, int]]) -> bool:
    """Equal and integer-typed: ``True`` or ``7.0`` does not count as a pinned count."""
    return (
        isinstance(pinned, dict)
        and pinned.keys() == actual.keys()
        and all(
            isinstance(pinned[split], dict)
            and pinned[split].keys() == counts.keys()
            and all(
                type(pinned[split][k]) is int and pinned[split][k] == v for k, v in counts.items()
            )
            for split, counts in actual.items()
        )
    )


def _is_one(value: object) -> bool:
    return type(value) is int and value == 1
