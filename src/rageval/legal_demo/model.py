"""The legal-demo data model and the rules its labels are checked against.

Three rules live here, so later retrieval and evaluation code can reuse them:

- ``ineligibility`` -- one fail-closed eligibility rule. A source is eligible
  for a case only if the requester may see it (public, or the same tenant and,
  for a matter-scoped source, the same matter), its jurisdiction and regime are
  known and match the question, and it is known to be in force on the as-of
  date. Effective intervals are half-open, so a version is in force on its
  ``effective_from`` and not on its ``effective_to``. ``"unknown"`` never
  matches.
- ``leakage_groups`` -- connected components of cases that share a link key.
- ``closest_cross_split`` -- the most similar pair of queries in different
  splits, by content-word Jaccard: a coarse screen for near-verbatim copies.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

DATA_DIR = Path("data") / "legal-demo"
SPLITS = ("train", "dev", "test")
CATEGORIES = (
    "answerable",
    "unanswerable",
    "unknown-metadata",
    "wrong-regime",
    "outdated",
    "conflicting-source",
)
OUTCOMES = ("answer", "clarify", "report-conflict", "abstain")
UNKNOWN = "unknown"
PUBLIC = "public"

# The expected outcomes each category admits.
CATEGORY_OUTCOMES = {
    "answerable": ("answer", "clarify"),
    "unanswerable": ("abstain",),
    "unknown-metadata": ("abstain",),
    "wrong-regime": ("answer", "abstain"),
    "outdated": ("answer", "abstain"),
    "conflicting-source": ("report-conflict",),
}
# A category about a trap needs at least one trap failing one of these rules.
# unknown-metadata: a source's text would answer, but a fact eligibility needs
# about it is unknown, so the rule fails closed and the case expects abstention.
CATEGORY_TRAP_REASONS = {
    "unknown-metadata": (
        "unknown-scope",
        "unknown-jurisdiction",
        "unknown-regime",
        "unknown-in-force",
    ),
    "wrong-regime": ("other-jurisdiction", "other-regime"),
    "outdated": ("not-yet-in-force", "no-longer-in-force"),
}
# Two queries in different splits must share less than this share of their
# content words. A coarse screen for near-verbatim copies; paraphrases are kept
# together by construction, since they are the queries of one case.
MAX_CROSS_SPLIT_JACCARD = 0.5
_STOPWORD_TEXT = (
    "a an and are at be by can could did do does for from had has have how i in is "
    "it its may me must my need of on or should that the their this to under was "
    "were what when which who why will with"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())
MAX_QUERY_CHARS = 300


class DatasetError(ValueError):
    """The legal-demo files are malformed, inconsistent, or not the frozen version."""


@dataclass(frozen=True)
class Source:
    """The fixture fields eligibility needs, plus each section's text."""

    source_id: str
    family_id: str
    tenant: str
    matter: str | None
    jurisdiction: str
    regime: str
    effective_from: date | str  # a str is always UNKNOWN
    effective_to: date | str | None  # None: no end date
    sections: Mapping[str, str]


@dataclass(frozen=True)
class Ref:
    """A section of one source version. Evidence carries a quote; a trap does not."""

    source_id: str
    section_id: str
    quote: str | None = None

    @property
    def label(self) -> str:
        return f"{self.source_id}#{self.section_id}"


@dataclass(frozen=True)
class Case:
    """One labelled question, asked by one requester on one date."""

    id: str
    split: str
    issue: str
    category: str
    queries: tuple[str, ...]
    tenant: str
    matter: str | None
    jurisdiction: str
    regime: str
    as_of: date
    outcome: str
    answer: str | None
    evidence: tuple[Ref, ...]
    traps: tuple[Ref, ...]
    author: str
    method: str
    note: str

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(f"{self.id}.q{n}" for n in range(1, len(self.queries) + 1))


@dataclass(frozen=True)
class Dataset:
    """A loaded, checked and frozen legal-demo set."""

    version: int
    sources: Mapping[str, Source]
    cases: tuple[Case, ...]
    provision_group_of: Mapping[str, str]  # "family#section" -> provision group
    groups: tuple[tuple[Case, ...], ...]  # leakage groups, each inside one split
    fixture_sha256: str
    cases_sha256: str
    closest_cross_split: tuple[float, tuple[str, str]]  # Jaccard, query ids

    def provisions(self, case: Case) -> tuple[str, ...]:
        """Sorted ``family#section`` keys of every evidence and trap ref in ``case``."""
        refs = case.evidence + case.traps
        return tuple(
            sorted({f"{self.sources[r.source_id].family_id}#{r.section_id}" for r in refs})
        )


def canonical_sha256(value: object) -> str:
    """SHA-256 of ``value`` as sorted, compact, UTF-8 JSON, so layout changes keep the hash."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def ineligibility(source: Source, case: Case) -> tuple[str, ...]:
    """Every eligibility rule ``source`` fails for ``case``; empty means eligible.

    Fails closed without relying on the loader's checks. An empty or
    ``"unknown"`` tenant or matter on either side is ``unknown-scope``. A
    source with a matter suits only that exact matter, even when its tenant
    reads ``public``.

    Raises:
        TypeError: A date is not a ``datetime.date`` (a ``datetime`` counts as wrong).
    """
    _require_dates(source, case)
    reasons = []
    scope = (source.tenant, source.matter, case.tenant, case.matter)
    if not source.tenant or not case.tenant or UNKNOWN in scope or "" in scope:
        reasons.append("unknown-scope")
    elif source.tenant != PUBLIC and source.tenant != case.tenant:
        reasons.append("other-tenant")
    elif source.matter is not None and source.matter != case.matter:
        reasons.append("other-matter")
    for field, asked in (("jurisdiction", case.jurisdiction), ("regime", case.regime)):
        value = getattr(source, field)
        if value == UNKNOWN:
            reasons.append(f"unknown-{field}")
        elif value != asked:
            reasons.append(f"other-{field}")
    start, end = source.effective_from, source.effective_to
    if start == UNKNOWN:
        reasons.append("unknown-in-force")
    elif start > case.as_of:
        reasons.append("not-yet-in-force")
    elif end == UNKNOWN:
        reasons.append("unknown-in-force")
    elif end is not None and end <= case.as_of:
        reasons.append("no-longer-in-force")
    return tuple(reasons)


def _require_dates(source: Source, case: Case) -> None:
    def is_day(value: object) -> bool:
        return isinstance(value, date) and not isinstance(value, datetime)

    start, end = source.effective_from, source.effective_to
    if not (
        is_day(case.as_of)
        and (is_day(start) or start == UNKNOWN)
        and (end is None or end == UNKNOWN or is_day(end))
    ):
        raise TypeError("eligibility dates must be datetime.date values or 'unknown'")


def leakage_groups(
    cases: Iterable[Case], links: Callable[[Case], Iterable[str]]
) -> tuple[tuple[Case, ...], ...]:
    """Connected components of ``cases`` joined by any shared link key, sorted by first id."""
    cases = tuple(cases)
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        while parent.setdefault(key, key) != key:
            key = parent[key]
        return key

    for case in cases:
        for link in links(case):
            parent[find(f"case:{case.id}")] = find(f"link:{link}")
    members: dict[str, list[Case]] = {}
    for case in cases:
        members.setdefault(find(f"case:{case.id}"), []).append(case)
    groups = (tuple(sorted(group, key=lambda c: c.id)) for group in members.values())
    return tuple(sorted(groups, key=lambda group: group[0].id))


def content_words(text: str) -> frozenset[str]:
    """Lowercase alphanumeric words of ``text`` minus STOPWORDS."""
    return frozenset(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def closest_cross_split(cases: Iterable[Case]) -> tuple[float, tuple[str, str]]:
    """The highest content-word Jaccard between queries in different splits, and their ids."""
    items = [
        (case.split, query_id, content_words(query))
        for case in cases
        for query_id, query in zip(case.query_ids, case.queries)
    ]
    best: tuple[float, tuple[str, str]] = (0.0, ("", ""))
    for (split_a, id_a, words_a), (split_b, id_b, words_b) in combinations(items, 2):
        union = words_a | words_b
        score = len(words_a & words_b) / len(union) if union else 1.0
        if split_a != split_b and score > best[0]:
            best = (score, (id_a, id_b))
    return best


def split_counts(cases: Iterable[Case]) -> dict[str, dict[str, int]]:
    """Cases and queries per split, in SPLITS order."""
    cases = tuple(cases)
    return {
        split: {
            "cases": sum(1 for case in cases if case.split == split),
            "queries": sum(len(case.queries) for case in cases if case.split == split),
        }
        for split in SPLITS
    }
