"""End-to-end test of ``python -m rageval.report``.

The CLI runs in a subprocess against a temporary copy of ``data/``, so the
test exercises the real entry point (argument parsing, directory creation,
every writer) without touching the committed ``results/``. Its outputs must
match the committed ones byte for byte -- the same reproduction check CI
runs on the repository itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIGURES = (
    "reranker_penalty",
    "recall_vs_mrr",
    "per_query_deltas",
    "ext204_replication",
    "calibration",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def run_report(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rageval.report", "--root", str(root)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def report_run(tmp_path_factory):
    """Copy only ``data/`` so the CLI has to create ``results/`` itself."""
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REPO / "data", root / "data")
    return root, run_report(root)


def test_exits_zero_and_reports_what_it_wrote(report_run) -> None:
    _, proc = report_run
    assert proc.returncode == 0, proc.stderr
    assert "wrote results/results.json" in proc.stdout
    assert "wrote results/TABLES.md" in proc.stdout


@pytest.mark.parametrize("name", ["results.json", "TABLES.md"])
def test_numeric_outputs_match_the_committed_ones(report_run, name: str) -> None:
    root, _ = report_run
    assert (root / "results" / name).read_bytes() == (REPO / "results" / name).read_bytes()


@pytest.mark.parametrize("figure", FIGURES)
def test_every_figure_is_written_as_a_png(report_run, figure: str) -> None:
    root, _ = report_run
    assert (root / "results" / "figures" / f"{figure}.png").read_bytes()[:8] == PNG_SIGNATURE


def test_missing_run_files_fail_with_a_clear_error(tmp_path) -> None:
    proc = run_report(tmp_path)
    assert proc.returncode != 0
    assert "rageval.report: error: expected run file is missing" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not (tmp_path / "results").exists()
