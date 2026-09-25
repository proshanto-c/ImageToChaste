"""
Unit tests for CLI interface commands and argument handling.
"""

import json
import subprocess
import sys
from pathlib import Path

from imagetochaste import __version__


def test_cli_version():
    res = subprocess.run(
        [sys.executable, "-m", "imagetochaste.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert f"imagetochaste {__version__}" in res.stdout


def test_cli_info():
    res = subprocess.run(
        [sys.executable, "-m", "imagetochaste.cli", "info"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert f"ImageToChaste Version: {__version__}" in res.stdout
    assert "Python Version:" in res.stdout


def test_cli_export(tmp_path: Path):
    # Create test centroid JSON
    json_path = tmp_path / "test_cells.json"
    cells = [{"cell_id": 0, "x": 10.0, "y": 20.0}, {"cell_id": 1, "x": 30.0, "y": 40.0}]
    json_path.write_text(json.dumps(cells))

    out_nodes = tmp_path / "exported.nodes"

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "imagetochaste.cli",
            "export",
            "--centroids",
            str(json_path),
            "--output",
            str(out_nodes),
            "--style",
            "simple",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert out_nodes.exists()
    content = out_nodes.read_text()
    assert content.startswith("2\n0 10.000000 20.000000 0\n")
