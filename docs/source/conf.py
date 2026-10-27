"""Sphinx configuration for MILES-ClimateSleuth documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "MILES-ClimateSleuth"
author = "NCAR Machine Integration and Learning for Earth Systems (MILES)"
copyright = "2026, NCAR MILES"

version_file = ROOT / "climatesleuth" / "VERSION"
release = version_file.read_text().strip() if version_file.exists() else "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if (Path(__file__).parent / "_static").exists() else []
