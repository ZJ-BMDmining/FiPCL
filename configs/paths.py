"""Central path resolution for the FiPCL release.

Everything is derived from the repository root (the directory that contains
``FiPCL/``, ``results/`` and ``scripts/``), so the repository can be cloned
anywhere.  Two absolute locations cannot be shipped with the code and are
resolved through environment variables:

===============================  =========================================================
Variable                         Meaning
===============================  =========================================================
``FIPCL_PROJECT_ROOT``           Override the repository root (defaults to the parent of
                                 this file).
``FIPCL_MRI_BASE_PATH``          Root folder of the raw NIfTI files, laid out as
                                 ``<MRI_BASE_PATH>/<dataset>/`` (e.g. ``All_MRI/ADNI``).
``FIPCL_MRI_CACHE``              Cache folder for preprocessed MRI .npy volumes
                                 (defaults to ``<root>/.mri_cache``).
``FIPCL_PYTHON``                 Python interpreter used by the multi-process orchestrators.
``FIPCL_DATA_ROOT``              Folder holding the raw csv tables (defaults to ``<root>/data``).
``FIPCL_SPLIT_ROOT``             Folder holding the generated 5-fold splits
                                 (defaults to ``<root>/split_data_5fold``).
===============================  =========================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ``configs/`` lives directly under the repository root.
REPO_ROOT = Path(
    os.environ.get("FIPCL_PROJECT_ROOT", Path(__file__).resolve().parent.parent)
).expanduser().resolve()

MRI_BASE_PATH = Path(
    os.environ.get("FIPCL_MRI_BASE_PATH", REPO_ROOT / "All_MRI")
).expanduser()

MRI_CACHE_DIR = Path(
    os.environ.get("FIPCL_MRI_CACHE", REPO_ROOT / ".mri_cache")
).expanduser()


def _guess_python() -> str:
    """Return an interpreter used by the multi-process runners.

    Preference order: ``FIPCL_PYTHON`` → the interpreter running this script →
    ``python`` on ``PATH``.
    """
    explicit = os.environ.get("FIPCL_PYTHON")
    if explicit:
        return explicit
    if sys.executable:
        return sys.executable
    return "python"


PYTHON = _guess_python()

DATA_ROOT = Path(
    os.environ.get("FIPCL_DATA_ROOT", REPO_ROOT / "data")
).expanduser()

SPLIT_ROOT = Path(
    os.environ.get("FIPCL_SPLIT_ROOT", REPO_ROOT / "split_data_5fold")
).expanduser()

# Shorthands used all over the codebase.
FIPCL_DIR = REPO_ROOT / "FiPCL"
SCRIPT_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"
CHECKPOINT_DIR = FIPCL_DIR / "checkpoints"
FIPCL_RESULT_DIR = RESULTS_DIR / "fipcl_5fold"


def ensure_repo_on_path() -> Path:
    """Insert the repository root on ``sys.path`` (idempotent)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


__all__ = [
    "REPO_ROOT",
    "MRI_BASE_PATH",
    "MRI_CACHE_DIR",
    "PYTHON",
    "DATA_ROOT",
    "SPLIT_ROOT",
    "FIPCL_DIR",
    "SCRIPT_DIR",
    "RESULTS_DIR",
    "LOG_DIR",
    "CHECKPOINT_DIR",
    "FIPCL_RESULT_DIR",
    "ensure_repo_on_path",
]
