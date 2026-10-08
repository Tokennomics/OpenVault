#!/usr/bin/env python3
"""Thin shim so `python baselines/baseline_agent.py ...` works from the repo root.

Delegates to the ``baseline_agent`` package's ``analyze`` CLI. See
``baseline_agent/cli.py`` for the implementation.
"""
from __future__ import annotations

import os
import sys

# Make the sibling package importable when invoked as a loose script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
