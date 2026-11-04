"""Project location shared across library modules, independent of current directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
