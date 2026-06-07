"""
Persistent history stored as a JSON array in history.json next to this file.
Thread-safe via a module-level lock (single-process FastAPI).
"""

import json
import threading
from pathlib import Path
from typing import List, Dict

HISTORY_FILE = Path(__file__).parent / "history.json"
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_run(entry: Dict) -> None:
    """Append one run record to the history file."""
    with _lock:
        runs = _load()
        runs.append(entry)
        _persist(runs)


def get_all_runs() -> List[Dict]:
    """Return all stored run records (oldest first)."""
    return _load()


def clear_history() -> None:
    """Delete all history records."""
    with _lock:
        _persist([])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> List[Dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _persist(data: List[Dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
