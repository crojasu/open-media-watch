"""
log_buffer.py — In-memory ring buffer for application logs.

Captures log messages so they can be viewed in the web UI at /admin/logs
while still printing to stdout for deploy logs.
"""
from __future__ import annotations
import threading
from collections import deque
from datetime import datetime, timezone


class LogBuffer:
    """Thread-safe ring buffer that stores the last *maxlen* log entries."""

    def __init__(self, maxlen: int = 1000):
        self._buf: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, message: str, source: str = "app", level: str = "info"):
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "source": source,
            "message": message,
        }
        with self._lock:
            self._buf.append(entry)

    def get(self, limit: int = 200, source: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self._buf)
        if source:
            sources = set(s.strip() for s in source.split(","))
            items = [e for e in items if e["source"] in sources]
        return list(reversed(items[-limit:]))


# ── Global instance ──────────────────────────────────────────────────────────
_buffer = LogBuffer()


def log(message: str, source: str = "app", level: str = "info"):
    """Print to stdout AND store in the ring buffer."""
    print(message, flush=True)
    _buffer.append(message, source=source, level=level)


def get_logs(limit: int = 200, source: str | None = None) -> list[dict]:
    """Retrieve recent log entries (newest first)."""
    return _buffer.get(limit=limit, source=source)
