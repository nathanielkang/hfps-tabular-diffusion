"""
scale/_common.py - shared path wiring + small utilities for the scale package.
"""
from __future__ import annotations
import os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "diffusion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROJECT_ROOT = _ROOT


def peak_rss_gb() -> float | None:
    """Best-effort peak resident memory in GB (Linux/mac via resource)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS reports bytes
        if sys.platform == "darwin":
            return kb / (1024 ** 3)
        return kb / (1024 ** 2)
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 ** 3)
        except Exception:
            return None


class Stopwatch:
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0