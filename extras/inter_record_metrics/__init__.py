"""
Inter-record distance metrics for evaluating synthetic tabular data.

Available metrics:
    snn_loss   — Similarity of Nearest Neighbors (fidelity)
    dcr        — Distance to Closest Record (privacy / fidelity)
"""

from .snn_loss import snn_loss
from .dcr import dcr

__all__ = ["snn_loss", "dcr"]
