"""HLPD — Hierarchical Label Pre-populated Decoder.

Reference: Torba, Laclau et al. ECIR 2025.
"""

from .attention_bias import build_M_hiera
from .losses import ZLPRLoss
from .models import HLPD
from .trainer import HLPDConfig, HLPDTrainer

__all__ = [
    "HLPD",
    "HLPDConfig",
    "HLPDTrainer",
    "ZLPRLoss",
    "build_M_hiera",
]
