"""HLPD — Hierarchical Label Prediction Decoder (HLPD).

Reference: Torba, Laclau et al. ECIR 2025.
"""

from .attention_bias import build_M_hiera
from .losses import ZLPRLoss
from .models import HLPD, HLPDDecoderLayer
from .trainer import HLPDConfig, HLPDTrainer

__all__ = [
    "HLPD",
    "HLPDConfig",
    "HLPDDecoderLayer",
    "HLPDTrainer",
    "ZLPRLoss",
    "build_M_hiera",
]
