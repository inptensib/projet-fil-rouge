"""Tests for rakuten/common/utils.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from rakuten.common.utils import count_parameters, get_device, set_seed


class TestSetSeed:
    def test_deterministic_random(self):
        set_seed(42)
        import random
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_deterministic_torch(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)


class TestGetDevice:
    def test_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda")


class TestCountParameters:
    def test_simple_model(self):
        model = nn.Linear(10, 5)
        counts = count_parameters(model)
        assert counts["total"] == 10 * 5 + 5  # weight + bias = 55
        assert counts["trainable"] == 55

    def test_frozen_model(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        counts = count_parameters(model)
        assert counts["total"] == 55
        assert counts["trainable"] == 0

    def test_mixed_model(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
        for p in model[0].parameters():
            p.requires_grad = False
        counts = count_parameters(model)
        assert counts["trainable"] == 5 * 3 + 3  # only second layer
        assert counts["total"] == 55 + 18
