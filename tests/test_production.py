"""Tests for 03_production/src/ modules."""

from __future__ import annotations

import sys
from unittest import mock

import pytest
from helpers import import_from_production


class TestDDPUtils:
    def test_print_rank0_rank0(self, capsys):
        ddp_utils = import_from_production("ddp_utils")
        ddp_utils.print_rank0("hello", rank=0)
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_print_rank0_other_rank(self, capsys):
        ddp_utils = import_from_production("ddp_utils")
        ddp_utils.print_rank0("hello", rank=1)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_cleanup_ddp_when_not_initialized(self):
        ddp_utils = import_from_production("ddp_utils")
        with mock.patch("torch.distributed.is_initialized", return_value=False), \
             mock.patch("torch.distributed.destroy_process_group") as mock_destroy:
            ddp_utils.cleanup_ddp()
            mock_destroy.assert_not_called()
