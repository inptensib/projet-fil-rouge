"""Tests for shared/data_loader.py."""

from __future__ import annotations

import torch

from shared.data_loader import (
    build_parent_child_mapping,
    build_parent_child_matrix,
    extract_level_2,
)


class TestExtractLevel2:
    def test_list_input(self):
        assert extract_level_2(["Electronics"]) == "Electronics"

    def test_list_with_spaces(self):
        assert extract_level_2(["  Books  "]) == "Books"

    def test_string_input(self):
        assert extract_level_2("Clothing") == "Clothing"

    def test_empty_list(self):
        assert extract_level_2([]) is None

    def test_none_value(self):
        assert extract_level_2(None) is None

    def test_invalid_values(self):
        assert extract_level_2("none") is None
        assert extract_level_2("nan") is None
        assert extract_level_2("unknown") is None
        assert extract_level_2("") is None

    def test_list_with_invalid_first(self):
        assert extract_level_2(["none", "Valid"]) is None


class TestBuildParentChildMapping:
    def test_simple_mapping(self):
        l1 = [0, 0, 1, 1, 2]
        l2 = [0, 1, 2, 3, 4]
        mapping = build_parent_child_mapping(l1, l2)
        assert mapping == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}

    def test_duplicate_pairs(self):
        l1 = [0, 0, 0, 1, 1]
        l2 = [0, 0, 1, 2, 2]
        mapping = build_parent_child_mapping(l1, l2)
        assert mapping == {0: 0, 1: 0, 2: 1}

    def test_empty(self):
        assert build_parent_child_mapping([], []) == {}


class TestBuildParentChildMatrix:
    def test_simple_matrix(self):
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        matrix = build_parent_child_matrix(mapping, num_l1_classes=2, num_l2_classes=4)
        assert matrix.shape == (2, 4)
        # L1=0 -> L2=0,1
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == 1.0
        assert matrix[0, 2] == 0.0
        # L1=1 -> L2=2,3
        assert matrix[1, 2] == 1.0
        assert matrix[1, 3] == 1.0
        assert matrix[1, 0] == 0.0

    def test_dtype(self):
        mapping = {0: 0}
        matrix = build_parent_child_matrix(mapping, 1, 1)
        assert matrix.dtype == torch.float32

    def test_empty_mapping(self):
        matrix = build_parent_child_matrix({}, 3, 5)
        assert matrix.shape == (3, 5)
        assert matrix.sum() == 0.0
