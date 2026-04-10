"""Tests for shared/data_loader.py."""

from __future__ import annotations

import json
import os

import torch

from shared.data_loader import (
    build_parent_child_mapping,
    build_parent_child_matrix,
    extract_level_2,
    prepare_hierarchical_data_from_splits,
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


def _write_jsonl(path, records):
    """Helper to write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_record(pid, designation, typelabel, producttype):
    return {
        "productid": pid,
        "designation": designation,
        "description": f"Description de {designation}",
        "prdtypecode": 1,
        "typelabel": typelabel,
        "producttype": [producttype],
    }


class TestPrepareFromSplits:
    """Tests for prepare_hierarchical_data_from_splits."""

    def _create_split_files(self, tmp_path):
        train = [
            _make_record("1", "Produit A", "Électronique", "Téléphone"),
            _make_record("2", "Produit B", "Électronique", "Tablette"),
            _make_record("3", "Produit C", "Livres", "Roman"),
            _make_record("4", "Produit D", "Livres", "BD"),
            _make_record("5", "Produit E", "Électronique", "Téléphone"),
            _make_record("6", "Produit F", "Livres", "Roman"),
            _make_record("7", "Produit G", "Électronique", "Tablette"),
        ]
        val = [
            _make_record("8", "Produit H", "Électronique", "Casque"),
            _make_record("9", "Produit I", "Livres", "Roman"),
        ]
        test = [
            _make_record("10", "Produit J", "Électronique", "Téléphone"),
            _make_record("11", "Produit K", "Mode", "Robe"),
        ]

        train_path = str(tmp_path / "train_split.jsonl")
        val_path = str(tmp_path / "val_split.jsonl")
        test_path = str(tmp_path / "test_split.jsonl")

        _write_jsonl(train_path, train)
        _write_jsonl(val_path, val)
        _write_jsonl(test_path, test)

        return train_path, val_path, test_path

    def test_returns_same_keys(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        expected_keys = {
            "X_train", "X_val", "X_test",
            "y_l1_train", "y_l1_val", "y_l1_test",
            "y_l2_train", "y_l2_val", "y_l2_test",
            "l1_encoder", "l2_encoder", "l2_to_l1_mapping",
            "num_classes_l1", "num_classes_l2",
        }
        assert set(result.keys()) == expected_keys

    def test_split_sizes(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        # _ensure_l2_coverage peut ajouter des échantillons dupliqués
        assert len(result["X_train"]) >= 7
        assert len(result["X_val"]) >= 2
        assert len(result["X_test"]) >= 2

    def test_encoders_union(self, tmp_path):
        """Labels from val/test that don't appear in train must be in encoders."""
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        l1_classes = set(result["l1_encoder"].classes_)
        l2_classes = set(result["l2_encoder"].classes_)
        # "Mode" only in test, "Casque" only in val, "Robe" only in test
        assert "Mode" in l1_classes
        assert "Casque" in l2_classes
        assert "Robe" in l2_classes

    def test_label_consistency(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        # All encoded labels must be valid indices
        assert result["y_l1_train"].max() < result["num_classes_l1"]
        assert result["y_l2_train"].max() < result["num_classes_l2"]
        assert result["y_l1_test"].max() < result["num_classes_l1"]
        assert result["y_l2_test"].max() < result["num_classes_l2"]

    def test_missing_file_raises(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            prepare_hierarchical_data_from_splits(
                str(tmp_path / "nonexistent.jsonl"),
                str(tmp_path / "nonexistent2.jsonl"),
                str(tmp_path / "nonexistent3.jsonl"),
            )
