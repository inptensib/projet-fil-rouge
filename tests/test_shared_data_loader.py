"""Tests for rakuten/common/data_loader.py."""

from __future__ import annotations

import json
import os

import torch

from rakuten.common.data_loader import (
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
            "train_is_synthetic", "real_class_counts_l1", "real_class_counts_l2",
            "synthetic_policy",
        }
        assert set(result.keys()) == expected_keys
        assert result["synthetic_policy"] == "keep_all"

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


def _expected_text(designation):
    return f"{designation}. Description de {designation}"


class TestSyntheticPolicy:
    """Tests for synthetic-row tracking and synthetic_policy (dataset v2)."""

    REAL_TRAIN = ["Tel A", "Tel B", "Roman A"]
    SYNTH_TRAIN = {
        "SYNTH-OVS-001": "Tel OVS",
        "SYNTH-OVL-001": "Tel OVL",
        "SYNTH-ORPH-001": "Roman ORPH",
        "SYNTH-GEN-001": "Tel GEN",
    }

    def _create_split_files(self, tmp_path):
        """Every L2 class is in every split: _ensure_l2_coverage adds nothing."""
        train = [
            _make_record("1", "Tel A", "Électronique", "Téléphone"),
            _make_record("2", "Tel B", "Électronique", "Téléphone"),
            _make_record("SYNTH-OVS-001", "Tel OVS", "Électronique", "Téléphone"),
            _make_record("SYNTH-OVL-001", "Tel OVL", "Électronique", "Téléphone"),
            _make_record("SYNTH-ORPH-001", "Roman ORPH", "Livres", "Roman"),
            _make_record("SYNTH-GEN-001", "Tel GEN", "Électronique", "Téléphone"),
            _make_record("3", "Roman A", "Livres", "Roman"),
        ]
        val = [
            _make_record("4", "Tel V", "Électronique", "Téléphone"),
            _make_record("SYNTH-OVS-100", "Roman OVS val", "Livres", "Roman"),
        ]
        test = [
            _make_record("5", "Tel T", "Électronique", "Téléphone"),
            _make_record("SYNTH-OVL-100", "Roman OVL test", "Livres", "Roman"),
        ]
        train_path = str(tmp_path / "train_split.jsonl")
        val_path = str(tmp_path / "val_split.jsonl")
        test_path = str(tmp_path / "test_split.jsonl")
        _write_jsonl(train_path, train)
        _write_jsonl(val_path, val)
        _write_jsonl(test_path, test)
        return train_path, val_path, test_path

    def test_keep_all_is_default_and_unchanged(self, tmp_path):
        """keep_all reproduces the previous behavior exactly: same X/y, in order."""
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        explicit = prepare_hierarchical_data_from_splits(
            train_p, val_p, test_p, synthetic_policy="keep_all"
        )
        expected_train = [
            _expected_text(d)
            for d in ["Tel A", "Tel B", "Tel OVS", "Tel OVL", "Roman ORPH", "Tel GEN", "Roman A"]
        ]
        assert result["X_train"] == expected_train
        assert result["X_train"] == explicit["X_train"]
        assert list(result["y_l1_train"]) == list(explicit["y_l1_train"])
        assert list(result["y_l2_train"]) == list(explicit["y_l2_train"])
        assert result["X_val"] == explicit["X_val"]
        assert result["X_test"] == explicit["X_test"]

    def test_drop_oversampled_excludes_ovs_ovl_keeps_orph_gen(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(
            train_p, val_p, test_p, synthetic_policy="drop_oversampled"
        )
        assert _expected_text("Tel OVS") not in result["X_train"]
        assert _expected_text("Tel OVL") not in result["X_train"]
        assert _expected_text("Roman ORPH") in result["X_train"]
        assert _expected_text("Tel GEN") in result["X_train"]
        assert len(result["X_train"]) == 5
        assert result["synthetic_policy"] == "drop_oversampled"

    def test_val_test_never_filtered(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(
            train_p, val_p, test_p, synthetic_policy="drop_oversampled"
        )
        assert _expected_text("Roman OVS val") in result["X_val"]
        assert _expected_text("Roman OVL test") in result["X_test"]
        assert len(result["X_val"]) == 2
        assert len(result["X_test"]) == 2

    def test_train_is_synthetic_aligned(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        for policy in ("keep_all", "drop_oversampled"):
            result = prepare_hierarchical_data_from_splits(
                train_p, val_p, test_p, synthetic_policy=policy
            )
            flags = result["train_is_synthetic"]
            assert len(flags) == len(result["X_train"])
            synth_texts = {_expected_text(d) for d in self.SYNTH_TRAIN.values()}
            for text, flag in zip(result["X_train"], flags):
                assert isinstance(flag, bool)
                assert flag == (text in synth_texts)

    def test_real_class_counts_ignore_synthetic(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        tel_idx = int(result["l2_encoder"].transform(["Téléphone"])[0])
        roman_idx = int(result["l2_encoder"].transform(["Roman"])[0])
        # Train keep_all: Téléphone = 2 real + 3 SYNTH, Roman = 1 real + 1 SYNTH
        assert result["real_class_counts_l2"][tel_idx] == 2
        assert result["real_class_counts_l2"][roman_idx] == 1
        elec_idx = int(result["l1_encoder"].transform(["Électronique"])[0])
        livres_idx = int(result["l1_encoder"].transform(["Livres"])[0])
        assert result["real_class_counts_l1"][elec_idx] == 2
        assert result["real_class_counts_l1"][livres_idx] == 1
        assert sum(result["real_class_counts_l2"].values()) == len(self.REAL_TRAIN)

    def test_real_class_counts_invariant_under_policy(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        kept = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        dropped = prepare_hierarchical_data_from_splits(
            train_p, val_p, test_p, synthetic_policy="drop_oversampled"
        )
        assert kept["real_class_counts_l1"] == dropped["real_class_counts_l1"]
        assert kept["real_class_counts_l2"] == dropped["real_class_counts_l2"]

    def test_drop_oversampled_pandas_branch(self, tmp_path):
        """n_samples routes through load_random_sample (pandas DataFrames)."""
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(
            train_p, val_p, test_p, n_samples=100, synthetic_policy="drop_oversampled"
        )
        assert _expected_text("Tel OVS") not in result["X_train"]
        assert _expected_text("Tel OVL") not in result["X_train"]
        assert _expected_text("Roman ORPH") in result["X_train"]
        assert len(result["train_is_synthetic"]) == len(result["X_train"])
        tel_idx = int(result["l2_encoder"].transform(["Téléphone"])[0])
        assert result["real_class_counts_l2"][tel_idx] == 2

    def test_invalid_policy_raises(self, tmp_path):
        import pytest
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        with pytest.raises(ValueError, match="synthetic_policy"):
            prepare_hierarchical_data_from_splits(
                train_p, val_p, test_p, synthetic_policy="drop_all"
            )

    def test_coverage_duplicates_keep_origin_flag(self, tmp_path):
        """A coverage copy of a real row stays real; of a synthetic row stays synthetic."""
        train = [
            _make_record("1", "Tel A", "Électronique", "Téléphone"),
            _make_record("SYNTH-GEN-001", "Poème GEN", "Livres", "Poésie"),
        ]
        val = [
            _make_record("2", "Tel V", "Électronique", "Téléphone"),
            _make_record("3", "Casque V", "Électronique", "Casque"),
        ]
        test = [_make_record("4", "Tel T", "Électronique", "Téléphone")]
        train_p = str(tmp_path / "train_split.jsonl")
        val_p = str(tmp_path / "val_split.jsonl")
        test_p = str(tmp_path / "test_split.jsonl")
        _write_jsonl(train_p, train)
        _write_jsonl(val_p, val)
        _write_jsonl(test_p, test)

        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        flags = dict(zip(result["X_train"], result["train_is_synthetic"]))
        # "Casque V" copied from val into train: real
        assert flags[_expected_text("Casque V")] is False
        # "Poème GEN" stays synthetic in train
        assert flags[_expected_text("Poème GEN")] is True
        # The synthetic-only class still counts 0 real samples
        poesie_idx = int(result["l2_encoder"].transform(["Poésie"])[0])
        assert result["real_class_counts_l2"][poesie_idx] == 0


class TestNormalize:
    """Tests for the normalize parameter (applied to the 3 splits)."""

    RAW = "Caf&eacute; <b>Premium</b>"
    CLEAN = "Café Premium"

    def _create_split_files(self, tmp_path):
        train = [
            _make_record("1", self.RAW, "Épicerie", "Café"),
            _make_record("2", self.RAW, "Épicerie", "Café"),
        ]
        val = [_make_record("3", self.RAW, "Épicerie", "Café")]
        test = [_make_record("4", self.RAW, "Épicerie", "Café")]
        train_p = str(tmp_path / "train_split.jsonl")
        val_p = str(tmp_path / "val_split.jsonl")
        test_p = str(tmp_path / "test_split.jsonl")
        _write_jsonl(train_p, train)
        _write_jsonl(val_p, val)
        _write_jsonl(test_p, test)
        return train_p, val_p, test_p

    def test_normalize_applied_to_all_splits(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(
            train_p, val_p, test_p, normalize=True
        )
        for key in ("X_train", "X_val", "X_test"):
            for text in result[key]:
                assert self.CLEAN in text
                assert "&eacute;" not in text
                assert "<b>" not in text

    def test_normalize_off_by_default(self, tmp_path):
        train_p, val_p, test_p = self._create_split_files(tmp_path)
        result = prepare_hierarchical_data_from_splits(train_p, val_p, test_p)
        for key in ("X_train", "X_val", "X_test"):
            assert all("&eacute;" in text for text in result[key])

    def test_single_file_path_normalize(self, tmp_path):
        from rakuten.common.data_loader import prepare_hierarchical_data

        records = [
            _make_record(str(i), f"{self.RAW} {i}", "Épicerie", "Café" if i % 2 else "Thé")
            for i in range(12)
        ]
        path = str(tmp_path / "products.jsonl")
        _write_jsonl(path, records)
        result = prepare_hierarchical_data(path, n_samples=100, normalize=True)
        for key in ("X_train", "X_val", "X_test"):
            for text in result[key]:
                assert "&eacute;" not in text
                assert "<b>" not in text
