"""Tests for shared/text_normalization.py.

Mojibake and HTML examples are verbatim samples from
dataset_v2/datasets_finaux/train_split.jsonl.
"""

from __future__ import annotations

import pytest

from shared.text_normalization import normalize_text

REAL_WORLD_INPUTS = [
    "Connecteur Sma - Sma Femme - RÃ©trÃ©cissement - DorÃ© - Par 1 Morceau (S)",
    "recouvert de peluchepoteaux habillÃ©s de jutecouleur: gris clair",
    "Plaque Vitrée Pour Cube 10 L Plaque vitrã©e pour cube 10 L",
    "Dresseur De Rã©Pulsif Pour Chien Portable",
    "que vous soyez Ã la maison ou non, il ajoutera automatiquement de l'eau Ã la plaque",
    "fait de matiÃ¿re plastique, qui est sÃ»re et fiable, pas facile Ã porter",
    "en contrÃ¿lant la température",
    "Aquarium ã\u0080\u0090Double faceã\u0080\u0091 brosse",
    "Conseils dâ€utilisation et dâ€™entretien",
    "la poussiÃ¿re, lâ\u0080\u0099aspirateur",
    "Ã\u0089cran Lcd Argent",
    "Température de 25Â°C Â« ProtÃ©gÃ© Â»",
    "Mangeoire Pour Poussins 1er Âge 1,5 Kg Avec Pattes, 19x20cm, Rouge",
    "Abreuvoir &agrave; Mamelon De Poulet Suspendu, Grande Capacit&eacute;",
    "M&amp;eacute;tal &amp;agrave; base chauffante",
    "DERMIDINE Spray <br /> - des plaies <br /> - des cals<br />",
    "Caractéristiques: </ p> 1. Chaque kit",
    "PEIGNE À AIGUILLES AUTONETTOYANTES : APRÈS AVOIR BROSSÉ LES POILS",
    "ABREUVOIR POUR POULES 8 L. VÉRIFIEZ L'ARTICLE AU MOMENT DE LA RÉCEPTION !",
    "Sifflet ï¿½ ultrasons rÃ©glable",
    "d�j� cass�",
    "Mangeoire pour poussins du premier âge. Fabriqué en Espagne.",
]


class TestNoneAndNonString:
    def test_none(self):
        assert normalize_text(None) == ""

    def test_non_string(self):
        assert normalize_text(123) == ""
        assert normalize_text(["a"]) == ""

    def test_empty(self):
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        assert normalize_text("   \t\n  ") == ""


class TestHtmlEntities:
    def test_named_entities(self):
        raw = "Abreuvoir &agrave; Mamelon De Poulet Suspendu, Grande Capacit&eacute;"
        assert normalize_text(raw) == (
            "Abreuvoir à Mamelon De Poulet Suspendu, Grande Capacité"
        )

    def test_numeric_entities(self):
        assert normalize_text("L&#39;abreuvoir d&#233;givr&#233;") == "L'abreuvoir dégivré"

    def test_double_encoded_entities(self):
        assert normalize_text("M&amp;eacute;tal &amp;agrave; base chauffante") == (
            "Métal à base chauffante"
        )

    def test_literal_ampersand_kept(self):
        assert normalize_text("Black & Decker") == "Black & Decker"


class TestHtmlTags:
    def test_br_tags_become_spaces(self):
        raw = "DERMIDINE Spray <br /> - des plaies <br /> - des cals<br />"
        assert normalize_text(raw) == "DERMIDINE Spray - des plaies - des cals"

    def test_nested_tags(self):
        raw = "<p><strong>Sans danger pour votre chien</strong>, ce collier</p>"
        assert normalize_text(raw) == "Sans danger pour votre chien , ce collier"

    def test_malformed_closing_tag(self):
        assert normalize_text("Caractéristiques: </ p> 1. Chaque kit") == (
            "Caractéristiques: 1. Chaque kit"
        )

    def test_entity_encoded_tags_stripped(self):
        assert normalize_text("&lt;b&gt;Texte en gras&lt;/b&gt; normal") == (
            "Texte en gras normal"
        )


class TestMojibakeRoundTrip:
    def test_pure_mojibake_designation(self):
        raw = "Connecteur Sma - Sma Femme - RÃ©trÃ©cissement - DorÃ© - Par 1 Morceau (S)"
        assert normalize_text(raw) == (
            "Connecteur Sma - Sma Femme - Rétrécissement - Doré - Par 1 Morceau (S)"
        )

    def test_pure_mojibake_description(self):
        raw = "recouvert de peluchepoteaux habillÃ©s de jutecouleur: gris clair"
        assert normalize_text(raw) == (
            "recouvert de peluchepoteaux habillés de jutecouleur: gris clair"
        )

    def test_replacement_char_left_as_is(self):
        assert normalize_text("d�j� cass�") == "d�j� cass�"
        assert normalize_text("Sifflet ï¿½ ultrasons rÃ©glable") == (
            "Sifflet � ultrasons réglable"
        )


class TestMojibakeTable:
    def test_mixed_clean_and_mojibake(self):
        raw = "Plaque Vitrée Pour Cube 10 L Plaque vitrã©e pour cube 10 L"
        assert normalize_text(raw) == (
            "Plaque Vitrée Pour Cube 10 L Plaque vitrée pour cube 10 L"
        )

    def test_title_cased_mojibake(self):
        assert normalize_text("Dresseur De Rã©Pulsif Pour Chien Portable") == (
            "Dresseur De RéPulsif Pour Chien Portable"
        )

    def test_a_grave_with_plain_space(self):
        raw = "que vous soyez Ã la maison ou non, il ajoutera automatiquement de l'eau Ã la plaque"
        assert normalize_text(raw) == (
            "que vous soyez à la maison ou non, il ajoutera automatiquement de l'eau à la plaque"
        )

    def test_y_diaeresis_repaired_to_e_grave(self):
        raw = "fait de matiÃ¿re plastique, qui est sÃ»re et fiable, pas facile Ã porter"
        assert normalize_text(raw) == (
            "fait de matière plastique, qui est sûre et fiable, pas facile à porter"
        )

    def test_controle_exception(self):
        assert normalize_text("en contrÃ¿lant la température") == (
            "en contrôlant la température"
        )

    def test_cjk_brackets(self):
        assert normalize_text("Aquarium ã\u0080\u0090Double faceã\u0080\u0091 brosse") == (
            "Aquarium 【Double face】 brosse"
        )

    def test_truncated_apostrophe(self):
        assert normalize_text("Conseils dâ€utilisation et dâ€™entretien") == (
            "Conseils d'utilisation et d'entretien"
        )

    def test_latin1_view_apostrophe(self):
        assert normalize_text("la poussiÃ¿re, lâ\u0080\u0099aspirateur") == (
            "la poussière, l’aspirateur"
        )

    def test_uppercase_accent_c1_control(self):
        assert normalize_text("Ã\u0089cran Lcd Argent") == "Écran Lcd Argent"

    def test_spurious_lead_before_symbols(self):
        raw = "Température de 25Â°C Â« ProtÃ©gÃ© Â»"
        assert normalize_text(raw) == (
            "Température de 25°C « Protégé »"
        )

    def test_legitimate_a_circumflex_kept(self):
        raw = "Mangeoire Pour Poussins 1er Âge 1,5 Kg Avec Pattes, 19x20cm, Rouge"
        assert normalize_text(raw) == raw

    def test_nested_lead_byte(self):
        assert normalize_text("griffes Ã lumiÃ¿re LED Ã\u0082 Scissor") == (
            "griffes à lumière LED Scissor"
        )

    def test_legitimate_y_diaeresis_without_markers_kept(self):
        assert normalize_text("Bidon Haÿ-les-Roses 5L") == "Bidon Haÿ-les-Roses 5L"


class TestUppercaseTitles:
    def test_all_caps_recased(self):
        raw = "PEIGNE À AIGUILLES AUTONETTOYANTES : APRÈS AVOIR BROSSÉ LES POILS"
        assert normalize_text(raw) == (
            "Peigne à aiguilles autonettoyantes : après avoir brossé les poils"
        )

    def test_sentence_boundaries(self):
        raw = "ABREUVOIR POUR POULES 8 L. VÉRIFIEZ L'ARTICLE AU MOMENT DE LA RÉCEPTION !"
        assert normalize_text(raw) == (
            "Abreuvoir pour poules 8 l. Vérifiez l'article au moment de la réception !"
        )

    def test_below_alpha_threshold_untouched(self):
        assert normalize_text("USB LED 3000K") == "USB LED 3000K"

    def test_mixed_case_untouched(self):
        raw = "Abreuvoir Poussins Premier Âge ABREUVOIR POUSSINS 8 L dimensions standard"
        assert normalize_text(raw) == raw

    def test_caps_after_mojibake_fix(self):
        assert normalize_text("CONSEILS Dâ€UTILISATION POUR VOTRE ANIMAL") == (
            "Conseils d'utilisation pour votre animal"
        )


class TestWhitespace:
    def test_collapse_and_strip(self):
        assert normalize_text("  Bol   pour\tchat \n double  ") == "Bol pour chat double"

    def test_non_breaking_space_collapsed(self):
        assert normalize_text("Bol\u00a0pour chat") == "Bol pour chat"


class TestCleanTextUnchanged:
    def test_clean_description(self):
        raw = "Mangeoire pour poussins du premier âge. Fabriqué en Espagne."
        assert normalize_text(raw) == raw

    def test_clean_punctuation_and_digits(self):
        raw = "Dimensions : 40x40x5 cm, capacité 1,5 litre (rouge) !"
        assert normalize_text(raw) == raw


class TestIdempotence:
    @pytest.mark.parametrize("raw", REAL_WORLD_INPUTS)
    def test_real_world_inputs(self, raw):
        once = normalize_text(raw)
        assert normalize_text(once) == once

    def test_output_is_stable_for_empty(self):
        assert normalize_text(normalize_text(None)) == ""
