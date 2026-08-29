import pandas as pd
import pytest

from src.data_pipeline.build_gold import build_gold_tables, validate_silver


def test_features_use_only_previous_years(silver_frame):
    ml = build_gold_tables(silver_frame)["ml_features"]
    row = ml.query("id_municipio == 110001 and ano == 2024").iloc[0]
    assert row["indicador_lag1"] == 50.0
    assert row["indicador_lag2"] == 40.0
    assert row["tendencia_historica"] == 10.0
    assert row["gap_historico_vs_meta_municipio"] == -10.0


def test_target_is_recalculated_from_documented_rule(silver_frame):
    silver_frame["meta_atingida"] = True  # coluna potencialmente contaminada
    ml = build_gold_tables(silver_frame)["ml_features"]
    actual = ml.query("id_municipio == 120001 and ano == 2024").iloc[0]
    assert bool(actual["meta_atingida"]) is False
    assert actual["target_meta_atingida"] == 0


def test_first_year_is_not_used_without_a_lag(silver_frame):
    ml = build_gold_tables(silver_frame)["ml_features"]
    assert set(ml["ano"]) == {2023, 2024}
    assert len(ml) == 4


def test_aggregations_are_consistent(silver_frame):
    tables = build_gold_tables(silver_frame)
    national_2024 = tables["painel_nacional"].query("ano == 2024").iloc[0]
    assert national_2024["total_municipios"] == 2
    assert national_2024["municipios_meta_atingida"] == 1
    assert national_2024["pct_municipios_alfabetizados"] == 50.0
    assert national_2024["indicador_medio_nacional"] == 60.0


@pytest.mark.parametrize("column", ["indicador_alfabetizacao", "meta_municipio", "meta_nacional"])
def test_percentages_must_be_in_valid_range(silver_frame, column):
    silver_frame.loc[0, column] = 101
    with pytest.raises(ValueError, match="entre 0 e 100"):
        validate_silver(silver_frame)


def test_duplicate_municipality_year_is_rejected(silver_frame):
    duplicated = pd.concat([silver_frame, silver_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="mesmo município e ano"):
        validate_silver(duplicated)


def test_missing_required_column_is_rejected(silver_frame):
    with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
        validate_silver(silver_frame.drop(columns="IDHM"))
