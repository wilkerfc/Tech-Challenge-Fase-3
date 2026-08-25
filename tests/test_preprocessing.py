import pandas as pd

from src.preprocessing.data import build_modeling_table


def test_modeling_table_uses_past_only_and_official_target():
    uf = pd.DataFrame([
        {"ano": 2023, "sigla_uf": "AA", "serie": "2", "rede": "5", "taxa_alfabetizacao": 50, "media_portugues": 700},
        {"ano": 2024, "sigla_uf": "AA", "serie": "2", "rede": "5", "taxa_alfabetizacao": 61, "media_portugues": 710},
        {"ano": 2023, "sigla_uf": "BB", "serie": "2", "rede": "5", "taxa_alfabetizacao": 40, "media_portugues": 690},
        {"ano": 2024, "sigla_uf": "BB", "serie": "2", "rede": "5", "taxa_alfabetizacao": 49, "media_portugues": 695},
    ])
    meta = pd.DataFrame([
        {"ano": 2024, "sigla_uf": "AA", "rede": "Pública", "meta_alfabetizacao_2024": 60},
        {"ano": 2024, "sigla_uf": "BB", "rede": "Pública", "meta_alfabetizacao_2024": 50},
    ])
    result = build_modeling_table(uf, meta)
    assert result["atingiu_meta_2024"].tolist() == [1, 0]
    assert result["gap_2023_para_meta_2024"].tolist() == [-10, -10]
