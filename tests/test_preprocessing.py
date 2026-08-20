import pandas as pd
from src.preprocessing.data import normalize_target


def test_normalize_target_handles_expected_values():
    result = normalize_target(pd.Series(["Sim", "NÃO", "alfabetizado", "nao alfabetizado", "desconhecido"]))
    assert result.tolist()[:4] == [1, 0, 1, 0]
    assert pd.isna(result.iloc[4])
