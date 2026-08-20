from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_COLUMNS = [
    "ano", "id_municipio", "id_escola", "serie", "rede", "presenca",
    "preenchimento_caderno", "alfabetizado",
]


def normalize_target(values: pd.Series) -> pd.Series:
    """Converte as codificacoes textuais do INEP em alvo binario, sem inferir casos ambiguos."""
    normalized = values.astype("string").str.strip().str.upper()
    positive = {"SIM", "S", "ALFABETIZADO", "1", "TRUE"}
    negative = {"NAO", "NÃO", "N", "NAO ALFABETIZADO", "NÃO ALFABETIZADO", "0", "FALSE"}
    target = pd.Series(pd.NA, index=values.index, dtype="Int64")
    target[normalized.isin(positive)] = 1
    target[normalized.isin(negative)] = 0
    return target


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    df = raw[RAW_COLUMNS].copy()
    df["alfabetizado_binario"] = normalize_target(df["alfabetizado"])
    df = df.dropna(subset=["alfabetizado_binario", "id_municipio"])
    df["alfabetizado_binario"] = df["alfabetizado_binario"].astype(int)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    for col in ["id_municipio", "id_escola", "serie", "rede", "presenca", "preenchimento_caderno"]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    return df.drop(columns="alfabetizado")


def load_parquet(path: str | Path) -> pd.DataFrame:
    return prepare_frame(pd.read_parquet(path))
