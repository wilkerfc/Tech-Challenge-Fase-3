from __future__ import annotations

from pathlib import Path

import pandas as pd

LEVEL_COLUMNS = [f"proporcao_aluno_nivel_{i}" for i in range(9)]


def load_sources(uf_path: str | Path, meta_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    uf = pd.read_csv(uf_path)
    meta = pd.read_csv(meta_path)
    required_uf = {"ano", "sigla_uf", "serie", "rede", "taxa_alfabetizacao", "media_portugues"}
    required_meta = {"ano", "sigla_uf", "rede", "meta_alfabetizacao_2024"}
    if missing := required_uf - set(uf.columns):
        raise ValueError(f"Colunas ausentes em uf.csv: {sorted(missing)}")
    if missing := required_meta - set(meta.columns):
        raise ValueError(f"Colunas ausentes em meta_alfabetizacao_uf.csv: {sorted(missing)}")
    return uf, meta


def build_modeling_table(uf: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Usa somente 2023 para explicar o atingimento da meta oficial em 2024.

    A rede 5 significa "Pública (Estadual e Municipal)", conforme o dicionário
    oficial da própria base. A taxa observada em 2024 é usada apenas para criar
    o alvo e nunca entra nas variáveis preditoras.
    """
    public = uf.loc[uf["rede"].astype(str) == "5"].copy()
    lag = public.loc[public["ano"] == 2023, ["sigla_uf", "taxa_alfabetizacao", "media_portugues"]]
    lag = lag.rename(columns={
        "taxa_alfabetizacao": "taxa_alfabetizacao_2023",
        "media_portugues": "media_portugues_2023",
    })
    actual = public.loc[public["ano"] == 2024, ["sigla_uf", "taxa_alfabetizacao"]]
    actual = actual.rename(columns={"taxa_alfabetizacao": "taxa_alfabetizacao_2024"})
    target_meta = meta.loc[meta["ano"] == 2024, ["sigla_uf", "meta_alfabetizacao_2024"]].copy()
    target_meta["meta_alfabetizacao_2024"] = pd.to_numeric(target_meta["meta_alfabetizacao_2024"], errors="coerce")
    frame = lag.merge(actual, on="sigla_uf", how="inner").merge(target_meta, on="sigla_uf", how="inner")
    numeric = ["taxa_alfabetizacao_2023", "media_portugues_2023", "taxa_alfabetizacao_2024", "meta_alfabetizacao_2024"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric).copy()
    frame["gap_2023_para_meta_2024"] = frame["taxa_alfabetizacao_2023"] - frame["meta_alfabetizacao_2024"]
    frame["atingiu_meta_2024"] = (frame["taxa_alfabetizacao_2024"] >= frame["meta_alfabetizacao_2024"]).astype(int)
    return frame.sort_values("sigla_uf").reset_index(drop=True)


def build_cluster_table(uf: pd.DataFrame) -> pd.DataFrame:
    cols = ["sigla_uf", "taxa_alfabetizacao", "media_portugues", *LEVEL_COLUMNS]
    frame = uf.loc[(uf["ano"] == 2024) & (uf["rede"].astype(str) == "5"), cols].copy()
    for col in cols[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["taxa_alfabetizacao", "media_portugues"]).reset_index(drop=True)
