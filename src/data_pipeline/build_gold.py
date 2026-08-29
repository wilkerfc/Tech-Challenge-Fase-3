"""Constrói a camada Gold a partir do painel municipal da camada Silver.

O módulo não baixa dados nem inventa uma fonte externa. Ele recebe uma tabela
municipal normalizada, valida seu contrato e deriva, de maneira determinística,
as features temporais e os painéis agregados usados pelo projeto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = {
    "id_municipio",
    "nome",
    "sigla_uf",
    "id_uf",
    "nome_uf",
    "regiao",
    "ano",
    "indicador_alfabetizacao",
    "meta_municipio",
    "meta_nacional",
    "quantidade_matriculas",
    "PIB_per_capita",
    "IDHM",
}

ML_COLUMNS = [
    "id_municipio", "nome", "sigla_uf", "id_uf", "regiao", "ano",
    "indicador_lag1", "indicador_lag2", "tendencia_historica",
    "gap_historico_vs_meta_municipio", "gap_historico_vs_meta_nacional",
    "meta_municipio", "meta_nacional", "quantidade_matriculas",
    "PIB_per_capita", "IDHM", "indicador_alfabetizacao",
    "meta_atingida", "target_meta_atingida",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_silver(frame: pd.DataFrame) -> None:
    """Falha cedo quando a entrada viola o contrato da camada Silver."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("A camada Silver está vazia.")
    if frame[["id_municipio", "ano"]].isna().any().any():
        raise ValueError("id_municipio e ano não podem conter valores nulos.")
    duplicated = frame.duplicated(["id_municipio", "ano"])
    if duplicated.any():
        raise ValueError("Há mais de uma linha para o mesmo município e ano.")
    numeric = [
        "ano", "indicador_alfabetizacao", "meta_municipio", "meta_nacional",
        "quantidade_matriculas", "PIB_per_capita", "IDHM",
    ]
    if frame[numeric].isna().any().any():
        raise ValueError("As colunas numéricas obrigatórias não podem ser nulas.")
    if not frame["indicador_alfabetizacao"].between(0, 100).all():
        raise ValueError("indicador_alfabetizacao deve estar entre 0 e 100.")
    if not frame["meta_municipio"].between(0, 100).all():
        raise ValueError("meta_municipio deve estar entre 0 e 100.")
    if not frame["meta_nacional"].between(0, 100).all():
        raise ValueError("meta_nacional deve estar entre 0 e 100.")
    if not frame["IDHM"].between(0, 1).all():
        raise ValueError("IDHM deve estar entre 0 e 1.")
    if (frame[["quantidade_matriculas", "PIB_per_capita"]] < 0).any().any():
        raise ValueError("Matrículas e PIB per capita não podem ser negativos.")


def build_gold_tables(silver: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Retorna features de ML e painéis UF/nacional derivados da Silver."""
    frame = silver.copy()
    validate_silver(frame)
    frame = frame.sort_values(["id_municipio", "ano"]).reset_index(drop=True)

    # O alvo é sempre recalculado pela regra documentada. Uma coluna homônima
    # recebida da origem nunca é aceita como verdade pronta.
    frame["meta_atingida"] = frame["indicador_alfabetizacao"] >= frame["meta_municipio"]
    frame["gap_vs_meta_municipio"] = (
        frame["indicador_alfabetizacao"] - frame["meta_municipio"]
    ).round(2)
    frame["gap_vs_meta_nacional"] = (
        frame["indicador_alfabetizacao"] - frame["meta_nacional"]
    ).round(2)
    frame["status_meta_municipio"] = frame["meta_atingida"].map(
        {True: "ATINGIDA", False: "NAO_ATINGIDA"}
    )

    by_city = frame.groupby("id_municipio", sort=False)["indicador_alfabetizacao"]
    frame["indicador_lag1"] = by_city.shift(1)
    frame["indicador_lag2"] = by_city.shift(2)
    frame["tendencia_historica"] = (frame["indicador_lag1"] - frame["indicador_lag2"]).round(2)
    frame["gap_historico_vs_meta_municipio"] = (
        frame["indicador_lag1"] - frame["meta_municipio"]
    ).round(2)
    frame["gap_historico_vs_meta_nacional"] = (
        frame["indicador_lag1"] - frame["meta_nacional"]
    ).round(2)
    frame["target_meta_atingida"] = frame["meta_atingida"].astype("int8")
    ml_features = frame.loc[frame["indicador_lag1"].notna(), ML_COLUMNS].reset_index(drop=True)

    uf = (
        frame.groupby(["id_uf", "sigla_uf", "nome_uf", "ano"], as_index=False)
        .agg(
            indicador_medio=("indicador_alfabetizacao", "mean"),
            indicador_min=("indicador_alfabetizacao", "min"),
            indicador_max=("indicador_alfabetizacao", "max"),
            total_municipios=("id_municipio", "nunique"),
            municipios_meta_atingida=("meta_atingida", "sum"),
            matriculas_total=("quantidade_matriculas", "sum"),
        )
        .sort_values(["id_uf", "ano"])
    )
    for column in ["indicador_medio", "indicador_min", "indicador_max"]:
        uf[column] = uf[column].round(2)
    uf["pct_municipios_meta_atingida"] = (
        100 * uf["municipios_meta_atingida"] / uf["total_municipios"]
    ).round(2)
    uf["variacao_yoy"] = uf.groupby("id_uf")["indicador_medio"].diff().fillna(0).round(2)

    national = (
        frame.groupby("ano", as_index=False)
        .agg(
            indicador_medio_nacional=("indicador_alfabetizacao", "mean"),
            total_municipios=("id_municipio", "nunique"),
            municipios_meta_atingida=("meta_atingida", "sum"),
            total_matriculas=("quantidade_matriculas", "sum"),
            meta_nacional=("meta_nacional", "first"),
        )
        .sort_values("ano")
    )
    national["indicador_medio_nacional"] = national["indicador_medio_nacional"].round(2)
    national["pct_municipios_alfabetizados"] = (
        100 * national["municipios_meta_atingida"] / national["total_municipios"]
    ).round(2)
    national["gap_meta"] = (
        national["indicador_medio_nacional"] - national["meta_nacional"]
    ).round(2)

    return {"ml_features": ml_features, "evolucao_uf": uf, "painel_nacional": national}


def write_gold(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        table.to_parquet(output_dir / f"{name}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrói a camada Gold municipal.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "indicador_municipio.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "generated_gold")
    parser.add_argument("--source-uri", default="snapshot local sem URI de extração registrada")
    args = parser.parse_args()

    source = args.input.resolve()
    silver = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    tables = build_gold_tables(silver)
    write_gold(tables, args.output_dir)
    manifest = {
        "source_uri": args.source_uri,
        "input_file": str(source),
        "input_sha256": _sha256(source),
        "input_rows": len(silver),
        "outputs": {name: len(table) for name, table in tables.items()},
        "target_rule": "indicador_alfabetizacao >= meta_municipio",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
