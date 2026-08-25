from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.modeling.clustering import cluster_ufs
from src.modeling.train import save_result, train_and_evaluate
from src.preprocessing.data import build_cluster_table, build_modeling_table, load_sources
from src.visualization.charts import save_charts


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise reproduzível de risco de alfabetização por UF")
    parser.add_argument("--uf", default="data/raw/uf.csv")
    parser.add_argument("--meta", default="data/raw/meta_alfabetizacao_uf.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    uf, meta = load_sources(args.uf, args.meta)
    modeling = build_modeling_table(uf, meta)
    clusters, cluster_metrics = cluster_ufs(build_cluster_table(uf), args.random_state)
    result = train_and_evaluate(modeling, args.random_state)
    save_result(result, "reports")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    modeling.to_csv("data/processed/base_modelagem.csv", index=False)
    clusters.to_csv("reports/clusters_ufs.csv", index=False)
    Path("reports/clustering_metricas.json").write_text(json.dumps(cluster_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "linhas_uf": int(len(uf)), "anos_uf": sorted(int(x) for x in uf["ano"].unique()),
        "ufs_pareadas_modelagem": int(len(modeling)),
        "atingiram_meta_2024": int(modeling["atingiu_meta_2024"].sum()),
        "nao_atingiram_meta_2024": int((1 - modeling["atingiu_meta_2024"]).sum()),
        "ufs_maior_deficit_observado": result.predictions.sort_values("gap_observado_2024").head(5)["sigla_uf"].tolist(),
    }
    Path("reports/resumo_dados.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_charts(modeling, result.predictions, result.importance, clusters, "images")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
