from __future__ import annotations

import argparse
import os
from pathlib import Path
from src.preprocessing.extract import extract_alunos
from src.preprocessing.data import load_parquet
from src.modeling.train import train, save_result
from src.evaluation.importance import feature_importance
from src.visualization.charts import save_charts


def main():
    parser = argparse.ArgumentParser(description="Pipeline reproduzivel de predicao de alfabetizacao")
    parser.add_argument("--extract", action="store_true", help="Consulta a tabela publica no BigQuery")
    parser.add_argument("--input", default="data/raw/alunos.parquet")
    parser.add_argument("--max-rows", type=int, default=int(os.getenv("BQ_MAX_ROWS", "500000")))
    args = parser.parse_args()
    raw_path = Path(args.input)
    if args.extract:
        project = os.environ.get("GCP_PROJECT_ID")
        if not project:
            raise SystemExit("Defina GCP_PROJECT_ID antes de usar --extract.")
        extract_alunos(project, args.max_rows, raw_path)
    if not raw_path.exists():
        raise SystemExit("Dataset ausente. Execute com --extract ou informe --input.")
    df = load_parquet(raw_path)
    result = train(df, int(os.getenv("RANDOM_STATE", "42")))
    save_result(result, "reports/model.joblib", "reports/metricas.json", "reports/predicoes_teste.csv")
    importance = feature_importance(result.pipeline)
    importance.to_csv("reports/importancia_variaveis.csv", index=False)
    save_charts(df, importance, "images")
    print("Concluido. Consulte reports/ e images/.")


if __name__ == "__main__":
    main()
