from __future__ import annotations

from pathlib import Path
from google.cloud import bigquery


def extract_alunos(project_id: str, max_rows: int, output: str | Path) -> Path:
    """Executa a consulta parametrizada no BigQuery e persiste um snapshot local ignorado pelo Git."""
    sql = (Path(__file__).parents[2] / "scripts" / "query_alunos.sql").read_text(encoding="utf-8")
    client = bigquery.Client(project=project_id)
    config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("max_rows", "INT64", max_rows)]
    )
    df = client.query(sql, job_config=config).result().to_dataframe(create_bqstorage_client=False)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    return target
