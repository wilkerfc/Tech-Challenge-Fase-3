from src.preprocessing.pipeline import EXCLUDE_COLUMNS, load_and_split_data


def test_temporal_split_and_leakage_columns(tmp_path, silver_frame):
    from src.data_pipeline.build_gold import build_gold_tables

    path = tmp_path / "ml_features.parquet"
    build_gold_tables(silver_frame)["ml_features"].to_parquet(path, index=False)
    X_train, y_train, groups, X_test, y_test, _ = load_and_split_data(
        path, train_years=[2023], test_year=2024
    )
    assert len(X_train) == len(y_train) == len(groups) == 2
    assert len(X_test) == len(y_test) == 2
    assert set(EXCLUDE_COLUMNS).isdisjoint(X_train.columns)
    assert "indicador_lag1" in X_train.columns
