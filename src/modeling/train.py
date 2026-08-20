from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["ano", "id_municipio", "id_escola", "serie", "rede", "presenca", "preenchimento_caderno"]
NUMERIC = ["ano"]
CATEGORICAL = [x for x in FEATURES if x not in NUMERIC]


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict
    test: pd.DataFrame


def grouped_split(df: pd.DataFrame, random_state: int = 42):
    """Separa por municipio: nenhuma localidade aparece simultaneamente em treino e teste."""
    first = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=random_state)
    train_idx, test_idx = next(first.split(df, groups=df["id_municipio"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    second = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=random_state + 1)
    tr_idx, val_idx = next(second.split(train, groups=train["id_municipio"]))
    return train.iloc[tr_idx].copy(), train.iloc[val_idx].copy(), test


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=42
    ))])


def train(df: pd.DataFrame, random_state: int = 42) -> TrainResult:
    train_df, val_df, test_df = grouped_split(df, random_state)
    pipe = build_pipeline()
    # Busca restrita para manter reproducibilidade e custo controlado; CV tambem respeita municipios.
    search = GridSearchCV(
        pipe,
        {"model__n_estimators": [200, 400], "model__max_depth": [None, 16], "model__min_samples_leaf": [1, 5]},
        scoring="roc_auc",
        cv=GroupKFold(n_splits=3),
        n_jobs=-1,
    )
    search.fit(train_df[FEATURES], train_df["alfabetizado_binario"], groups=train_df["id_municipio"])
    val_prob = search.predict_proba(val_df[FEATURES])[:, 1]
    test_prob = search.predict_proba(test_df[FEATURES])[:, 1]
    test_pred = (test_prob >= .5).astype(int)
    metrics = {
        "best_params": search.best_params_,
        "validation_roc_auc": float(roc_auc_score(val_df["alfabetizado_binario"], val_prob)),
        "test_roc_auc": float(roc_auc_score(test_df["alfabetizado_binario"], test_prob)),
        "test_average_precision": float(average_precision_score(test_df["alfabetizado_binario"], test_prob)),
        "test_classification_report": classification_report(test_df["alfabetizado_binario"], test_pred, output_dict=True),
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
    }
    test_df["probabilidade_alfabetizado"] = test_prob
    test_df["predicao"] = test_pred
    return TrainResult(search.best_estimator_, metrics, test_df)


def save_result(result: TrainResult, model_path: str | Path, metrics_path: str | Path, predictions_path: str | Path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, model_path)
    Path(metrics_path).write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    result.test.to_csv(predictions_path, index=False)
