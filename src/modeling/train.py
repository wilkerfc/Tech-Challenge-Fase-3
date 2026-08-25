from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["taxa_alfabetizacao_2023", "media_portugues_2023", "meta_alfabetizacao_2024"]
TARGET = "atingiu_meta_2024"


@dataclass
class TrainResult:
    model: Pipeline
    metrics: dict
    predictions: pd.DataFrame
    importance: pd.DataFrame


def _candidate_models(random_state: int):
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=2000)),
    ])
    forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1)),
    ])
    return {
        "logistic_regression": (logistic, {"model__C": [0.1, 1.0, 10.0]}),
        "random_forest": (forest, {"model__n_estimators": [200], "model__max_depth": [2, None], "model__min_samples_leaf": [2]}),
    }


def _metrics(y: pd.Series, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train_and_evaluate(frame: pd.DataFrame, random_state: int = 42) -> TrainResult:
    """Validação cruzada aninhada; cada UF recebe previsão fora da amostra."""
    x, y = frame[FEATURES], frame[TARGET]
    if len(frame) < 20 or y.nunique() != 2:
        raise ValueError("A amostra precisa de ao menos 20 UFs e das duas classes.")
    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + 1)
    model_scores: dict[str, dict] = {}
    model_probs: dict[str, np.ndarray] = {}
    candidates = _candidate_models(random_state)
    for name, (pipeline, grid) in candidates.items():
        probs = np.zeros(len(frame), dtype=float)
        chosen: list[dict] = []
        for train_idx, test_idx in outer.split(x, y):
            search = GridSearchCV(clone(pipeline), grid, scoring="roc_auc", cv=inner, n_jobs=1)
            search.fit(x.iloc[train_idx], y.iloc[train_idx])
            probs[test_idx] = search.best_estimator_.predict_proba(x.iloc[test_idx])[:, 1]
            chosen.append(search.best_params_)
        model_probs[name] = probs
        model_scores[name] = {**_metrics(y, probs), "outer_folds": 4, "inner_folds": 3, "selected_params_by_fold": chosen}

    dummy_probs = np.zeros(len(frame), dtype=float)
    for train_idx, test_idx in outer.split(x, y):
        dummy = DummyClassifier(strategy="prior").fit(x.iloc[train_idx], y.iloc[train_idx])
        dummy_probs[test_idx] = dummy.predict_proba(x.iloc[test_idx])[:, 1]
    model_scores["dummy_prior"] = _metrics(y, dummy_probs)

    best_name = max(candidates, key=lambda name: model_scores[name]["roc_auc"])
    best_pipeline, best_grid = candidates[best_name]
    final_search = GridSearchCV(best_pipeline, best_grid, scoring="roc_auc", cv=inner, n_jobs=1)
    final_search.fit(x, y)
    model = final_search.best_estimator_
    pred = frame[["sigla_uf", "taxa_alfabetizacao_2024", "meta_alfabetizacao_2024", TARGET]].copy()
    pred["score_atingimento_oof_experimental"] = model_probs[best_name]
    pred["score_nao_atingimento_experimental"] = 1 - pred["score_atingimento_oof_experimental"]
    pred["predicao_oof"] = (pred["score_atingimento_oof_experimental"] >= 0.5).astype(int)
    pred["gap_observado_2024"] = pred["taxa_alfabetizacao_2024"] - pred["meta_alfabetizacao_2024"]
    pred = pred.sort_values("score_nao_atingimento_experimental", ascending=False).reset_index(drop=True)
    values = (np.abs(model.named_steps["model"].coef_[0]) if best_name == "logistic_regression" else model.named_steps["model"].feature_importances_)
    importance = pd.DataFrame({"variavel": FEATURES, "importancia_modelo": values}).sort_values("importancia_modelo", ascending=False)
    metrics = {
        "n_ufs": int(len(frame)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "validation": "nested_stratified_cross_validation",
        "models": model_scores,
        "selected_model": best_name,
        "final_best_params": final_search.best_params_,
        "beats_dummy_roc_auc": bool(model_scores[best_name]["roc_auc"] > model_scores["dummy_prior"]["roc_auc"]),
        "deployment_recommendation": "DO_NOT_DEPLOY",
        "caveat": "Amostra pequena (24 UFs pareadas); métricas têm alta incerteza e não validam uso operacional.",
    }
    return TrainResult(model, metrics, pred, importance)


def save_result(result: TrainResult, reports_dir: str | Path) -> None:
    out = Path(reports_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, out / "modelo.joblib")
    (out / "metricas.json").write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    result.predictions.to_csv(out / "predicoes_ufs.csv", index=False)
    result.importance.to_csv(out / "importancia_variaveis.csv", index=False)
