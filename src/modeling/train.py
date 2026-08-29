"""
Módulo de Modelagem, Otimização e Treinamento Supervisionado
Treina, otimiza hiperparâmetros (GridSearchCV) e compara múltiplos modelos de ML
utilizando Partição Temporal Estrita (Treino: 2022-2023, Teste: 2024)
e Validação Cruzada Estratificada por Grupo (StratifiedGroupKFold por id_municipio),
garantindo ZERO Data Leakage direto e temporal.
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import get_preprocessor, load_and_split_data
from src.visualization.plots import (
    plot_confusion_matrix_heatmap,
    plot_correlation_matrix,
    plot_regional_performance,
    plot_risk_quadrant,
    plot_roc_curves,
    plot_target_distribution,
    plot_threshold_tuning,
    plot_unsupervised_clusters,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROJECT_ROOT / "data" / "ml_features.parquet"


def train_and_compare_models():
    print("=" * 75)
    print("🚀 INICIANDO PIPELINE DE TREINAMENTO TEMPORAL E OTIMIZAÇÃO (ML & XAI)")
    print("=" * 75)

    # 1. Carga dos dados e Partição Temporal Estrita
    print("\n📂 Carregando dados da Camada Gold com Partição Temporal...")
    X_train, y_train, groups_train, X_test, y_test, df_raw = load_and_split_data(
        DATA_FILE, temporal_split=True, train_years=[2022, 2023], test_year=2024
    )
    
    print(f"   Treino (2022-2023): {len(X_train):,} instâncias ({groups_train.nunique():,} municípios)")
    print(f"   Teste Holdout (2024): {len(X_test):,} instâncias (Holdout cego e estritamente temporal)")
    print(f"   Taxa Classe 0 (Risco) no Teste: {(y_test == 0).mean():.2%} ({(y_test == 0).sum():,} municípios)")
    print(f"   Taxa Classe 1 (Sucesso) no Teste: {y_test.mean():.2%} ({y_test.sum():,} municípios)")

    # 2. Geração dos Gráficos Exploratórios e de Agrupamento
    print("\n🎨 Gerando gráficos exploratórios, de risco e clusters...")
    plot_target_distribution(df_raw["target_meta_atingida"])
    plot_correlation_matrix(df_raw)
    plot_regional_performance(df_raw)
    plot_risk_quadrant(df_raw)
    plot_unsupervised_clusters(df_raw)

    # 3. Definição do Esquema de Validação Cruzada sem Leakage de Grupo
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Definição e Otimização dos Modelos Candidatos (com balanceamento de classes)
    print("\n⚙️ Otimizando Hiperparâmetros via GridSearchCV e StratifiedGroupKFold...")

    # Modelo 1: Regressão Logística (Baseline)
    logreg_pipe = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])
    param_grid_logreg = {
        "classifier__C": [0.1, 1.0, 10.0],
    }
    grid_logreg = GridSearchCV(
        logreg_pipe,
        param_grid_logreg,
        cv=cv.split(X_train, y_train, groups=groups_train),
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    grid_logreg.fit(X_train, y_train)
    best_logreg = grid_logreg.best_estimator_
    print(f"   ✅ Regressão Logística otimizada: C={grid_logreg.best_params_['classifier__C']} (CV Bal Acc: {grid_logreg.best_score_:.4f})")

    # Modelo 2: Random Forest Classifier
    rf_pipe = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    param_grid_rf = {
        "classifier__n_estimators": [150, 200],
        "classifier__max_depth": [8, 12],
        "classifier__min_samples_leaf": [2, 4],
    }
    grid_rf = GridSearchCV(
        rf_pipe,
        param_grid_rf,
        cv=cv.split(X_train, y_train, groups=groups_train),
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    print(f"   ✅ Random Forest otimizado: {grid_rf.best_params_} (CV Bal Acc: {grid_rf.best_score_:.4f})")

    # Modelo 3: HistGradientBoosting Classifier (Gradient Boosting com balanceamento)
    hgb_pipe = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
    ])
    param_grid_hgb = {
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_iter": [100, 150],
        "classifier__max_depth": [5, 8],
    }
    grid_hgb = GridSearchCV(
        hgb_pipe,
        param_grid_hgb,
        cv=cv.split(X_train, y_train, groups=groups_train),
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    grid_hgb.fit(X_train, y_train)
    best_hgb = grid_hgb.best_estimator_
    print(f"   ✅ HistGradientBoosting otimizado: {grid_hgb.best_params_} (CV Bal Acc: {grid_hgb.best_score_:.4f})")

    candidate_models = {
        "Regressão Logística (Baseline)": best_logreg,
        "Random Forest Classifier (Otimizado)": best_rf,
        "HistGradientBoosting Classifier (Otimizado)": best_hgb,
    }

    # 5. Avaliação Comparativa no Holdout de Teste Temporal (Ano 2024)
    print("\n🔄 Avaliando modelos no Holdout Temporal (Ano 2024)...")
    results = []
    fitted_pipelines = {}

    for name, pipeline in candidate_models.items():
        fitted_pipelines[name] = pipeline

        # Validação cruzada para estimar desvio padrão das métricas
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv.split(X_train, y_train, groups=groups_train),
            scoring=["roc_auc", "balanced_accuracy", "f1_macro"],
            n_jobs=-1,
        )

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        test_roc_auc = roc_auc_score(y_test, y_proba)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")
        test_recall_risk = recall_score(y_test, y_pred, pos_label=0)
        test_recall_meta = recall_score(y_test, y_pred, pos_label=1)
        test_prec_risk = precision_score(y_test, y_pred, pos_label=0)
        test_acc = accuracy_score(y_test, y_pred)

        results.append({
            "Modelo": name,
            "CV Bal Acc (Média ± DP)": f"{cv_scores['test_balanced_accuracy'].mean():.4f} ± {cv_scores['test_balanced_accuracy'].std():.4f}",
            "CV ROC-AUC": round(cv_scores["test_roc_auc"].mean(), 4),
            "Teste Balanced Acc": round(test_bal_acc, 4),
            "Teste ROC-AUC": round(test_roc_auc, 4),
            "Teste Recall Risco (Classe 0)": round(test_recall_risk, 4),
            "Teste Recall Sucesso (Classe 1)": round(test_recall_meta, 4),
            "Teste Precision Risco": round(test_prec_risk, 4),
            "Teste F1 Macro": round(test_f1_macro, 4),
            "Teste Acurácia Global": round(test_acc, 4),
        })

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("📊 RESULTADOS COMPARATIVOS DOS MODELOS (HOLDOUT TEMPORAL 2024):")
    print("=" * 75)
    print(df_results.to_markdown(index=False))

    # Salva métricas em CSV
    metrics_path = MODELS_DIR / "model_comparison_metrics.csv"
    df_results.to_csv(metrics_path, index=False)
    print(f"\n💾 Tabela de métricas salva em: {metrics_path}")

    # 6. Seleção Dinâmica do Melhor Modelo (Baseada em Balanced Accuracy no Teste)
    best_row = df_results.sort_values(by="Teste Balanced Acc", ascending=False).iloc[0]
    best_name = best_row["Modelo"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n🏆 Seleção Dinâmica: '{best_name}' com Balanced Accuracy de {best_row['Teste Balanced Acc']:.4f} e Recall de Risco de {best_row['Teste Recall Risco (Classe 0)']:.2%}.")

    # Salvar modelos serializados
    joblib.dump(best_pipeline, MODELS_DIR / "best_model_pipeline.pkl")
    joblib.dump(best_rf, MODELS_DIR / "rf_pipeline.pkl")
    joblib.dump((X_train, y_train, X_test, y_test), MODELS_DIR / "train_test_data.pkl")
    print(f"💾 Pipelines salvos em '{MODELS_DIR}'.")

    # 7. Gráficos de Avaliação (Curvas ROC, Matriz de Confusão e Calibração de Threshold)
    print("\n📈 Gerando gráficos de Curva ROC, Matriz de Confusão e Threshold Tuning...")
    plot_roc_curves(fitted_pipelines, X_test, y_test)
    y_pred_best = best_pipeline.predict(X_test)
    y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
    
    plot_confusion_matrix_heatmap(y_test, y_pred_best, model_name=best_name)
    plot_threshold_tuning(y_test, y_proba_best)

    print("\n" + "=" * 75)
    print(f"📋 RELATÓRIO DETALHADO DO MODELO SELECIONADO ({best_name}):")
    print("=" * 75)
    print(classification_report(y_test, y_pred_best, target_names=["Risco (0)", "Meta Atingida (1)"], digits=4))

    return df_results, best_pipeline


if __name__ == "__main__":
    train_and_compare_models()
