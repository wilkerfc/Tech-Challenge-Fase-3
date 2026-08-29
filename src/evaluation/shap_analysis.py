"""
Módulo de Avaliação e Interpretabilidade com SHAP (Explainable AI - XAI)
Calcula valores SHAP para explicar as decisões dos modelos nos âmbitos global e local,
identificando os fatores socioeconômicos e pedagógicos com maior impacto na alfabetização.
Gera gráficos de impacto global (Beeswarm, Bar Plot), dependência não-linear e explicações locais (Waterfall).
"""

from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_and_explain(sample_size: int = 2000):
    print("=" * 75)
    print("🧠 INICIANDO ANÁLISE DE INTERPRETABILIDADE COM SHAP (XAI)")
    print("=" * 75)

    rf_path = MODELS_DIR / "rf_pipeline.pkl"
    data_path = MODELS_DIR / "train_test_data.pkl"

    if not rf_path.exists() or not data_path.exists():
        raise FileNotFoundError("Modelos não encontrados. Execute 'src/modeling/train.py' primeiro.")

    clf = joblib.load(rf_path)
    X_train, y_train, X_test, y_test = joblib.load(data_path)

    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["classifier"]

    # Amostragem representativa para cálculo de SHAP
    if len(X_test) > sample_size:
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        y_test_sample = y_test.loc[X_test_sample.index]
    else:
        X_test_sample = X_test
        y_test_sample = y_test

    print(f"📊 Transformando {len(X_test_sample):,} amostras do teste temporal (Ano 2024)...")
    X_test_transformed = preprocessor.transform(X_test_sample)

    # Obter nomes das features transformadas
    num_cols = NUMERIC_FEATURES
    cat_cols = preprocessor.transformers_[1][1].named_steps["encoder"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = np.array(num_cols + cat_cols)

    # Dicionário de tradução para rótulos elegantes nos gráficos
    clean_labels = {
        "indicador_lag1": "Indicador Ano Anterior (t-1)",
        "indicador_lag2": "Indicador 2 Anos Anteriores (t-2)",
        "tendencia_historica": "Tendência Histórica (t-1 - t-2)",
        "gap_historico_vs_meta_municipio": "Gap vs Meta Municipal",
        "gap_historico_vs_meta_nacional": "Gap vs Meta Nacional",
        "meta_municipio": "Meta Municipal Pactuada",
        "meta_nacional": "Meta Nacional Brasil",
        "quantidade_matriculas": "Volume de Matrículas",
        "PIB_per_capita": "PIB per capita Municipal",
        "IDHM": "Índice de Desenv. Humano (IDHM)",
    }
    feature_names_clean = np.array([clean_labels.get(f, f.replace("sigla_uf_", "UF: ").replace("regiao_", "Região: ")) for f in feature_names])

    print("⚙️ Calculando SHAP Values via TreeExplainer no Random Forest Otimizado...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    # Formato do SHAP para classificação binária
    if isinstance(shap_values, list):
        shap_values_target = shap_values[1]  # Classe 1: Meta Atingida
    elif len(np.shape(shap_values)) == 3:
        shap_values_target = shap_values[:, :, 1]
    else:
        shap_values_target = shap_values

    # 1. SHAP Summary Plot (Beeswarm)
    print("\n🎨 Gerando SHAP Summary Plot (Impacto e Direção das Features)...")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values_target,
        X_test_transformed,
        feature_names=feature_names_clean,
        max_display=12,
        show=False,
    )
    plt.title("Impacto das Variáveis na Predição de Alfabetização (SHAP Beeswarm)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = IMAGES_DIR / "06_shap_summary_plot.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {summary_path.name}")

    # 2. SHAP Bar Plot (Importância Média Global)
    print("🎨 Gerando SHAP Feature Importance (Bar Plot)...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_target,
        X_test_transformed,
        feature_names=feature_names_clean,
        plot_type="bar",
        max_display=12,
        show=False,
    )
    plt.title("Ranking de Importância Global das Variáveis (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    bar_path = IMAGES_DIR / "07_shap_bar_importance.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {bar_path.name}")

    # 3. SHAP Dependence Plot (Relação não-linear de Indicador t-1 e IDHM)
    print("🎨 Gerando SHAP Dependence Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Índice dinâmico das features (robusto a reordenação)
    feature_list = feature_names.tolist()
    idx_lag1 = feature_list.index("indicador_lag1")
    idx_idhm = feature_list.index("IDHM")

    # Scatter manual de dependência com cor por IDHM
    scatter1 = ax1.scatter(
        X_test_transformed[:, idx_lag1],
        shap_values_target[:, idx_lag1],
        c=X_test_transformed[:, idx_idhm],
        cmap="viridis",
        alpha=0.6,
        s=25
    )
    ax1.set_xlabel("Indicador de Alfabetização no Ano Anterior (Padronizado)", fontweight="bold")
    ax1.set_ylabel("Valor SHAP (Impacto na Probabilidade de Sucesso)", fontweight="bold")
    ax1.set_title("Efeito do Indicador Prévio (t-1) na Predição", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.7)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("IDHM Padronizado", fontweight="bold")

    # Scatter de IDHM com cor por Indicador Lag 1
    scatter2 = ax2.scatter(
        X_test_transformed[:, idx_idhm],
        shap_values_target[:, idx_idhm],
        c=X_test_transformed[:, idx_lag1],
        cmap="coolwarm",
        alpha=0.6,
        s=25
    )
    ax2.set_xlabel("Índice de Desenvolvimento Humano Municipal (IDHM Padronizado)", fontweight="bold")
    ax2.set_ylabel("Valor SHAP (Impacto)", fontweight="bold")
    ax2.set_title("Efeito do IDHM Municipal na Predição", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Indicador t-1 Padronizado", fontweight="bold")

    plt.suptitle("Análise de Dependência Não-Linear e Interação via SHAP (XAI)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    dep_path = IMAGES_DIR / "08_shap_dependence_plot.png"
    plt.savefig(dep_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {dep_path.name}")

    # 4. SHAP Local Explanation (Comparativo de Caso em Risco vs Caso de Sucesso)
    print("🎨 Gerando SHAP Local Explanation (Comparação de Casos Individuais)...")
    
    # Encontrar um caso real de risco (target = 0) e um de sucesso (target = 1)
    risk_indices = np.where(y_test_sample.values == 0)[0]
    meta_indices = np.where(y_test_sample.values == 1)[0]
    
    risk_idx = risk_indices[0] if len(risk_indices) > 0 else 0
    meta_idx = meta_indices[0] if len(meta_indices) > 0 else 1

    fig, (ax_r, ax_m) = plt.subplots(1, 2, figsize=(15, 6))

    # Top 8 contribuições para o caso de risco
    risk_contributions = pd.DataFrame({
        "Feature": feature_names_clean,
        "SHAP": shap_values_target[risk_idx]
    }).sort_values(by="SHAP", key=abs, ascending=False).head(8)

    colors_r = ["#2ecc71" if val > 0 else "#e74c3c" for val in risk_contributions["SHAP"]]
    ax_r.barh(risk_contributions["Feature"], risk_contributions["SHAP"], color=colors_r, edgecolor="black")
    ax_r.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax_r.set_xlabel("Contribuição SHAP para a Predição", fontweight="bold")
    ax_r.set_title("Exemplo Local: Município em ALTO RISCO (Target = 0)\nFatores negativos empurram para risco de não atingimento", fontsize=11, fontweight="bold")
    ax_r.invert_yaxis()

    # Top 8 contribuições para o caso de sucesso
    meta_contributions = pd.DataFrame({
        "Feature": feature_names_clean,
        "SHAP": shap_values_target[meta_idx]
    }).sort_values(by="SHAP", key=abs, ascending=False).head(8)

    colors_m = ["#2ecc71" if val > 0 else "#e74c3c" for val in meta_contributions["SHAP"]]
    ax_m.barh(meta_contributions["Feature"], meta_contributions["SHAP"], color=colors_m, edgecolor="black")
    ax_m.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax_m.set_xlabel("Contribuição SHAP para a Predição", fontweight="bold")
    ax_m.set_title("Exemplo Local: Município com META ATINGIDA (Target = 1)\nFatores positivos sustentam a probabilidade de sucesso", fontsize=11, fontweight="bold")
    ax_m.invert_yaxis()

    plt.suptitle("Explicação Local de Decisões Individuais via SHAP Values", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    local_path = IMAGES_DIR / "09_shap_local_waterfall.png"
    plt.savefig(local_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {local_path.name}")

    # 5. Tabela de Importância SHAP consolidada
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names_clean,
        "Nome_Original": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    importance_path = REPORTS_DIR / "shap_feature_importance.csv"
    df_importance.to_csv(importance_path, index=False)
    print(f"💾 Tabela de importância salva em: {importance_path}")

    print("\n" + "=" * 75)
    print("🏆 TOP 10 FATORES DETERMINANTES NA ALFABETIZAÇÃO (SHAP):")
    print("=" * 75)
    print(df_importance[["Feature", "Mean_Abs_SHAP"]].head(10).to_markdown(index=False))

    # 6. Coeficientes do Modelo Campeão (Regressão Logística) — Triangulação
    best_model_path = MODELS_DIR / "best_model_pipeline.pkl"
    if best_model_path.exists():
        best_clf = joblib.load(best_model_path)
        best_estimator = best_clf.named_steps["classifier"]
        if hasattr(best_estimator, "coef_"):
            print("\n🔬 Gerando Triangulação: Coeficientes Logísticos vs SHAP...")
            plot_logistic_coefficients(best_clf, feature_names, feature_names_clean)

    return df_importance


def plot_logistic_coefficients(
    pipeline,
    feature_names: np.ndarray,
    feature_names_clean: np.ndarray,
    output_name: str = "10_logistic_coefficients.png",
):
    """
    Extrai e plota os coeficientes padronizados da Regressão Logística campeã,
    permitindo triangulação com os SHAP Values do Random Forest.
    Coeficientes positivos indicam associação com Meta Atingida (Classe 1).
    Coeficientes negativos indicam associação com Risco (Classe 0).
    """
    estimator = pipeline.named_steps["classifier"]
    coefs = estimator.coef_[0]

    # Alinhar coeficientes com nomes de features
    n_features = min(len(coefs), len(feature_names_clean))
    df_coef = pd.DataFrame({
        "Feature": feature_names_clean[:n_features],
        "Nome_Original": feature_names[:n_features],
        "Coeficiente": coefs[:n_features],
        "Abs_Coef": np.abs(coefs[:n_features]),
    }).sort_values("Abs_Coef", ascending=False)

    top_n = min(12, len(df_coef))
    df_top = df_coef.head(top_n).sort_values("Coeficiente")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in df_top["Coeficiente"]]
    ax.barh(df_top["Feature"], df_top["Coeficiente"], color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_xlabel("Coeficiente Padronizado (Impacto no Log-Odds de Meta Atingida)", fontweight="bold")
    ax.set_title(
        "Coeficientes da Regressão Logística Campeã (Triangulação com SHAP)",
        fontsize=13, fontweight="bold", pad=15
    )

    # Legenda de interpretação
    ax.text(
        0.98, 0.02,
        "[+] Positivo (Protege Meta)  |  [-] Negativo (Aumenta Risco)",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9)
    )

    plt.tight_layout()
    coef_path = IMAGES_DIR / output_name
    plt.savefig(coef_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {coef_path.name}")

    # Salvar tabela de coeficientes
    coef_csv_path = REPORTS_DIR / "logistic_coefficients.csv"
    df_coef.to_csv(coef_csv_path, index=False)
    print(f"💾 Tabela de coeficientes salva em: {coef_csv_path}")

    print("\n" + "=" * 75)
    print("🔬 TRIANGULAÇÃO: TOP 10 COEFICIENTES LOGÍSTICOS vs SHAP:")
    print("=" * 75)
    print(df_coef[["Feature", "Coeficiente", "Abs_Coef"]].head(10).to_markdown(index=False))


if __name__ == "__main__":
    evaluate_and_explain()
