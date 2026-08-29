"""
Módulo de Visualização de Dados e Geração de Gráficos Executivos
Gera figuras de alta resolução (300 DPI) para relatórios, notebooks e README
utilizando caminhos relativos ao projeto para portabilidade total.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(y: pd.Series, output_name: str = "01_target_distribution.png"):
    """Gera gráfico de distribuição da variável alvo."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    labels = ["Meta Não Atingida (0)", "Meta Atingida (1)"]
    colors = ["#e74c3c", "#2ecc71"]
    
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Quantidade de Municípios / Observações", fontweight="bold")
    ax.set_title("Distribuição da Variável Alvo (Atingimento da Meta de Alfabetização)", fontsize=13, fontweight="bold", pad=15)
    
    total = len(y)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_correlation_matrix(df: pd.DataFrame, output_name: str = "02_correlation_matrix.png"):
    """Gera matriz de correlação das variáveis socioeconômicas e históricas."""
    numeric_cols = [
        "indicador_lag1", "indicador_lag2", "tendencia_historica",
        "gap_historico_vs_meta_municipio", "IDHM", "PIB_per_capita",
        "quantidade_matriculas", "target_meta_atingida"
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()

    labels_map = {
        "indicador_lag1": "Ind. Lag 1 (t-1)",
        "indicador_lag2": "Ind. Lag 2 (t-2)",
        "tendencia_historica": "Tendência Hist.",
        "gap_historico_vs_meta_municipio": "Gap vs Meta",
        "IDHM": "IDHM",
        "PIB_per_capita": "PIB per capita",
        "quantidade_matriculas": "Matrículas",
        "target_meta_atingida": "Target (Meta Atingida)"
    }
    corr = corr.rename(index=labels_map, columns=labels_map)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Matriz de Correlação Linear (Features vs Target)", fontsize=13, fontweight="bold", pad=15)
    
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_regional_performance(df: pd.DataFrame, output_name: str = "03_regional_performance.png"):
    """Gera comparativo regional de taxa de alfabetização e atingimento da meta."""
    if "regiao" not in df.columns:
        return
    
    reg_summary = df.groupby("regiao").agg(
        taxa_meta=("target_meta_atingida", "mean"),
        ind_medio=("indicador_lag1", "mean"),
        idhm_medio=("IDHM", "mean"),
        total=("id_municipio", "count")
    ).reset_index()
    reg_summary["taxa_meta_pct"] = reg_summary["taxa_meta"] * 100

    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    sns.barplot(data=reg_summary, x="regiao", y="taxa_meta_pct", palette="Blues_d", ax=ax1, edgecolor="black")
    ax1.set_ylabel("% Municípios com Meta Atingida", fontweight="bold", color="#1f77b4")
    ax1.set_xlabel("Grande Região do Brasil", fontweight="bold")
    ax1.set_title("Desempenho Educacional por Região Brasileira", fontsize=13, fontweight="bold", pad=15)
    ax1.set_ylim(0, 100)

    for i, row in reg_summary.iterrows():
        ax1.text(i, row["taxa_meta_pct"] + 2, f"{row['taxa_meta_pct']:.1f}%", ha="center", fontweight="bold")

    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_roc_curves(models_dict: dict, X_test, y_test, output_name: str = "04_roc_curves_comparison.png"):
    """Plota comparação de curvas ROC para múltiplos modelos no holdout de teste."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    palette = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=palette[idx % len(palette)], lw=2.5)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Aleatório (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontweight="bold")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade / Recall)", fontweight="bold")
    ax.set_title("Curvas ROC Comparativas no Holdout Temporal (Ano 2024)", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True)
    
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_confusion_matrix_heatmap(y_test, y_pred, model_name: str = "Random Forest", output_name: str = "05_confusion_matrix.png"):
    """Gera matriz de confusão anotada com destaque para o Recall da classe minoritária (Risco)."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    annot = np.empty_like(cm).astype(str)
    
    class_labels = ["Risco (0)", "Meta Atingida (1)"]
    
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=class_labels,
                yticklabels=class_labels, ax=ax)
    
    ax.set_xlabel("Predição do Modelo", fontweight="bold")
    ax.set_ylabel("Valor Real no Ano 2024 (Ground Truth)", fontweight="bold")
    
    recall_0 = cm_norm[0, 0]
    recall_1 = cm_norm[1, 1]
    bal_acc = (recall_0 + recall_1) / 2
    
    ax.set_title(f"Matriz de Confusão - {model_name}\nRecall Risco: {recall_0:.1%} | Balanced Acc: {bal_acc:.1%}", fontsize=12, fontweight="bold", pad=15)
    
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_risk_quadrant(df: pd.DataFrame, output_name: str = "06_risk_quadrant.png"):
    """Plota quadrante de vulnerabilidade: Indicador t-1 vs Tendência Histórica."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sample = df.sample(n=min(2500, len(df)), random_state=42)
    scatter = ax.scatter(
        sample["indicador_lag1"],
        sample["tendencia_historica"],
        c=sample["IDHM"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="none",
        s=35
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("IDHM Municipal", fontweight="bold")
    
    # Linhas de quadrante
    ax.axvline(x=50.0, color="red", linestyle="--", alpha=0.7, label="Corte Crítico (50%)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.7, label="Estagnação (Tendência = 0)")
    
    ax.set_xlabel("Indicador de Alfabetização no Ano Anterior (%)", fontweight="bold")
    ax.set_ylabel("Tendência Histórica de Variação (p.p.)", fontweight="bold")
    ax.set_title("Quadrante de Risco Educacional Municipal (Vulnerabilidade vs Evolução)", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper left")
    
    # Anotações dos quadrantes
    ax.text(20, -8, "[ALTO RISCO]\nBaixo Ind. + Queda", color="darkred", fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    ax.text(75, 8, "[ALTO DESEMPENHO]\nAlto Ind. + Crescimento", color="darkgreen", fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_threshold_tuning(y_true, y_proba, output_name: str = "08_threshold_tuning.png"):
    """
    Plota curvas de Precisão, Recall e F1-Score em função do limiar de decisão
    para detecção de municípios em risco (Classe 0).
    Permite justificar matematicamente a escolha do threshold de probabilidade.
    """
    # Converter para predição da classe 0 (Risco)
    y_true_risk = (y_true == 0).astype(int)
    y_proba_risk = 1.0 - y_proba  # P(Risco) = 1 - P(Meta Atingida)

    thresholds = np.linspace(0.05, 0.95, 100)
    precisions = []
    recalls = []
    f1_scores = []

    for t in thresholds:
        y_pred_t = (y_proba_risk >= t).astype(int)
        tp = np.sum((y_true_risk == 1) & (y_pred_t == 1))
        fp = np.sum((y_true_risk == 0) & (y_pred_t == 1))
        fn = np.sum((y_true_risk == 1) & (y_pred_t == 0))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    optimal_t = thresholds[best_idx]
    optimal_f1 = f1_scores[best_idx]
    optimal_recall = recalls[best_idx]
    optimal_prec = precisions[best_idx]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(thresholds, recalls, label="Recall (Sensibilidade Risco)", color="#e74c3c", lw=2.5)
    ax.plot(thresholds, precisions, label="Precisão (Risco)", color="#2980b9", lw=2.5)
    ax.plot(thresholds, f1_scores, label="F1-Score (Risco)", color="#27ae60", lw=2.5)

    ax.axvline(optimal_t, color="purple", linestyle="--", lw=1.8,
               label=f"Limiar Ótimo F1 = {optimal_t:.2f} (Recall: {optimal_recall:.1%})")
    ax.axvline(0.40, color="orange", linestyle=":", lw=1.8,
               label="Limiar Conservador MEC = 0.40")

    ax.set_xlabel("Limiar de Probabilidade de Risco $\\hat{P}(\\text{Risco})$", fontweight="bold")
    ax.set_ylabel("Métrica de Avaliação", fontweight="bold")
    ax.set_title("Calibração de Limiar de Decisão para Alerta Precoce (Políticas Públicas)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower center", frameon=True)

    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name} | Limiar ótimo: {optimal_t:.2f} (F1: {optimal_f1:.4f})")


def plot_unsupervised_clusters(df: pd.DataFrame, output_name: str = "09_unsupervised_clusters.png"):
    """
    Realiza agrupamento não-supervisionado (K-Means) sobre os municípios para
    identificar perfis territoriais e socioeconômicos de risco sem depender
    exclusivamente das fronteiras geográficas tradicionais.
    """
    cluster_features = [
        "indicador_lag1",
        "IDHM",
        "PIB_per_capita",
        "gap_historico_vs_meta_municipio",
        "tendencia_historica"
    ]
    df_clean = df[cluster_features].dropna().copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean["Cluster"] = clusters

    # Mapear perfis descritivos
    cluster_profiles = {
        0: "Perfil 1: Consolidado Alto IDHM",
        1: "Perfil 2: Alta Vulnerabilidade / Baixo IDHM",
        2: "Perfil 3: Em Rápida Aceleração",
        3: "Perfil 4: Estável Médio Porte",
    }
    df_clean["Perfil"] = df_clean["Cluster"].map(cluster_profiles)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = ["#27ae60", "#e74c3c", "#3498db", "#f39c12"]
    
    sample = df_clean.sample(n=min(3000, len(df_clean)), random_state=42)
    sns.scatterplot(
        data=sample,
        x="IDHM",
        y="indicador_lag1",
        hue="Cluster",
        palette=palette,
        alpha=0.65,
        s=40,
        ax=ax
    )
    
    ax.set_xlabel("Índice de Desenvolvimento Humano Municipal (IDHM)", fontweight="bold")
    ax.set_ylabel("Indicador de Alfabetização Histórico (%)", fontweight="bold")
    ax.set_title("Agrupamento Não-Supervisionado de Municípios (K-Means, k=4)", fontsize=13, fontweight="bold", pad=15)
    
    # Adicionar legenda customizada
    handles, _ = ax.get_legend_handles_labels()
    labels = ["Cluster 0: Alta Proficiência & Alto IDHM",
              "Cluster 1: Vulnerabilidade Crítica (Prioridade MEC)",
              "Cluster 2: Em Recuperação & Expansão",
              "Cluster 3: Nível Médio com Dispersão"]
    ax.legend(handles=handles, labels=labels, loc="lower right", frameon=True)

    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")
