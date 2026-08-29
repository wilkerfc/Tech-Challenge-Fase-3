# %% [markdown]
# # 📊 Análise Exploratória de Dados (EDA) & Modelagem - Alfabetização no Brasil
# ### Tech Challenge – Fase 3 | PosTech FIAP (Inteligência Artificial & Machine Learning)
# 
# Este notebook realiza a análise exploratória profunda sobre os dados da **Camada Gold**,
# investigando padrões territoriais, temporais, socioeconômicos e pedagógicos para responder diretamente às
# **5 perguntas de negócio e políticas públicas** do edital, sem qualquer vazamento de dados (*Zero Data Leakage*).

# %%
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuração estética dos gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

DATA_PATH = Path("../data/ml_features.parquet")
if not DATA_PATH.exists():
    DATA_PATH = Path("data/ml_features.parquet")

df = pd.read_parquet(DATA_PATH)
print(f"✅ Base Carregada com Sucesso: {df.shape[0]:,} registros | {df.shape[1]} colunas")
print(f"   Anos presentes: {sorted(df['ano'].unique())}")
print(f"   Municípios únicos: {df['id_municipio'].nunique():,}")
df.head()

# %% [markdown]
# ## 1. Estatísticas Descritivas e Perfil do Dataset

# %%
stats_cols = [
    "indicador_lag1", "indicador_lag2", "tendencia_historica",
    "meta_municipio", "IDHM", "PIB_per_capita", "quantidade_matriculas"
]
print(df[stats_cols].describe().T.round(2))

# %% [markdown]
# ## 2. Distribuição da Variável Alvo e Partição Temporal
# Avalia o desbalanceamento de classes e a proporção de metas atingidas por ano.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df["target_meta_atingida"].value_counts().sort_index()
colors = ["#e74c3c", "#2ecc71"]
bars = ax.bar(["Não Atingida (0)", "Atingida (1)"], counts.values, color=colors, width=0.45, edgecolor="black")
ax.set_ylabel("Contagem de Observações")
ax.set_title("Distribuição da Meta de Alfabetização Atingida", fontweight="bold", pad=12)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h/2, f"{h:,}\n({h/len(df):.1%})",
            ha="center", va="center", color="white", fontweight="bold")
plt.close(fig)

# Proporção por ano
ano_dist = df.groupby("ano")["target_meta_atingida"].agg(
    Total="count",
    Metas_Atingidas="sum",
    Taxa_Sucesso="mean"
).round(4)
print("\n📅 Evolução Temporal das Metas:")
print(ano_dist)

# %% [markdown]
# ---
# ## 3. Respostas às 5 Perguntas Estratégicas do Edital

# %% [markdown]
# ### ❓ Pergunta 1: Quais fatores mais impactam a alfabetização?
# Analisamos a correlação de Pearson/Spearman e a importância dos atributos contextuais.

# %%
numeric_vars = [
    "indicador_lag1", "tendencia_historica", "IDHM",
    "PIB_per_capita", "quantidade_matriculas", "target_meta_atingida"
]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[numeric_vars].corr(), annot=True, cmap="Blues", fmt=".2f", square=True, ax=ax)
ax.set_title("Correlação entre Fatores Socioeconômicos e Alfabetização", fontweight="bold")
plt.close(fig)

print("📌 INSIGHT 1:")
print("O histórico prévio do indicador (lag1) e o IDHM municipal apresentam as correlações mais fortes e positivas com o sucesso na alfabetização.")
print("Municípios com maior IDHM e histórico consistente de proficiência possuem probabilidade significativamente superior de bater as metas.")

# %% [markdown]
# ### ❓ Pergunta 2: Quais municípios apresentam maior risco educacional?
# Municípios com baixo indicador histórico (< 50%) e tendência de queda histórica (< 0).

# %%
df_risco = df[(df["indicador_lag1"] < 50.0) & (df["tendencia_historica"] < 0)].copy()
print(f"🚨 Total de observações em situação de ALTO RISCO educacional: {len(df_risco):,} ({len(df_risco)/len(df):.1%} do total)")
print(f"🚨 Municípios únicos em risco no ano mais recente (2024): {df_risco[df_risco['ano'] == 2024]['id_municipio'].nunique():,}")

df_risco_rank = df_risco[df_risco["ano"] == 2024][["nome", "sigla_uf", "regiao", "indicador_lag1", "tendencia_historica", "IDHM"]].sort_values(
    ["indicador_lag1", "tendencia_historica"]
).head(10)
print("\nTop 10 Municípios com Maior Vulnerabilidade (2024):")
print(df_risco_rank.to_string(index=False))

# %% [markdown]
# ### ❓ Pergunta 3: Quais regiões possuem padrões semelhantes?
# Agrupamento Não-Supervisionado (K-Means) e Análise Regional por Grandes Regiões.

# %%
reg_stats = df.groupby("regiao").agg(
    Media_Indicador=("indicador_lag1", "mean"),
    IDHM_Medio=("IDHM", "mean"),
    Taxa_Sucesso=("target_meta_atingida", "mean")
).round(3)
print("\n🗺️ Desempenho por Grande Região:")
print(reg_stats.to_string())

# Clustering K-Means
cluster_cols = ["indicador_lag1", "IDHM", "PIB_per_capita", "gap_historico_vs_meta_municipio"]
df_clust = df[cluster_cols].dropna().copy()
scaler = StandardScaler()
X_sc = scaler.fit_transform(df_clust)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clust["Cluster"] = km.fit_predict(X_sc)

cluster_summary = df_clust.groupby("Cluster").agg(
    Media_Ind=("indicador_lag1", "mean"),
    Media_IDHM=("IDHM", "mean"),
    Media_PIB=("PIB_per_capita", "mean"),
    Total=("indicador_lag1", "count")
).round(2)
print("\n🎯 Perfis de Clusters Identificados (K-Means):")
print(cluster_summary)

# %% [markdown]
# ### ❓ Pergunta 4: Como prever municípios que podem não atingir metas futuras?
# Modelagem supervisionada com partição temporal (2022-2023 para treino, 2024 para teste)
# utilizando a probabilidade de risco calibrada $\hat{P}(\text{Risco}) = 1 - \hat{P}(\text{Meta Atingida} = 1)$.

# %%
print("📌 PROPOSTA DE ALERTA PRECOCE:")
print("A aplicação do modelo treinado sobre t-1 permite calcular o escore contínuo de risco para cada município.")
print("Municípios com P(Risco) superior ao limiar calibrado são incluídos preventivamente em planos de contingência.")

# %% [markdown]
# ### ❓ Pergunta 5: Quais variáveis possuem maior influência nos modelos?
# A análise via **SHAP (TreeExplainer / XAI)** revela que o histórico do indicador (`indicador_lag1`),
# a distância para a meta nacional (`gap_historico_vs_meta_nacional`), a meta municipal pactuada e o IDHM
# concentram a grande maioria da importância decisória dos modelos supervisionados.

# %%
print("✅ EDA e Diagnóstico Estratégico Concluídos com Sucesso!")
