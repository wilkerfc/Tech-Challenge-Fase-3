from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_charts(modeling: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame, clusters: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    ranked = modeling.sort_values("gap_2023_para_meta_2024")
    colors = ["#c0392b" if value < 0 else "#1b7f5a" for value in ranked["gap_2023_para_meta_2024"]]
    plt.figure(figsize=(11, 7)); plt.barh(ranked["sigla_uf"], ranked["gap_2023_para_meta_2024"], color=colors)
    plt.axvline(0, color="black", linewidth=1); plt.xlabel("Taxa de 2023 - meta oficial de 2024 (p.p.)"); plt.ylabel("UF")
    plt.title("Distância inicial para a meta de alfabetização de 2024"); plt.tight_layout(); plt.savefig(out / "gap_2023_meta_2024.png", dpi=180); plt.close()
    risk = predictions.sort_values("gap_observado_2024")
    plt.figure(figsize=(11, 7)); plt.barh(risk["sigla_uf"], risk["gap_observado_2024"], color="#d97706")
    plt.axvline(0, color="black", linewidth=1); plt.xlabel("Taxa observada - meta oficial (p.p.)"); plt.ylabel("UF")
    plt.title("Gap observado para a meta de 2024 (não é previsão)")
    plt.tight_layout(); plt.savefig(out / "gap_observado_2024.png", dpi=180); plt.close()
    plt.figure(figsize=(8, 5)); sns.barplot(data=importance, x="importancia_modelo", y="variavel", color="#176b87")
    plt.title("Influência relativa das variáveis no modelo selecionado"); plt.tight_layout(); plt.savefig(out / "importancia_variaveis.png", dpi=180); plt.close()
    plt.figure(figsize=(9, 6)); sns.scatterplot(data=clusters, x="media_portugues", y="taxa_alfabetizacao", hue="cluster", palette="tab10", s=90)
    for row in clusters.itertuples(): plt.text(row.media_portugues + .15, row.taxa_alfabetizacao + .15, row.sigla_uf, fontsize=8)
    plt.title("Agrupamentos de UFs por desempenho em 2024"); plt.xlabel("Média de Língua Portuguesa"); plt.ylabel("Taxa de alfabetização (%)")
    plt.tight_layout(); plt.savefig(out / "clusters_ufs.png", dpi=180); plt.close()
