from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_charts(df: pd.DataFrame, importance: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4)); sns.countplot(data=df, x="alfabetizado_binario")
    plt.xticks([0, 1], ["Não alfabetizado", "Alfabetizado"]); plt.title("Distribuição do alvo")
    plt.tight_layout(); plt.savefig(out / "distribuicao_alvo.png", dpi=160); plt.close()
    plt.figure(figsize=(9, 5)); top = importance.head(15).sort_values("importancia")
    sns.barplot(data=top, x="importancia", y="variavel", color="#176B87")
    plt.title("Importância de variáveis - Random Forest"); plt.tight_layout()
    plt.savefig(out / "importancia_variaveis.png", dpi=160); plt.close()
