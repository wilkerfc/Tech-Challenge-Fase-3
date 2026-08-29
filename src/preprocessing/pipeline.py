"""
Módulo de Pré-processamento e Construção de Pipelines Scikit-Learn
Implementa transformações robustas, tratamento de nulos, normalização e encoding
garantindo ZERO Data Leakage (direto e temporal) para o modelo de Machine Learning.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Definição estrita das features preditoras (sem vazamento da variável alvo)
NUMERIC_FEATURES = [
    "indicador_lag1",
    "indicador_lag2",
    "tendencia_historica",
    "gap_historico_vs_meta_municipio",
    "gap_historico_vs_meta_nacional",
    "meta_municipio",
    "meta_nacional",
    "quantidade_matriculas",
    "PIB_per_capita",
    "IDHM",
]

CATEGORICAL_FEATURES = [
    "sigla_uf",
    "regiao",
]

# Colunas identificadoras e targets a serem removidas da matriz X
EXCLUDE_COLUMNS = [
    "id_municipio",
    "nome",
    "id_uf",
    "ano",
    "indicador_alfabetizacao",
    "meta_atingida",
    "target_meta_atingida",
]


def get_preprocessor() -> ColumnTransformer:
    """
    Constrói o pipeline de pré-processamento Scikit-Learn.
    - Numéricas: Imputação pela mediana + Padronização (StandardScaler)
    - Categóricas: Imputação pelo mais frequente + OneHotEncoder
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def load_and_split_data(
    filepath: str | Path = "data/ml_features.parquet",
    target_col: str = "target_meta_atingida",
    temporal_split: bool = True,
    train_years: list[int] | None = None,
    test_year: int = 2024,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Carrega os dados e realiza a partição temporal estrita:
    - Treino: anos <= 2023 (11.140 instâncias)
    - Teste Holdout: ano == 2024 (5.570 instâncias)
    
    Retorna:
        X_train, y_train, groups_train, X_test, y_test, df_raw
    """
    if train_years is None:
        train_years = [2022, 2023]

    df = pd.read_parquet(filepath)
    
    # Validação do target
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no dataset.")
    
    cols_to_drop = [c for c in EXCLUDE_COLUMNS if c in df.columns]

    if temporal_split:
        train_mask = df["ano"].isin(train_years)
        test_mask = df["ano"] == test_year

        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        X_train = df_train.drop(columns=cols_to_drop)
        y_train = df_train[target_col].astype(int)
        groups_train = df_train["id_municipio"]

        X_test = df_test.drop(columns=cols_to_drop)
        y_test = df_test[target_col].astype(int)

        return X_train, y_train, groups_train, X_test, y_test, df
    else:
        X = df.drop(columns=cols_to_drop)
        y = df[target_col].astype(int)
        groups = df["id_municipio"]
        return X, y, groups, pd.DataFrame(), pd.Series(dtype=int), df


if __name__ == "__main__":
    X_train, y_train, groups_train, X_test, y_test, df = load_and_split_data()
    print("✅ Partição Temporal e Blindagem contra Leakage Concluídas:")
    print(f"   Treino (2022-2023): {X_train.shape[0]:,} amostras | Features: {X_train.shape[1]}")
    print(f"   Grupos únicos de treino: {groups_train.nunique():,} municípios")
    print(f"   Teste Holdout (2024): {X_test.shape[0]:,} amostras | Proporção Classe 1: {y_test.mean():.2%}")
