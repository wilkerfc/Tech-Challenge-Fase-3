"""
Módulo de Pré-processamento e Pipelines de Machine Learning
Tech Challenge - Fase 3 | FIAP PosTech
"""

from src.preprocessing.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDE_COLUMNS,
    NUMERIC_FEATURES,
    get_preprocessor,
    load_and_split_data,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "EXCLUDE_COLUMNS",
    "NUMERIC_FEATURES",
    "get_preprocessor",
    "load_and_split_data",
]
