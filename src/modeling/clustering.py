from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cluster_ufs(frame: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    features = ["taxa_alfabetizacao", "media_portugues"]
    x = frame[features]
    best = None
    for k in range(2, 5):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KMeans(n_clusters=k, n_init=20, random_state=random_state)),
        ])
        labels = pipeline.fit_predict(x)
        score = float(silhouette_score(pipeline[:-1].transform(x), labels))
        if best is None or score > best[0]:
            best = (score, k, labels)
    output = frame[["sigla_uf", *features]].copy()
    output["cluster"] = best[2]
    return output.sort_values(["cluster", "taxa_alfabetizacao"], ascending=[True, False]), {
        "features": features, "selected_k": best[1], "silhouette": best[0]
    }
