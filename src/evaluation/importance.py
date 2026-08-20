from __future__ import annotations

import pandas as pd


def feature_importance(pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    return pd.DataFrame({"variavel": names, "importancia": values}).sort_values("importancia", ascending=False)
