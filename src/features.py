from __future__ import annotations

from typing import Iterable
import pandas as pd

from dataset import load_data


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    #데이터 프레임 feature vector로 변환
    agg = (
        df.groupby(["category", "data_id"])
        .agg(
            x_mean=("x", "mean"),
            x_std=("x", "std"),
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_mean=("y", "mean"),
            y_std=("y", "std"),
            y_min=("y", "min"),
            y_max=("y", "max"),
            z_mean=("z", "mean"),
            z_std=("z", "std"),
            z_min=("z", "min"),
            z_max=("z", "max"),
            length=("time_step", "count"),
        )
        .reset_index()
    )

    # 라벨, 피처 분리
    y = agg["category"]
    X = agg.drop(columns=["category", "data_id"])

    return X, y


def load_folders_and_build_features(folders: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    dfs = []
    for folder in folders:
        df = load_data(folder)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    X, y = build_feature_table(merged)
    return X, y