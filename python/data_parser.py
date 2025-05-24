import copy
import time
from typing import Any

import numpy as np
import polars as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from polars import DataFrame
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from collections import Counter
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from model_explorer import visualize_model_accuracy_and_time


def _preprocess(df: DataFrame) -> DataFrame:
    # Start by cleaning up, making everything lower case & converting data types
    df.columns = [col.lower() for col in df.columns]
    df.replace_column(
        df.get_column_index("class"), df.get_column("class").cast(pd.String)
    )
    # Scale amount by log
    df.insert_column(-1, pd.Series("amount_log", np.log(df["amount"] + 0.0001)))
    # Convert everything to float, otherwise we get errors with string - float operations
    df.replace_column(
        df.get_column_index("class"), df.get_column("class").cast(pd.Float64)
    )

    return df


def prepare_dataset(
    df: DataFrame, show_graphs: bool = False, save_graphs: bool = False
) -> Any:
    return _prepare_w_holdout(df, show_graphs, save_graphs)


def _prepare_w_holdout(
    df: DataFrame, show_graphs: bool = False, save_graphs: bool = False
):
    df = _preprocess(df)
    x = df.drop(["class", "time"])
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Original dataset shape %s" % Counter(y_train))
    smt = SMOTE(random_state=42, sampling_strategy=0.1)
    x_train_smt, y_train_smt = smt.fit_resample(x_train, y_train)
    print("Resampled dataset shape %s" % Counter(y_train_smt))

    barplot_data(y_train, y_train_smt, show_graph=show_graphs, save_graph=save_graphs)

    return df, x_train_smt, y_train_smt, x_test, y_test


def _prepare_w_k_fold(
    df: DataFrame, show_graphs: bool = False, save_graphs: bool = False
):
    df = _preprocess(df)
    x = df.drop(["class", "time"])
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Original dataset shape %s" % Counter(y_train))
    smt = SMOTE(random_state=42, sampling_strategy=0.1)
    x_train_smt, y_train_smt = smt.fit_resample(x_train, y_train)
    print("Resampled dataset shape %s" % Counter(y_train_smt))

    barplot_data(y_train, y_train_smt, show_graph=show_graphs, save_graph=save_graphs)

    return df, x_train_smt, y_train_smt, x_test, y_test


def barplot_data(
    y_train, y_train_smt, show_graph: bool = False, save_graph: bool = False
):
    plt.figure(figsize=(14, 6))
    class_colors = {0: "skyblue", 1: "salmon"}

    # First subplot - Original data
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(y_train, return_counts=True)
    bars = plt.bar(unique, counts)
    # Assign colors based on class
    for bar, cls in zip(bars, unique):
        bar.set_color(class_colors[cls])
    plt.title("Class Distribution - Original Data")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(unique)

    # Second subplot - SMOTE data
    plt.subplot(1, 2, 2)
    unique_smt, counts_smt = np.unique(y_train_smt, return_counts=True)
    bars = plt.bar(unique_smt, counts_smt)
    # Assign colors based on class
    for bar, cls in zip(bars, unique):
        bar.set_color(class_colors[cls])
    plt.title("Class Distribution - After SMOTE")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(unique_smt)

    plt.tight_layout()
    if show_graph:
        plt.show()
    if save_graph:
        plt.savefig(f"images/models/SMOTE_balanced_dataset.png", transparent=True)


def check_validation_method(
    df: DataFrame, show_graphs: bool = False, save_graphs: bool = False
):
    # Create pipeline with SMOTE and model
    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "LightGBM",
        "Catboost",
        "XGBoost",
        "AdaBoost",
        "TabNet",
    ]
    df = _preprocess(copy.deepcopy(df))
    x = df.drop(["class", "time"]).to_pandas()
    y = df["class"].to_pandas()
    for smote_perc in [0.1, 0.25, 0.5]:
        ho_f1_scores = []
        kf5_f1_scores = []
        kf10_f1_scores = []
        ho_time_taken = []
        kf5_time_taken = []
        kf10_time_taken = []
        for model in get_model_list():
            ((ho, ho_time), (kf5, kf5_time), (kf10, kf10_time)) = explore_validators(
                x.values, y.values, model, smote_percentage=smote_perc
            )
            ho_f1_scores.append(ho)
            ho_time_taken.append(ho_time)
            kf5_f1_scores.append(kf5)
            kf5_time_taken.append(kf5_time)
            kf10_f1_scores.append(kf10)
            kf10_time_taken.append(kf10_time)

        visualize_model_accuracy_and_time(
            {
                "Model": models,
                "F1 Score": ho_f1_scores,
                "Time taken": ho_time_taken,
            },
            f"Hold-Out/SMOTE{smote_perc}-models_f1_and_time",
            show_graphs,
            save_graphs,
        )
        visualize_model_accuracy_and_time(
            {
                "Model": models,
                "F1 Score": kf5_f1_scores,
                "Time taken": kf5_time_taken,
            },
            f"K-Fold-5/SMOTE{smote_perc}-models_f1_and_time",
            show_graphs,
            save_graphs,
        )

        visualize_model_accuracy_and_time(
            {
                "Model": models,
                "F1 Score": kf10_f1_scores,
                "Time taken": kf10_time_taken,
            },
            f"K-Fold-10/SMOTE{smote_perc}-models_f1_and_time",
            show_graphs,
            save_graphs,
        )


def explore_validators(
    x, y, model, smote_percentage: float = 0.1
) -> ((float, float), (float, float)):
    pipeline = make_pipeline(
        SMOTE(sampling_strategy=smote_percentage, random_state=42), model
    )
    ## 1. Hold-Out Validation with SMOTE
    curr_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    # SMOTE is automatically applied only to training data via pipeline
    pipeline.fit(x_train, y_train)
    holdout_score = f1_score(y_test, pipeline.predict(x_test))
    ho_time_taken = time.time() - curr_time
    ## 2. K-Fold 5 Cross-Validation with SMOTE
    curr_time = time.time()
    kfold5 = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold5_scores = cross_val_score(pipeline, x, y, cv=kfold5, scoring="f1")
    kf5_time_taken = time.time() - curr_time

    ## 3. K-Fold 10 Cross-Validation with SMOTE
    curr_time = time.time()
    kfold10 = KFold(n_splits=10, shuffle=True, random_state=42)
    kfold10_scores = cross_val_score(pipeline, x, y, cv=kfold10, scoring="f1")
    kf10_time_taken = time.time() - curr_time

    ## 3. Leave-One-Out with SMOTE (careful with large datasets)
    # LOO is generally not recommended with SMOTE for very large datasets
    # due to computational cost, but here's how to do it properly:
    # loo = LeaveOneOut()
    # loo_scores = cross_val_score(pipeline, x, y, cv=loo, scoring='accuracy')

    # print(f"\nLOO CV with SMOTE - Mean Accuracy: {np.mean(loo_scores):.4f}")
    return (
        (holdout_score, ho_time_taken),
        (np.mean(kfold5_scores), kf5_time_taken),
        (np.mean(kfold10_scores), kf10_time_taken),
    )


def get_model_list() -> list:
    return [
        LogisticRegression(**{"penalty": "l1", "solver": "liblinear"}),
        DecisionTreeClassifier(**{"max_depth": 16, "max_features": "sqrt"}),
        RandomForestClassifier(),
        LGBMClassifier(),
        CatBoostClassifier(**{"iterations": 20, "max_depth": 16}),
        XGBClassifier(**{"n_estimators": 20, "max_depth": 16}),
        AdaBoostClassifier(
            DecisionTreeClassifier(**{"max_depth": 16, "max_features": "sqrt"})
        ),
        TabNetClassifier(
            **dict(
                n_d=64,
                n_a=64,
                n_steps=5,
                gamma=1.5,
                n_independent=2,
                n_shared=2,
                cat_idxs=[],
                cat_dims=[],
                cat_emb_dim=1,
                lambda_sparse=1e-4,
                momentum=0.3,
                clip_value=2.0,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                scheduler_params=dict(mode="max", patience=5, min_lr=1e-5, factor=0.9),
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                mask_type="entmax",
                verbose=10,
            )
        ),
    ]
