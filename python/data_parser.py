import copy
import pathlib
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
    StratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from model_explorer import visualize_models_results
import seaborn as sns

import pickle


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


def prepare_dataset_w_holdout(
    df: DataFrame, show_graphs: bool = False, save_graphs: bool = False
) -> Any:
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


def prepare_dataset_for_kfold(df: DataFrame) -> Any:
    df = _preprocess(df)
    x = df.drop(["class", "time"])
    y = df["class"]
    return x, y


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
    result_dicts = {
        "Hold-Out": {0.1: ([], []), 0.25: ([], []), 0.5: ([], [])},
        "KF": {0.1: ([], []), 0.25: ([], []), 0.5: ([], [])},
        "SKF": {0.1: ([], []), 0.25: ([], []), 0.5: ([], [])},
    }
    if pathlib.Path("run_data.pickle").exists():
        with open("run_data.pickle", "rb") as f:
            data = pickle.load(f)
    else:
        for smote_perc in [0.1, 0.25, 0.5]:
            for model in get_model_list():
                ((ho, ho_time), (kf5, kf5_time), (kf10, kf10_time)) = (
                    explore_validators(
                        x.values, y.values, model, smote_percentage=smote_perc
                    )
                )
                result_dicts = _update_result_dicts(
                    result_dicts, ("Hold-Out", smote_perc), ho_time, ho
                )
                result_dicts = _update_result_dicts(
                    result_dicts, ("KF", smote_perc), kf5_time, kf5
                )
                result_dicts = _update_result_dicts(
                    result_dicts, ("SKF", smote_perc), kf10_time, kf10
                )
            visualize_models_results(
                {
                    "Model": models,
                    "F1 Score": result_dicts.get("Hold-Out").get(smote_perc)[1],
                    "Time taken": result_dicts.get("Hold-Out").get(smote_perc)[0],
                },
                f"Hold-Out/SMOTE{smote_perc}-models_f1_and_time",
                show_graphs,
                save_graphs,
            )
            visualize_models_results(
                {
                    "Model": models,
                    "F1 Score": result_dicts.get("KF").get(smote_perc)[1],
                    "Time taken": result_dicts.get("KF").get(smote_perc)[0],
                },
                title=f"K-Fold/SMOTE{smote_perc}-models_f1_and_time",
                show_graph=show_graphs,
                save_graph=save_graphs,
            )

            visualize_models_results(
                {
                    "Model": models,
                    "F1 Score": result_dicts.get("SKF").get(smote_perc)[1],
                    "Time taken": result_dicts.get("SKF").get(smote_perc)[0],
                },
                title=f"Stratified-K-Fold/SMOTE{smote_perc}-models_f1_and_time",
                show_graph=show_graphs,
                save_graph=save_graphs,
            )
        data = pickle_rick(result_dicts)
    plot_validation_data(data, show_graphs, save_graphs)


def pickle_rick(x: dict[str, dict[float, list[tuple[float, float]]]]):
    data = []
    for val_method, inner_dict in x.items():
        for smote_perc, tpl in inner_dict.items():
            for tempo, f1 in zip(tpl[0], tpl[1]):
                data.append(
                    {
                        "method": val_method,
                        "smote": smote_perc,
                        "time": tempo,
                        "f1_score": f1,
                    }
                )

    with open("run_data.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def plot_validation_data(
    data: list[dict], show_graph: bool = False, save_graph: bool = True
):
    df = pd.from_dicts(data)

    palette = [
        "#735DEE",
        "#DE217D",
        "#FF5F01",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    sns.set_palette(palette)

    g = sns.FacetGrid(df, col="smote", hue="method", col_wrap=3, height=4, sharey=True)
    g.map(sns.lineplot, "time", "f1_score", estimator=None, sort=False, alpha=0.1)
    g.map(sns.scatterplot, "time", "f1_score", s=80, alpha=0.8)

    for ax in g.axes.flat:
        smote_method: str = ax.get_title()

        subplot_data = df.filter(pd.col("smote") == float(smote_method.split("=")[1]))

        for row in subplot_data.iter_rows():
            ax.text(
                x=row[2],
                y=row[3],
                s=f"{row[3]:.2f}",
                fontsize=8,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

    g.set(xscale="log")
    g.set_axis_labels("Time (log scale)", "F1 Score")
    g.add_legend(title="Method")
    g.set_titles(col_template="{col_name}")
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle("Time vs F1 Score with Method Trends by SMOTE")

    if show_graph:
        plt.show()
    if save_graph:
        plt.savefig(
            f"images/models/SMOTE_Validation_summary.png", dpi=1200, transparent=True
        )


def _update_result_dicts(x: dict, key: tuple[str, float], elapsed, f1: float) -> dict:
    x[key[0]][key[1]][0].append(elapsed)
    x[key[0]][key[1]][1].append(f1)
    return x


def explore_validators(
    x, y, model, smote_percentage: float = 0.1
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
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
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold_scores = cross_val_score(pipeline, x, y, cv=kfold, scoring="f1")
    kf_time_taken = time.time() - curr_time

    ## 3. K-Fold 10 Cross-Validation with SMOTE
    curr_time = time.time()
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skfold_scores = cross_val_score(pipeline, x, y, cv=skfold, scoring="f1")
    skf_time_taken = time.time() - curr_time

    ## 3. Leave-One-Out with SMOTE (careful with large datasets)
    # LOO is generally not recommended with SMOTE for very large datasets
    # due to computational cost, but here's how to do it properly:
    # loo = LeaveOneOut()
    # loo_scores = cross_val_score(pipeline, x, y, cv=loo, scoring='accuracy')

    # print(f"\nLOO CV with SMOTE - Mean Accuracy: {np.mean(loo_scores):.4f}")
    return (
        (holdout_score, ho_time_taken),
        (np.mean(kfold_scores), kf_time_taken),
        (np.mean(skfold_scores), skf_time_taken),
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
