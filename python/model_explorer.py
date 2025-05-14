from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import polars as pd
import seaborn as sns

from model_handler import run_model
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

from tabnet_model import run_tabnet_model


# Logistic Regression with SMOTE
def logistic_regression(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    params_lr = {"penalty": "l1", "solver": "liblinear"}  # "class_weight": "balanced"}

    model_lrsmt = LogisticRegression(**params_lr)
    (
        model_lrsmt,
        accuracy_lrsmt,
        roc_auc_lrsmt,
        f1_score_lrsmt,
        coh_kap_lrsmt,
        tt_lrsmt,
    ) = run_model(
        "logistic_regression",
        model_lrsmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        show_graph=show_graphs,
        save_graph=save_graphs,
    )
    return accuracy_lrsmt, roc_auc_lrsmt, f1_score_lrsmt, coh_kap_lrsmt, tt_lrsmt


# Decision Tree
def decision_tree(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    params_dt = {"max_depth": 16, "max_features": "sqrt"}

    model_dtsmt = DecisionTreeClassifier(**params_dt)
    (
        model_dtsmt,
        accuracy_dtsmt,
        roc_auc_dtsmt,
        f1_score_dtsmt,
        coh_kap_dtsmt,
        tt_dtsmt,
    ) = run_model(
        "decision_tree",
        model_dtsmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        show_graph=show_graphs,
        save_graph=save_graphs,
    )
    return accuracy_dtsmt, roc_auc_dtsmt, f1_score_dtsmt, coh_kap_dtsmt, tt_dtsmt


def random_forest(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    # Random Forest with SMOTE
    model_rfsmt = RandomForestClassifier()
    (
        model_rfsmt,
        accuracy_rfsmt,
        roc_auc_rfsmt,
        f1_score_rfsmt,
        coh_kap_rfsmt,
        tt_rfsmt,
    ) = run_model(
        "random_forest",
        model_rfsmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        show_graph=show_graphs,
        save_graph=save_graphs,
    )
    return accuracy_rfsmt, roc_auc_rfsmt, f1_score_rfsmt, coh_kap_rfsmt, tt_rfsmt


def lightGBM(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    # Light GBM with SMOTE

    model_lgbsmt = lgb.LGBMClassifier()
    (
        model_lgbsmt,
        accuracy_lgbsmt,
        roc_auc_lgbsmt,
        f1_score_lgbsmt,
        coh_kap_lgbsmt,
        tt_lgbsmt,
    ) = run_model(
        "lightGBM",
        model_lgbsmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        show_graph=show_graphs,
        save_graph=save_graphs,
    )
    return accuracy_lgbsmt, roc_auc_lgbsmt, f1_score_lgbsmt, coh_kap_lgbsmt, tt_lgbsmt


def catboost(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    params_cb = {"iterations": 20, "max_depth": 16}

    model_cbsmt = cb.CatBoostClassifier(**params_cb)
    (
        model_cbsmt,
        accuracy_cbsmt,
        roc_auc_cbsmt,
        f1_score_cbsmt,
        coh_kap_cbsmt,
        tt_cbsmt,
    ) = run_model(
        "catboost",
        model_cbsmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        show_graph=show_graphs,
        save_graph=save_graphs,
    )
    return accuracy_cbsmt, roc_auc_cbsmt, f1_score_cbsmt, coh_kap_cbsmt, tt_cbsmt


def xgboost(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    params_xgb = {"n_estimators": 20, "max_depth": 16}

    model_xgbsmt = xgb.XGBClassifier(**params_xgb)
    (
        model_xgbsmt,
        accuracy_xgbsmt,
        roc_auc_xgbsmt,
        f1_score_xgbsmt,
        coh_kap_xgbsmt,
        tt_xgbsmt,
    ) = run_model(
        "xgboost",
        model_xgbsmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        show_graph=show_graphs,
        save_graph=save_graphs,
    )
    return accuracy_xgbsmt, roc_auc_xgbsmt, f1_score_xgbsmt, coh_kap_xgbsmt, tt_xgbsmt


def adaboost(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    model_adasmt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=50,
        algorithm="SAMME",
        learning_rate=0.5,
    )
    (
        model_adasmt,
        accuracy_adasmt,
        roc_auc_adasmt,
        f1_score_adasmt,
        coh_kap_adasmt,
        tt_adasmt,
    ) = run_model(
        "adaboost",
        model_adasmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        show_graph=show_graphs,
        save_graph=save_graphs,
    )
    return accuracy_adasmt, roc_auc_adasmt, f1_score_adasmt, coh_kap_adasmt, tt_adasmt


def tabnet(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
) -> (float, float, float, float, float):
    tabnet_params = dict(
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
    model_tabnetsmt = TabNetClassifier(**tabnet_params)
    (
        model_tabnetsmt,
        accuracy_tabnetsmt,
        roc_auc_tabnetsmt,
        f1_score_tabnetsmt,
        coh_kap_tabnetsmt,
        tt_tabnetsmt,
    ) = run_tabnet_model(
        model_tabnetsmt,
        x_train_smt,
        y_train_smt,
        x_test,
        y_test,
        save_graph=save_graphs,
        show_graph=show_graphs,
    )
    return (
        accuracy_tabnetsmt,
        roc_auc_tabnetsmt,
        f1_score_tabnetsmt,
        coh_kap_tabnetsmt,
        tt_tabnetsmt,
    )


def visualize_model_accuracy_and_time(
    model_data, show_graph: bool = True, save_graph: bool = False
):
    data = pd.DataFrame(model_data)

    fig, ax1 = plt.subplots(figsize=(12, 10))
    ax1.set_title(
        "Model Comparison: Accuracy and Time taken for execution", fontsize=13
    )
    color = "tab:green"
    ax1.set_xlabel("Model", fontsize=13)
    ax1.set_ylabel("Time taken", fontsize=13, color=color)
    ax2 = sns.barplot(
        x="Model",
        y="Time taken",
        data=data,
        palette="summer",
        hue="Model",
        legend=False,
    )
    ax1.tick_params(axis="y")
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("F1 Score", fontsize=13, color=color)
    ax2 = sns.lineplot(x="Model", y="F1 Score", data=data, sort=False, color=color)
    ax2.tick_params(axis="y", color=color)
    if save_graph:
        plt.savefig("images/models_f1_and_time.png", transparent=True)
    if show_graph:
        plt.show()


def explore_models(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
    show_graphs: bool = False,
    save_graphs: bool = False,
):
    (
        accuracy_tabnetsmt,
        roc_auc_tabnetsmt,
        f1_score_tabnetsmt,
        coh_kap_tabnetsmt,
        tt_tabnetsmt,
    ) = tabnet(x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs)

    accuracy_lrsmt, roc_auc_lrsmt, f1_score_lrsmt, coh_kap_lrsmt, tt_lrsmt = (
        logistic_regression(
            x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs
        )
    )
    accuracy_dtsmt, roc_auc_dtsmt, f1_score_dtsmt, coh_kap_dtsmt, tt_dtsmt = (
        decision_tree(
            x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs
        )
    )
    accuracy_rfsmt, roc_auc_rfsmt, f1_score_rfsmt, coh_kap_rfsmt, tt_rfsmt = (
        random_forest(
            x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs
        )
    )

    accuracy_lgbsmt, roc_auc_lgbsmt, f1_score_lgbsmt, coh_kap_lgbsmt, tt_lgbsmt = (
        lightGBM(x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs)
    )

    accuracy_cbsmt, roc_auc_cbsmt, f1_score_cbsmt, coh_kap_cbsmt, tt_cbsmt = catboost(
        x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs
    )

    accuracy_xgbsmt, roc_auc_xgbsmt, f1_score_xgbsmt, coh_kap_xgbsmt, tt_xgbsmt = (
        xgboost(x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs)
    )

    accuracy_adasmt, roc_auc_adasmt, f1_score_adasmt, coh_kap_adasmt, tt_adasmt = (
        adaboost(x_train_smt, y_train_smt, x_test, y_test, show_graphs, save_graphs)
    )

    plot_spider_chart(
        "logistic_regression",
        [accuracy_lrsmt, roc_auc_lrsmt, f1_score_lrsmt, coh_kap_lrsmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )
    plot_spider_chart(
        "decision_tree",
        [accuracy_dtsmt, roc_auc_dtsmt, f1_score_dtsmt, coh_kap_dtsmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )
    plot_spider_chart(
        "random_forest",
        [accuracy_rfsmt, roc_auc_rfsmt, f1_score_rfsmt, coh_kap_rfsmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )
    plot_spider_chart(
        "lightGBM",
        [accuracy_lgbsmt, roc_auc_lgbsmt, f1_score_lgbsmt, coh_kap_lgbsmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )

    plot_spider_chart(
        "catboost",
        [accuracy_cbsmt, roc_auc_cbsmt, f1_score_cbsmt, coh_kap_cbsmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )

    plot_spider_chart(
        "xgboost",
        [accuracy_xgbsmt, roc_auc_xgbsmt, f1_score_xgbsmt, coh_kap_xgbsmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )

    plot_spider_chart(
        "adaboost",
        [accuracy_adasmt, roc_auc_adasmt, f1_score_adasmt, coh_kap_adasmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )

    plot_spider_chart(
        "tabnet",
        [accuracy_tabnetsmt, roc_auc_tabnetsmt, f1_score_tabnetsmt, coh_kap_tabnetsmt],
        show_graphs=show_graphs,
        save_graphs=save_graphs,
    )

    accuracy_scores = [
        accuracy_lrsmt,
        accuracy_dtsmt,
        accuracy_rfsmt,
        accuracy_lgbsmt,
        accuracy_cbsmt,
        accuracy_xgbsmt,
        accuracy_adasmt,
        accuracy_tabnetsmt,
    ]
    roc_auc_scores = [
        roc_auc_lrsmt,
        roc_auc_dtsmt,
        roc_auc_rfsmt,
        roc_auc_lgbsmt,
        roc_auc_cbsmt,
        roc_auc_xgbsmt,
        roc_auc_adasmt,
        roc_auc_tabnetsmt,
    ]
    f1_scores = [
        f1_score_lrsmt,
        f1_score_dtsmt,
        f1_score_rfsmt,
        f1_score_lgbsmt,
        f1_score_cbsmt,
        f1_score_xgbsmt,
        f1_score_adasmt,
        f1_score_tabnetsmt,
    ]
    coh_kap_scores = [
        coh_kap_lrsmt,
        coh_kap_dtsmt,
        coh_kap_rfsmt,
        coh_kap_lgbsmt,
        coh_kap_cbsmt,
        coh_kap_xgbsmt,
        coh_kap_adasmt,
        coh_kap_tabnetsmt,
    ]
    tt = [
        tt_lrsmt,
        tt_dtsmt,
        tt_rfsmt,
        tt_lgbsmt,
        tt_cbsmt,
        tt_xgbsmt,
        tt_adasmt,
        tt_tabnetsmt,
    ]

    model_data = {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "LightGBM",
            "Catboost",
            "XGBoost",
            "AdaBoost",
            "TabNet",
        ],
        "Accuracy": accuracy_scores,
        "ROC_AUC": roc_auc_scores,
        "F1 Score": f1_scores,
        "Cohen_Kappa": coh_kap_scores,
        "Time taken": tt,
    }
    visualize_model_accuracy_and_time(
        model_data, show_graph=show_graphs, save_graph=save_graphs
    )


def plot_spider_chart(
    model_name: str,
    values: list[float],
    show_graphs: bool = False,
    save_graphs: bool = False,
):
    metrics = ["Accuracy", "ROC AUC", "F1 Score", "Cohen Kappa"]
    num_vars = len(values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]  # Close the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="blue", alpha=0.25)
    ax.plot(angles, values, color="blue", marker="o")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title("Model Performance Radar Chart", size=14, pad=20)
    if save_graphs:
        plt.savefig(f"images/{model_name}_spider_chart", transparent=True)
    if show_graphs:
        plt.show()
