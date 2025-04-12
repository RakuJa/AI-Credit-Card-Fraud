from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import polars as pd
import seaborn as sns

from model_handler import run_model
import lightgbm as lgb


# Logistic Regression with SMOTE
def logistic_regression(x_train_smt, y_train_smt, x_test, y_test):
    params_lr = {"penalty": "l1", "solver": "liblinear"}  # "class_weight": "balanced"}

    model_lrsmt = LogisticRegression(**params_lr)
    (
        model_lrsmt,
        accuracy_lrsmt,
        roc_auc_lrsmt,
        f1_score_lrsmt,
        coh_kap_lrsmt,
        tt_lrsmt,
    ) = run_model(model_lrsmt, x_train_smt, y_train_smt, x_test, y_test)
    return accuracy_lrsmt, roc_auc_lrsmt, f1_score_lrsmt, coh_kap_lrsmt, tt_lrsmt


# Decision Tree
def decision_tree(x_train_smt, y_train_smt, x_test, y_test):
    params_dt = {"max_depth": 16, "max_features": "sqrt"}

    model_dtsmt = DecisionTreeClassifier(**params_dt)
    (
        model_dtsmt,
        accuracy_dtsmt,
        roc_auc_dtsmt,
        f1_score_dtsmt,
        coh_kap_dtsmt,
        tt_dtsmt,
    ) = run_model(model_dtsmt, x_train_smt, y_train_smt, x_test, y_test)
    return accuracy_dtsmt, roc_auc_dtsmt, f1_score_dtsmt, coh_kap_dtsmt, tt_dtsmt


def random_forest(x_train_smt, y_train_smt, x_test, y_test):
    # Random Forest with SMOTE
    model_rfsmt = RandomForestClassifier()
    (
        model_rfsmt,
        accuracy_rfsmt,
        roc_auc_rfsmt,
        f1_score_rfsmt,
        coh_kap_rfsmt,
        tt_rfsmt,
    ) = run_model(model_rfsmt, x_train_smt, y_train_smt, x_test, y_test)
    return accuracy_rfsmt, roc_auc_rfsmt, f1_score_rfsmt, coh_kap_rfsmt, tt_rfsmt


def lightGBM(x_train_smt, y_train_smt, x_test, y_test):
    # Light GBM with SMOTE

    model_lgbsmt = lgb.LGBMClassifier()
    (
        model_lgbsmt,
        accuracy_lgbsmt,
        roc_auc_lgbsmt,
        f1_score_lgbsmt,
        coh_kap_lgbsmt,
        tt_lgbsmt,
    ) = run_model(model_lgbsmt, x_train_smt, y_train_smt, x_test, y_test)
    return accuracy_lgbsmt, roc_auc_lgbsmt, f1_score_lgbsmt, coh_kap_lgbsmt, tt_lgbsmt


def visualize_model_accuracy_and_time(model_data):
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
    plt.show()


def explore_models(x_train_smt, y_train_smt, x_test, y_test):
    accuracy_lrsmt, roc_auc_lrsmt, f1_score_lrsmt, coh_kap_lrsmt, tt_lrsmt = (
        logistic_regression(x_train_smt, y_train_smt, x_test, y_test)
    )
    accuracy_dtsmt, roc_auc_dtsmt, f1_score_dtsmt, coh_kap_dtsmt, tt_dtsmt = (
        decision_tree(x_train_smt, y_train_smt, x_test, y_test)
    )
    accuracy_rfsmt, roc_auc_rfsmt, f1_score_rfsmt, coh_kap_rfsmt, tt_rfsmt = (
        random_forest(x_train_smt, y_train_smt, x_test, y_test)
    )

    accuracy_lgbsmt, roc_auc_lgbsmt, f1_score_lgbsmt, coh_kap_lgbsmt, tt_lgbsmt = (
        lightGBM(x_train_smt, y_train_smt, x_test, y_test)
    )

    accuracy_scores = [accuracy_lrsmt, accuracy_dtsmt, accuracy_rfsmt, accuracy_lgbsmt]
    roc_auc_scores = [roc_auc_lrsmt, roc_auc_dtsmt, roc_auc_rfsmt, roc_auc_lgbsmt]
    f1_scores = [f1_score_lrsmt, f1_score_dtsmt, f1_score_rfsmt, f1_score_lgbsmt]
    coh_kap_scores = [coh_kap_lrsmt, coh_kap_dtsmt, coh_kap_rfsmt, coh_kap_lgbsmt]
    tt = [tt_lrsmt, tt_dtsmt, tt_rfsmt, tt_lgbsmt]

    model_data = {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "LightGBM",
            # "Catboost",
            # "XGBoost",
            # "AdaBoost",
            # "TabNet",
        ],
        "Accuracy": accuracy_scores,
        "ROC_AUC": roc_auc_scores,
        "F1 Score": f1_scores,
        "Cohen_Kappa": coh_kap_scores,
        "Time taken": tt,
    }
    visualize_model_accuracy_and_time(model_data)
