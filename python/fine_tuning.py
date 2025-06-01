import numpy as np
import polars as pd
import optuna  # pip install optuna
import xgboost as xgb
from plotly.graph_objs import Figure
from polars import DataFrame
import plotly.graph_objs as go
import plotly.subplots as tls
import plotly.io as pio
from sklearn.metrics import (
    make_scorer,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from model_handler import run_model


def objective(trial, x, y):
    param_grid = {
        # Tree params
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 5),
        # Learning
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        # Randomness & Regularization
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        # Imbalanced Data
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    model = xgb.XGBClassifier(**param_grid)
    model.fit(x, y)
    scores = cross_val_score(
        model,
        x,
        y,
        scoring=make_scorer(f1_score, average="weighted", labels=[1]),
        cv=cv,
        n_jobs=-1,
    )
    return np.mean(scores)


def execute(
    x_train_smt,
    y_train_smt,
    x_test,
    y_test,
):
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial, x_train_smt, y_train_smt)
    study.optimize(func, n_trials=100)

    print(f"\tBest value (Accuracy): {study.best_value:.5f}")
    print("\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")

    param_xgb = study.best_params

    xgb_opt = xgb.XGBClassifier(**param_xgb)
    run_model(
        model=xgb_opt,
        x_train=x_train_smt,
        y_train=y_train_smt,
        x_test=x_test,
        y_test=y_test,
        show_graph=False,
        save_graph=False,
        verbose=False,
        model_name="XGBoost",
    )
    return xgb_opt


pio.renderers.default = "colab"


def confusion_heatmap(conf_matrix):
    return go.Heatmap(
        z=conf_matrix,
        x=["0 (pred)", "1 (pred)"],
        y=["0 (true)", "1 (true)"],
        xgap=2,
        ygap=2,
        text=conf_matrix,
        colorscale="Viridis",
        showscale=False,
    )


def plot_metrics(show_metrics):
    colors = ["gold", "lightgreen", "lightcoral", "lightskyblue"]
    return go.Bar(
        x=show_metrics.to_numpy()[0],
        y=["Accuracy", "Precision", "Recall", "F1_score"],
        text=np.round(show_metrics.to_numpy()[0], 4),
        textposition="auto",
        orientation="h",
        opacity=0.8,
        marker=dict(color=colors, line=dict(color="#000000", width=1.5)),
    )


def plot_roc_curve(fpr, tpr, model_roc_auc):
    trace3 = go.Scatter(
        x=fpr,
        y=tpr,
        name="Roc : " + str(model_roc_auc),
        line=dict(color="rgb(22, 96, 167)", width=2),
        fill="tozeroy",
    )
    trace4 = go.Scatter(
        x=[0, 1], y=[0, 1], line=dict(color="black", width=1.5, dash="dot")
    )
    return trace3, trace4


def plot_precision_recall_curve(recall, precision):
    return go.Scatter(
        x=recall,
        y=precision,
        name="Precision" + str(precision),
        line=dict(color="lightcoral", width=2),
        fill="tozeroy",
    )


def plot_feature_importance(coef_sumry):
    return go.Bar(
        x=coef_sumry["features"].to_list(),
        y=coef_sumry["coefficients"].to_list(),
        name="coefficients",
        marker=dict(
            color=coef_sumry["coefficients"].to_list(),
            colorscale="Viridis",
            line=dict(width=0.6, color="black"),
        ),
    )


def plot_cumulative_gain(size, recall):
    return go.Scatter(
        x=size,
        y=recall,
        name="Lift curve",
        line=dict(color="gold", width=2),
        fill="tozeroy",
    )


def plot_empty_fig(model_roc_auc):
    # Subplots
    return tls.make_subplots(
        rows=4,
        cols=2,
        print_grid=False,
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
        subplot_titles=(
            "Confusion Matrix",
            "Metrics",
            "ROC curve" + " " + "(" + str(model_roc_auc) + ")",
            "Precision - Recall curve",
            "Cumulative gains curve",
            "Feature importance",
        ),
    )


def fig_with_traces(fig, trace1, trace2, trace3, trace4, trace5, trace6, trace7):
    fig.add_trace(trace1, 1, 1)
    fig.add_trace(trace2, 1, 2)
    fig.add_trace(trace3, 2, 1)
    fig.add_trace(trace4, 2, 1)
    fig.add_trace(trace5, 2, 2)
    fig.add_trace(trace6, 4, 1)
    fig.add_trace(trace7, 3, 1)

    fig["layout"].update(
        showlegend=False,
        title="Model performance report" + "XGBoost after tuning",
        autosize=False,
        height=1500,
        width=830,
        plot_bgcolor="rgba(240,240,240, 0.95)",
        paper_bgcolor="rgba(240,240,240, 0.95)",
        margin=dict(b=195),
    )
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig["layout"]["xaxis3"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis3"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis4"].update(dict(title="recall"), range=[0, 1.05])
    fig["layout"]["yaxis4"].update(dict(title="precision"), range=[0, 1.05])
    fig["layout"]["xaxis5"].update(dict(title="Percentage contacted"))
    fig["layout"]["yaxis5"].update(dict(title="Percentage positive targeted"))
    fig.layout.title.font.size = 14
    return fig


def model_performance(
    model: xgb.XGBClassifier,
    df: DataFrame,
    x_test: np.array,
    y_test: pd.Series,
):
    y_test = pd.DataFrame(y_test).to_numpy()
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    # Conf matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    trace1 = confusion_heatmap(conf_matrix)

    # Show metrics
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    fp = conf_matrix[0, 1]
    tn = conf_matrix[0, 0]
    Accuracy = (tp + tn) / (tp + tn + fp + fn)
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1_score = 2 * (
        ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))
    )

    trace2 = plot_metrics(
        pd.DataFrame(data=[[Accuracy, Precision, Recall, F1_score]]).transpose()
    )

    # Roc curve
    model_roc_auc = round(roc_auc_score(y_test, y_score), 3)
    fpr, tpr, t = roc_curve(y_test, y_score)
    trace3, trace4 = plot_roc_curve(fpr, tpr, model_roc_auc)

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    trace5 = plot_precision_recall_curve(recall, precision)

    # Feature importance
    feature_importance = model.get_booster().get_score(importance_type="weight")
    coefficients = pd.DataFrame({"coefficients": feature_importance.values()})
    column_data = pd.DataFrame(
        {"features": df.drop("amount_log").drop("class").columns}
    )

    # Combine and process
    coef_sumry = (
        coefficients.hstack(column_data)
        .sort("coefficients", descending=True)
        .filter(pd.col("coefficients") != 0)
    )

    # Feature coefficients visualization
    trace6 = plot_feature_importance(coef_sumry)

    # Cumulative gain
    pos = (
        DataFrame(y_test).to_dummies().to_numpy()
    )  # pandas.get_dummies(y_test).to_numpy()
    pos = pos[:, 1]
    npos = np.sum(pos)
    index = np.argsort(y_score)
    index = index[::-1]
    sort_pos = pos[index]
    # cumulative sum
    cpos = np.cumsum(sort_pos)
    # recall
    recall = cpos / npos
    # size obs test
    n = y_test.shape[0]
    size = np.arange(start=1, stop=369, step=1)
    # proportion
    size = size / n
    # plots
    trace7 = plot_cumulative_gain(size, recall)
    fig: Figure = plot_empty_fig(model_roc_auc)
    fig: Figure = fig_with_traces(
        fig, trace1, trace2, trace3, trace4, trace5, trace6, trace7
    )
    pio.write_image(fig, "model/result.png")
