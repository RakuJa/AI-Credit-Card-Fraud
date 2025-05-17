import numpy as np
import polars as pd
import optuna  # pip install optuna
import xgboost as xgb
from polars import DataFrame
import plotly.graph_objs as go
import plotly.subplots as tls
import plotly.io as pio

# from pandas.core.interchange.dataframe_protocol import DataFrame
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


def objective(trial, x_train_smt, y_train_smt):
    param_grid = {
        #         "device_type": trial.suggest_categorical("device_type", ['gpu']),
        # "n_estimators": trial.suggest_categorical("n_estimators", 1,300 ),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.3),
        # "num_leaves": trial.suggest_int("num_leaves", 4, 2**max_depth),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 500),
        "max_bin": trial.suggest_int("max_bin", 10, 300),
        # "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        # "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "bagging_fraction": trial.suggest_float(
        #    "bagging_fraction", 0.2, 0.95, step=0.1
        # ),
        # "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        # "feature_fraction": trial.suggest_float(
        #    "feature_fraction", 0.2, 0.95, step=0.1
        # ),
    }
    # scoring = {'accuracy' : make_scorer(accuracy_score),
    #      'precision' : make_scorer(precision_score),
    #     'recall' : make_scorer(recall_score),
    #    'f1_score' : make_scorer(f1_score)}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = xgb.XGBClassifier(**param_grid)
    model.fit(x_train_smt, y_train_smt)
    scores = cross_val_score(
        model,
        x_train_smt,
        y_train_smt,
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

    param_xgb = {
        "max_depth": 13,
        "learning_rate": 0.22690320686746146,
        # "num_leaves": 501,
        # "min_data_in_leaf": 168,
        "max_bin": 61,
        # "lambda_l1": 0,
        # "lambda_l2": 0,
        # "min_gain_to_split": 1.2780588498979437,
        # "bagging_fraction": 0.9,
        # "bagging_freq": 1,
        # "feature_fraction": 0.2,
    }

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


def model_performance(
    model: xgb.XGBClassifier, df: DataFrame, x_test: np.array, y_test: pd.Series
):
    y_test = pd.DataFrame(y_test).to_numpy()
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    # Conf matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    trace1 = go.Heatmap(
        z=conf_matrix,
        x=["0 (pred)", "1 (pred)"],
        y=["0 (true)", "1 (true)"],
        xgap=2,
        ygap=2,
        text=conf_matrix,
        colorscale="Viridis",
        showscale=False,
    )

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

    show_metrics = pd.DataFrame(data=[[Accuracy, Precision, Recall, F1_score]])
    show_metrics = show_metrics.transpose()

    colors = ["gold", "lightgreen", "lightcoral", "lightskyblue"]
    trace2 = go.Bar(
        x=show_metrics.to_numpy()[0],
        y=["Accuracy", "Precision", "Recall", "F1_score"],
        text=np.round(show_metrics.to_numpy()[0], 4),
        textposition="auto",
        orientation="h",
        opacity=0.8,
        marker=dict(color=colors, line=dict(color="#000000", width=1.5)),
    )
    # Roc curve
    model_roc_auc = round(roc_auc_score(y_test, y_score), 3)
    fpr, tpr, t = roc_curve(y_test, y_score)
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

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    trace5 = go.Scatter(
        x=recall,
        y=precision,
        name="Precision" + str(precision),
        line=dict(color="lightcoral", width=2),
        fill="tozeroy",
    )

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
    trace6 = go.Bar(
        x=coef_sumry["features"].to_list(),
        y=coef_sumry["coefficients"].to_list(),
        name="coefficients",
        marker=dict(
            color=coef_sumry["coefficients"].to_list(),
            colorscale="Viridis",
            line=dict(width=0.6, color="black"),
        ),
    )

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
    trace7 = go.Scatter(
        x=size,
        y=recall,
        name="Lift curve",
        line=dict(color="gold", width=2),
        fill="tozeroy",
    )

    # Subplots
    fig = tls.make_subplots(
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

    # fig.show("colab")
    pio.write_image(fig, "model/result.png")
