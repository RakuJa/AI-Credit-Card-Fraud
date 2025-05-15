import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve,
    confusion_matrix,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import random

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from pathlib import Path


import itertools

from model_handler import plot_roc_cur


class F1Score(Metric):
    def __init__(self):
        self._name = "f1_score"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        return f1_score(y_true, np.argmax(y_pred, axis=1))


# Create a confusion matrix
def custom_plot_confusion_matrix(
    model_name: str,
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    show_graph: bool = False,
    save_graph: bool = False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")


    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save_graph:
        plt.savefig(f"images/{model_name}_confusion_matrix.png", transparent=True)


def run_tabnet_model(
    model: TabNetClassifier,
    x_train,
    y_train,
    x_test,
    y_test,
    show_graph: bool = False,
    save_graph: bool = False,
):
    t0 = time.time()
    model_path: str = "model/tabnet.zip"
    if Path(model_path).exists():
        model = TabNetClassifier()
        model.load_model(model_path)
    else:
        model.fit(
            X_train=x_train,
            y_train=y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            eval_name=["train", "valid"],
            max_epochs=200, #200, metto 10
            patience=50,
            batch_size=1024 * 15,
            virtual_batch_size=256 * 10,
            num_workers=4,
            drop_last=False,
            eval_metric=["f1_score"],
        )

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time() - t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    # print("Precision  = {}".format(precision))
    # print("Recall  = {}".format(recall))
    print("F1 Score  = {}".format(f1))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test, y_pred, digits=5))

    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fper, tper, thresholds = roc_curve(y_test, probs)
    plot_roc_cur("tabnet", fper, tper, show_graph=show_graph, save_graph=save_graph)

    oversample_cm = confusion_matrix(y_test, y_pred)
    custom_plot_confusion_matrix(
        "tabnet",
        oversample_cm,
        classes=[0, 1],
        title="TabNet + SMOTE \n Confusion Matrix",
        cmap=plt.cm.Blues,
        save_graph=save_graph,
        show_graph=show_graph,
    )

    model.save_model("model/tabnet")
    return model, accuracy, roc_auc, f1, coh_kap, time_taken
