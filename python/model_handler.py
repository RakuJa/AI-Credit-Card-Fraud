import time

from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    f1_score,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)


def plot_roc_cur(fper, tper):
    plt.plot(fper, tper, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()


def run_model(model, x_train, y_train, x_test, y_test, verbose=True):
    t0 = time.time()
    if not verbose:
        model.fit(x_train, y_train, verbose=0)
    else:
        model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time() - t0
    # Accuracy is useful with balanced datasets, but kinda useless with imbalanced data.
    # If I'm operating in a sector in which 90% of data is class 1, then a model that only says
    # class 1 has 90% accuracy
    print("Accuracy = {}".format(accuracy))
    # Performance metric for binary classification (YES/NO)
    # Works well with unbalanced data
    print("ROC Area under Curve = {}".format(roc_auc))
    # When a model predicts 'X', how often is it correct? Minimizes false alarms
    if verbose:
        print("Precision  = {}".format(precision))
        # Minimizes missed positives
        print("Recall  = {}".format(recall))
        # The F1 score is a single metric that balances precision and recall. It answers:
        # "Can the model achieve high precision and high recall at the same time?"
        print("F1 Score  = {}".format(f1))
    # Cohen's Kappa measures how much better your model is compared to random chance
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test, y_pred, digits=5))

    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fper, tper, thresholds = roc_curve(y_test, probs)
    plot_roc_cur(fper, tper)

    # Updated confusion matrix plotting
    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        cmap=plt.cm.Blues,
        display_labels=model.classes_,  # Optional: show class labels
    )
    plt.title("Confusion Matrix")
    plt.show()

    return model, accuracy, roc_auc, f1, coh_kap, time_taken
