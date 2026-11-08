"""
common.py - helpers shared by training.py AND app.py.

Why a shared file?  The #1 reason "app metrics differ from notebook metrics" is
two slightly different metric implementations.  Both scripts import the SAME
functions from here, so a number shown in the Streamlit app is computed exactly
the way it was computed during training.
"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "dropout"                      # 1 = student dropped out, 0 = stayed enrolled
CLASS_LABELS = ["Stayed (0)", "Dropout (1)"]

# Display name -> file stored in model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def score_model(pipe, X):
    """Run a fitted pipeline ONCE on X and return (y_pred, y_score).

    * Models with predict_proba (LR, DT, KNN, NB): score = P(class 1),
      label = 1 when P(class 1) > 0.5  (identical to estimator.predict for a
      binary problem).
    * SVM (no probability calibration): score = signed distance from the
      hyperplane, label = 1 when distance > 0 (identical to SVC.predict).
      AUC only needs a ranking, so the raw decision score is the right input.

    training.py asserts that these derived labels equal pipe.predict(X), so
    nothing is lost - but SVM scoring (the slowest step) is done only once.
    """
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X)[:, 1]
        y_pred = (y_score > 0.5).astype(int)
    else:
        y_score = pipe.decision_function(X)
        y_pred = (y_score > 0.0).astype(int)
    return y_pred, y_score


def compute_metrics(y_true, y_pred, y_score):
    """The six metrics required by the assignment (positive class = dropout)."""
    both_classes = len(set(y_true)) == 2
    raw = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if both_classes else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # Round once to 6 dp so every place that later prints "{:.4f}" (training log,
    # metrics.csv, README, Streamlit cards) shows exactly the same 4-decimal value.
    return {k: round(float(v), 6) for k, v in raw.items()}
