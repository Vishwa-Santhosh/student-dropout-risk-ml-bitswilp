"""
training.py - trains the five classifiers, evaluates them on a held-out test
set and saves everything the Streamlit app needs.

Run from the project root:      python training.py

Outputs
-------
model/*.joblib        five complete pipelines (preprocessing + classifier)
model/metrics.csv     the 6 metrics x 5 models comparison table
model/metadata.json   features, split info, class distribution, best params,
                      library versions, confusion matrices
test_data.csv         the held-out 20 % test split (features + target) - this is
                      the file you upload in the Streamlit app
assets/*.png          ROC curves / confusion matrices / metric comparison plots
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import CLASS_LABELS, MODEL_FILES, TARGET, compute_metrics, score_model

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_dropout_risk_2026.csv"
MODEL_DIR = ROOT / "model"
ASSET_DIR = ROOT / "assets"
TEST_CSV = ROOT / "test_data.csv"

ID_COLUMNS = ["person_id"]   # row identifier: carries no information -> dropped
TEST_SIZE = 0.20
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def class_dist(y):
    counts = y.value_counts().sort_index()
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "percent": {int(k): round(100 * v / len(y), 2) for k, v in counts.items()},
    }


def build_preprocessor(num_cols, cat_cols):
    """Impute -> scale numeric; impute -> one-hot encode categorical.
    Fitted ONLY on the training split (it lives inside each Pipeline)."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ]
    )


def main(skip_predict_check=False):
    ASSET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    log(f"Loading {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    n_rows_raw, n_cols_raw = df.shape
    n_missing = int(df.isna().sum().sum())
    n_dups = int(df.drop(columns=ID_COLUMNS).duplicated().sum())
    log(f"Raw shape {df.shape} | missing cells: {n_missing} | duplicate rows (ignoring id): {n_dups}")

    X = df.drop(columns=ID_COLUMNS + [TARGET])
    y = df[TARGET].astype(int)
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    log(f"{X.shape[1]} features ({len(num_cols)} numeric, {len(cat_cols)} categorical) | "
        f"target '{TARGET}' distribution: {class_dist(y)['percent']}")

    # ----------------------------------------------------------------- split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log(f"Train {X_train.shape} | Test {X_test.shape} (stratified, seed {RANDOM_STATE})")

    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(TEST_CSV, index=False)
    log(f"Saved held-out test set -> {TEST_CSV.name}")

    # ---------------------------------------------------------------- models
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def pipe(clf):
        return Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("clf", clf)])

    # (estimator, grid or None).  Tuning uses 3-fold CV on the TRAINING split only.
    # SVM / Naive Bayes use sensible defaults: SVM tuning is O(n^2) in rows and too
    # slow for 80,000 samples; GaussianNB has no meaningful hyper-parameter to tune.
    specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": [3, 5, 7, 10, 15], "clf__min_samples_leaf": [20, 100, 300]}),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [51, 101, 201, 301, 501]}),
        "Naive Bayes": (GaussianNB(), None),
        "Support Vector Machine": (
            SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000), None),
    }

    best_params, fit_seconds = {}, {}
    for name, (clf, grid) in specs.items():
        t0 = time.time()
        log(f"Training {name} ...")
        if grid:
            gs = GridSearchCV(pipe(clf), grid, scoring="roc_auc", cv=cv, n_jobs=1)
            gs.fit(X_train, y_train)
            fitted = gs.best_estimator_      # already refit on the full training split
            best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
            log(f"   best params {best_params[name]} (CV AUC {gs.best_score_:.4f})")
        else:
            fitted = pipe(clf).fit(X_train, y_train)
            best_params[name] = "defaults"
        fit_seconds[name] = round(time.time() - t0, 1)
        joblib.dump(fitted, MODEL_DIR / MODEL_FILES[name], compress=3)
        log(f"   done in {fit_seconds[name]}s -> model/{MODEL_FILES[name]}")

    # ------------------------------------------- evaluate the SAVED models
    # Metrics come from models re-loaded from disk, i.e. exactly what the app uses.
    rows, curves, cms = [], {}, {}
    for name in specs:
        loaded = joblib.load(MODEL_DIR / MODEL_FILES[name])
        t0 = time.time()
        y_pred, y_score = score_model(loaded, X_test)
        m = compute_metrics(y_test, y_pred, y_score)
        rows.append({"ML Model Name": name, **m})
        curves[name] = roc_curve(y_test, y_score)[:2]
        cms[name] = confusion_matrix(y_test, y_pred).tolist()
        if not skip_predict_check:
            assert np.array_equal(y_pred, loaded.predict(X_test)), f"{name}: derived labels != predict()"
        log(f"{name:24s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
            + f"  ({time.time() - t0:.0f}s)")

    metrics = pd.DataFrame(rows).set_index("ML Model Name")   # values already rounded to 6 dp
    metrics.to_csv(MODEL_DIR / "metrics.csv", float_format="%.6f")
    log("\n" + metrics.map(lambda v: f"{v:.4f}").to_string())

    # ---------------------------------------------------------------- plots
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {metrics.loc[name, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves - held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(ASSET_DIR / "roc_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(name, fontsize=10)
    fig.tight_layout(); fig.savefig(ASSET_DIR / "confusion_matrices.png", dpi=150); plt.close(fig)

    ax = metrics.T.plot(kind="bar", figsize=(10, 4.5), width=0.8)
    ax.set(title="Metric comparison - held-out test set", ylabel="score", ylim=(0, 1))
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(ASSET_DIR / "metric_comparison.png", dpi=150); plt.close()

    # ------------------------------------------------------------- metadata
    meta = {
        "dataset": {
            "file": DATA_PATH.name, "rows": n_rows_raw, "raw_columns": n_cols_raw,
            "n_features": X.shape[1], "missing_cells": n_missing, "duplicate_rows": n_dups,
            "dropped_columns": ID_COLUMNS, "target": TARGET,
        },
        "features": X.columns.tolist(),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "split": {
            "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True,
            "n_train": len(X_train), "n_test": len(X_test),
            "class_distribution": {"full": class_dist(y), "train": class_dist(y_train), "test": class_dist(y_test)},
        },
        "best_params": best_params,
        "fit_seconds": fit_seconds,
        "confusion_matrices": cms,
        "versions": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__,
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    log("Saved model/metadata.json - training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-predict-check", action="store_true",
                    help="skip the (slow for SVM) assertion that derived labels == model.predict()")
    main(ap.parse_args().skip_predict_check)
