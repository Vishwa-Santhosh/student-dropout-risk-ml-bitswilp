"""
app.py - Streamlit front-end for the Student Dropout Risk models.

The app NEVER trains anything.  It loads the five pipelines saved by training.py
(model/*.joblib), scores the uploaded test CSV and shows the evaluation results.

Run locally:   streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from common import CLASS_LABELS, MODEL_FILES, TARGET, compute_metrics, score_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

MODEL_NOTES = {
    "Logistic Regression": "Linear model; outputs calibrated probabilities. Strong baseline when the signal is mostly additive.",
    "Decision Tree": "Rule-based model (depth-limited to avoid over-fitting). Easy to explain, but unstable and usually lower AUC.",
    "KNN": "Votes among the nearest training students in scaled feature space. Scoring is slower because it searches the training set.",
    "Naive Bayes": "Gaussian Naive Bayes - assumes features are independent given the class. Very fast.",
    "Support Vector Machine": "RBF-kernel SVM; AUC uses the signed distance to the decision boundary. Scoring 20,000 rows takes roughly a minute.",
}

st.set_page_config(page_title="Student Dropout Risk - Model Evaluation", page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text())


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])  # saved pipeline: preprocessing + classifier


@st.cache_data(show_spinner=False)
def run_inference(model_name, X):
    """Score X with the SAVED model (predict only - no fit anywhere in this app)."""
    return score_model(load_model(model_name), X)


def read_csv_safely(source):
    try:
        return pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001 - show any parsing problem to the user
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()


# --------------------------------------------------------------------------- startup checks
try:
    meta = load_metadata()
except FileNotFoundError:
    st.error("model/metadata.json not found. Run `python training.py` first and commit the model/ folder.")
    st.stop()

FEATURES = meta["features"]
trained_with = meta["versions"]["scikit-learn"]
if trained_with != sklearn.__version__:
    st.warning(
        f"Models were trained with scikit-learn {trained_with} but this app runs {sklearn.__version__}. "
        "Pin the same version in requirements.txt to avoid loading problems."
    )

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + target)", type="csv")
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False,
                             disabled=not SAMPLE_CSV.exists())
    if SAMPLE_CSV.exists():
        st.download_button("⬇ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2 · Model")
    model_name = st.selectbox("Select a trained model", list(MODEL_FILES))
    st.caption(MODEL_NOTES[model_name])
    st.info("Models are loaded from `model/*.joblib`. Uploaded data is only scored - it is never used for retraining.")

# --------------------------------------------------------------------------- header
st.title("🎓 Student Dropout Risk - Model Evaluation")
st.markdown(
    "Evaluate five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on a **held-out test set** "
    f"to predict whether a student will drop out (`{TARGET}` = 1)."
)

with st.expander("Expected CSV format"):
    st.write(f"The file must contain these **{len(FEATURES)} feature columns** plus the target column `{TARGET}` (0 / 1). "
             "Extra columns (e.g. an id column) are ignored.")
    st.code(", ".join(FEATURES + [TARGET]), language=None)

# --------------------------------------------------------------------------- load + validate data
if uploaded is not None:
    df = read_csv_safely(uploaded)
    source_label = uploaded.name
elif use_sample:
    df = read_csv_safely(SAMPLE_CSV)
    source_label = "bundled test_data.csv"
else:
    st.info("👈 Upload the test CSV in the sidebar (or tick *Use the bundled test_data.csv*) to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required column(s): {', '.join(missing)}")
    st.stop()

y_true = pd.to_numeric(df[TARGET], errors="coerce")
if y_true.isna().any() or not set(y_true.unique()) <= {0, 1}:
    st.error(f"Column `{TARGET}` must contain only 0 and 1.")
    st.stop()
y_true = y_true.astype(int).to_numpy()
X = df[FEATURES]

st.success(f"Loaded **{source_label}** - {len(df):,} rows · dropout rate "
           f"{y_true.mean():.1%} ({int(y_true.sum()):,} positives)")
if len(set(y_true)) < 2:
    st.warning("Only one class present in the target column, so AUC cannot be computed.")

# --------------------------------------------------------------------------- inference
try:
    with st.spinner(f"Scoring {len(X):,} rows with {model_name} ..."):
        y_pred, y_score = run_inference(model_name, X)
except Exception as exc:  # noqa: BLE001
    st.error(f"Scoring failed - check that column types match the training data.\n\n`{exc}`")
    st.stop()

metrics = compute_metrics(y_true, y_pred, y_score)

st.subheader(f"Results · {model_name}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")

tab_cm, tab_roc, tab_pred, tab_bench = st.tabs(
    ["Confusion matrix & report", "ROC curve", "Predictions", "All saved models (benchmark)"]
)

with tab_cm:
    left, right = st.columns(2)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1f}% of row)" for j in range(2)] for i in range(2)])
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        ax.set(xlabel="Predicted", ylabel="Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_LABELS,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN = {tn:,} · FP = {fp:,} · FN = {fn:,} · TP = {tp:,}")

with tab_roc:
    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, label=f"{model_name} (AUC {metrics['AUC']:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("ROC curve needs both classes in the target column.")

with tab_pred:
    out = df.copy()
    out["predicted_dropout"] = y_pred
    out["score"] = y_score.round(4)
    st.dataframe(out.head(200), width="stretch")
    st.download_button("⬇ Download all predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv", mime="text/csv")

with tab_bench:
    bench_path = MODEL_DIR / "metrics.csv"
    if bench_path.exists():
        st.caption("Metrics recorded by training.py on the original 20 % hold-out set (same file as test_data.csv).")
        bench = pd.read_csv(bench_path).set_index("ML Model Name")
        st.dataframe(bench.style.format("{:.4f}").highlight_max(axis=0, color="#cdeccd"), width="stretch")
        st.bar_chart(bench[["AUC", "F1", "MCC"]])
    else:
        st.info("model/metrics.csv not found.")
