"""
app.py - Streamlit front-end for the Student Dropout Risk models.

The app NEVER trains anything.  It loads the five pipelines saved by training.py
(model/*.joblib), scores the uploaded test CSV and shows the evaluation results.

Run locally:   streamlit run app.py
"""
import html
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from common import CLASS_LABELS, MODEL_FILES, TARGET, compute_metrics, score_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

# ----------------------------------------------------------------------------- design tokens
NAVY, GOLD, SLATE, INK, MUTED, LINE = "#14284B", "#B08D57", "#7C8DA6", "#1B2430", "#5B6677", "#D9DEE6"
SERIF = "Georgia, 'Times New Roman', serif"

CSS = f"""
<style>
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

/* header banner */
.hero {{ background: {NAVY}; color: #fff; padding: 1.7rem 2rem 1.5rem; border-radius: 3px;
        border-bottom: 4px solid {GOLD}; margin-bottom: 1.4rem; }}
.hero .kicker {{ font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; color: #C4CFE4; }}
.hero h1 {{ font-family: {SERIF}; color: #fff; font-size: 2rem; font-weight: 700; margin: .3rem 0 .45rem; padding: 0; line-height: 1.2; }}
.hero code {{ background: rgba(255,255,255,.16); color: #fff; padding: .05rem .4rem; border-radius: 2px; font-size: .85em; }}
.hero p {{ color: #DCE3F0; margin: 0; font-size: .95rem; max-width: 54rem; line-height: 1.55; }}
.chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.1rem; }}
.chip {{ border: 1px solid rgba(255,255,255,.30); padding: .2rem .65rem; font-size: .74rem; letter-spacing: .03em; color: #E8EDF7; }}

/* section headings */
.section {{ font-family: {SERIF}; font-size: 1.2rem; font-weight: 700; color: {NAVY};
           border-bottom: 1px solid {LINE}; padding-bottom: .4rem; margin: 1.5rem 0 .9rem; }}
.section small {{ font-family: sans-serif; font-weight: 400; color: {MUTED}; font-size: .8rem; margin-left: .6rem; }}
.subhead {{ font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: {MUTED}; font-weight: 700; margin: .2rem 0 .5rem; }}

/* metric cards */
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-top: 3px solid {NAVY};
                           border-radius: 2px; padding: .85rem 1rem; }}
[data-testid="stMetricLabel"] p {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .12em; color: {MUTED}; font-weight: 700; }}
[data-testid="stMetricValue"] {{ font-family: {SERIF}; color: {NAVY}; font-size: 1.85rem; font-weight: 700; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: .65rem 1.2rem; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: .92rem; color: {MUTED}; }}
.stTabs [aria-selected="true"] p {{ color: {NAVY}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {GOLD}; height: 3px; }}

/* info strip, empty state, sidebar labels, footer */
.strip {{ background: #fff; border: 1px solid {LINE}; border-left: 4px solid {GOLD}; padding: .7rem 1rem;
         font-size: .9rem; color: {INK}; margin-bottom: .4rem; }}
.strip b {{ color: {NAVY}; }}
.empty {{ background: #fff; border: 1px solid {LINE}; border-top: 3px solid {NAVY}; padding: 1.4rem 1.6rem; }}
.empty h3 {{ font-family: {SERIF}; color: {NAVY}; margin: 0 0 .6rem; font-size: 1.15rem; }}
.empty ol {{ margin: 0; padding-left: 1.2rem; color: {INK}; line-height: 1.9; font-size: .92rem; }}
.step {{ font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; color: {GOLD}; font-weight: 700; margin: 0; }}
.stitle {{ font-family: {SERIF}; font-size: 1.1rem; font-weight: 700; color: {NAVY}; margin: 0 0 .5rem; }}
.note {{ font-size: .82rem; color: {MUTED}; line-height: 1.5; }}
.footer {{ margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid {LINE}; font-size: .78rem; color: {MUTED}; text-align: center; letter-spacing: .02em; }}
</style>
"""

MODEL_NOTES = {
    "Logistic Regression": "Linear model; outputs calibrated probabilities. Strong baseline when the signal is mostly additive.",
    "Decision Tree": "Rule-based model (depth-limited to avoid over-fitting). Easy to explain, but unstable and usually lower AUC.",
    "KNN": "Votes among the nearest training students in scaled feature space. Scoring is slower because it searches the training set.",
    "Naive Bayes": "Gaussian Naive Bayes - assumes features are independent given the class. Very fast.",
    "Support Vector Machine": "RBF-kernel SVM; AUC uses the signed distance to the decision boundary. Scoring 20,000 rows takes roughly a minute.",
}

st.set_page_config(page_title="Student Dropout Risk | Model Evaluation", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# matplotlib look that matches the page
plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E7EAF0", "grid.linewidth": 0.8, "axes.axisbelow": True,
})
NAVY_MAP = LinearSegmentedColormap.from_list("navy", ["#F4F6FA", "#9DAAC4", NAVY])


def section(title, hint=""):
    small = f"<small>{html.escape(hint)}</small>" if hint else ""
    st.markdown(f'<div class="section">{html.escape(title)}{small}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached loaders
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


# ----------------------------------------------------------------------------- startup checks
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

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<p class="step">Step 1</p><p class="stitle">Test data</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload test CSV (features + target)", type="csv")
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False,
                             disabled=not SAMPLE_CSV.exists())
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.markdown("---")
    st.markdown('<p class="step">Step 2</p><p class="stitle">Model</p>', unsafe_allow_html=True)
    model_name = st.selectbox("Select a trained model", list(MODEL_FILES))
    st.markdown(f'<p class="note">{html.escape(MODEL_NOTES[model_name])}</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="note">Models are loaded from <code>model/*.joblib</code>. Uploaded data is only '
                'scored - it is never used for retraining.</p>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- header
chips = [
    "Binary classification",
    f"{len(FEATURES)} input features",
    "5 classifiers",
    f"Hold-out test set: {meta['split']['n_test']:,} students",
    f"scikit-learn {trained_with}",
]
st.markdown(
    '<div class="hero"><div class="kicker">Machine Learning &middot; Assignment 2</div>'
    "<h1>Student Dropout Risk &mdash; Model Evaluation</h1>"
    "<p>Evaluate five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbor, Naive Bayes, "
    f"Support Vector Machine) on a held-out test set to predict whether a student will drop out "
    f"(<code>{TARGET}</code> = 1).</p>"
    '<div class="chips">' + "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips) + "</div></div>",
    unsafe_allow_html=True,
)

with st.expander("Expected CSV format"):
    st.write(f"The file must contain these **{len(FEATURES)} feature columns** plus the target column `{TARGET}` (0 / 1). "
             "Extra columns (e.g. an id column) are ignored.")
    st.code(", ".join(FEATURES + [TARGET]), language=None)

# ----------------------------------------------------------------------------- load + validate data
if uploaded is not None:
    df = read_csv_safely(uploaded)
    source_label = uploaded.name
elif use_sample:
    df = read_csv_safely(SAMPLE_CSV)
    source_label = "bundled test_data.csv"
else:
    st.markdown(
        '<div class="empty"><h3>Getting started</h3><ol>'
        "<li>Upload the test CSV in the sidebar (or tick <b>Use the bundled test_data.csv</b>).</li>"
        "<li>Select one of the five trained models.</li>"
        "<li>Review the metrics, confusion matrix, classification report and ROC curve.</li></ol></div>",
        unsafe_allow_html=True,
    )
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

st.markdown(
    f'<div class="strip"><b>Data loaded:</b> {html.escape(source_label)} &nbsp;&middot;&nbsp; '
    f"<b>{len(df):,}</b> rows &nbsp;&middot;&nbsp; dropout rate <b>{y_true.mean():.1%}</b> "
    f"({int(y_true.sum()):,} positives)</div>",
    unsafe_allow_html=True,
)
if len(set(y_true)) < 2:
    st.warning("Only one class present in the target column, so AUC cannot be computed.")

# ----------------------------------------------------------------------------- inference
try:
    with st.spinner(f"Scoring {len(X):,} rows with {model_name} ..."):
        y_pred, y_score = run_inference(model_name, X)
except Exception as exc:  # noqa: BLE001
    st.error(f"Scoring failed - check that column types match the training data.\n\n`{exc}`")
    st.stop()

metrics = compute_metrics(y_true, y_pred, y_score)

section(f"Evaluation: {model_name}", "positive class = dropout (1)")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")

st.write("")
tab_cm, tab_roc, tab_pred, tab_bench = st.tabs(
    ["Confusion Matrix & Report", "ROC Curve", "Predictions", "Benchmark: All Models"]
)

with tab_cm:
    left, right = st.columns(2)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1f}% of row)" for j in range(2)] for i in range(2)])
    with left:
        st.markdown('<p class="subhead">Confusion matrix</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=annot, fmt="", cmap=NAVY_MAP, cbar=False, ax=ax, linewidths=3, linecolor="white",
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, annot_kws={"fontsize": 10})
        ax.set(xlabel="Predicted", ylabel="Actual")
        ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown('<p class="subhead">Classification report</p>', unsafe_allow_html=True)
        report = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_LABELS,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f'<p class="note">TN = {tn:,} &nbsp;&middot;&nbsp; FP = {fp:,} &nbsp;&middot;&nbsp; '
                    f"FN = {fn:,} &nbsp;&middot;&nbsp; TP = {tp:,}</p>", unsafe_allow_html=True)

with tab_roc:
    if len(set(y_true)) == 2:
        roc_left, roc_right = st.columns([3, 2])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        with roc_left:
            st.markdown('<p class="subhead">Receiver operating characteristic</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.plot(fpr, tpr, color=NAVY, lw=2.2, label=f"{model_name} (AUC {metrics['AUC']:.4f})")
            ax.fill_between(fpr, tpr, alpha=0.07, color=NAVY)
            ax.plot([0, 1], [0, 1], "--", color=GOLD, lw=1.4, label="Random guess")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
            ax.legend(loc="lower right", frameon=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with roc_right:
            st.markdown('<p class="subhead">How to read this</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="note">The curve shows how many real dropouts are caught (true positive rate) against how many '
                "students are wrongly flagged (false positive rate) as the decision threshold moves. A curve closer to the "
                "top-left corner is better; the dashed line is random guessing (AUC = 0.5). "
                f"For this model, <b>AUC = {metrics['AUC']:.4f}</b>.</p>", unsafe_allow_html=True)
    else:
        st.info("ROC curve needs both classes in the target column.")

with tab_pred:
    out = df.copy()
    out["predicted_dropout"] = y_pred
    out["score"] = y_score.round(4)
    st.markdown('<p class="subhead">First 200 rows</p>', unsafe_allow_html=True)
    st.dataframe(out.head(200), width="stretch")
    st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv", mime="text/csv")

with tab_bench:
    bench_path = MODEL_DIR / "metrics.csv"
    if bench_path.exists():
        st.markdown('<p class="note">Metrics recorded by training.py on the original 20 % hold-out set '
                    "(same file as test_data.csv). Highlighted cell = best in column.</p>", unsafe_allow_html=True)
        bench = pd.read_csv(bench_path).set_index("ML Model Name")
        st.dataframe(bench.style.format("{:.4f}").highlight_max(axis=0, color="#F1E6CF"), width="stretch")
        st.bar_chart(bench[["AUC", "F1", "MCC"]], color=[NAVY, GOLD, SLATE])
    else:
        st.info("model/metrics.csv not found.")

st.markdown('<div class="footer">Machine Learning (S1-26_AIMLZG565) &middot; Assignment 2 &middot; '
            "Saved scikit-learn pipelines are loaded from <code>model/</code>; no retraining occurs in this app.</div>",
            unsafe_allow_html=True)
