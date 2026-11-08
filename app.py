"""
app.py - Streamlit front-end for the Student Dropout Risk models.

The app NEVER trains anything.  It loads the five pipelines saved by training.py
(model/*.joblib), scores the uploaded test CSV and shows the evaluation results.

Run locally:   streamlit run app.py
"""
import html
import json
import os
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

THEME = os.environ.get("UI_THEME", "journal")

TOKENS = {
    "journal": dict(bg="#F6F2E9", paper="#FFFDF8", ink="#25221D", muted="#6D665A", line="#D9D1BF", side="#EFE8D8",
                    accent="#8E2F1C", accent2="#476A7A", hl="#F0DDD3", cm_lo="#FBF6EA", serif=True),}
T = TOKENS[THEME]
INK, MUTED, LINE, ACCENT, ACCENT2, PAPER = T["ink"], T["muted"], T["line"], T["accent"], T["accent2"], T["paper"]
SERIF = "'Source Serif', Georgia, serif"
SANS = "'Source Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Source Code Pro', Consolas, Menlo, monospace"

BASE_CSS = f"""
.stApp {{ background: {T['bg']}; }}
.block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {T['side']}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
[data-testid="stSidebar"] hr {{ border-color: {LINE}; }}
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {{ background-color: {ACCENT} !important; border-color: {ACCENT} !important; }}
[data-testid="stMetric"] {{ background: transparent; }}
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {{ gap: 0; }}
[data-testid="stTabs"] [role="tablist"] {{ gap: .2rem; border-bottom: 1px solid {LINE}; }}
[data-testid="stTab"] {{ padding: .6rem 1rem; height: auto; }}
[data-testid="stTab"] p {{ font-size: .95rem; color: {MUTED}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {INK}; font-weight: 700; }}
.react-aria-SelectionIndicator {{ background-color: {ACCENT} !important; }}
.note {{ font-size: .84rem; color: {MUTED}; line-height: 1.55; }}
.cap {{ font-size: .84rem; color: {MUTED}; margin: .45rem 0 1rem; line-height: 1.5; }}
.cap b {{ color: {INK}; }}
.footer {{ margin-top: 2.6rem; padding-top: .8rem; border-top: 1px solid {LINE}; font-size: .78rem; color: {MUTED}; }}
.empty h3 {{ margin: 0 0 .6rem; font-size: 1.15rem; color: {INK}; }}
.empty ol {{ margin: 0; padding-left: 1.2rem; color: {INK}; line-height: 1.9; font-size: .95rem; }}
"""

THEME_CSS = {
    "journal": f"""
.mast-top {{ display: flex; justify-content: space-between; font-family: {MONO}; font-size: .72rem; color: {MUTED};
            border-bottom: 1px solid {LINE}; padding-bottom: .45rem; }}
.mast h1 {{ font-family: {SERIF}; font-weight: 600; font-size: 2.7rem; color: {INK}; margin: 1rem 0 .1rem; padding: 0; letter-spacing: -.01em; line-height: 1.1; }}
.mast .sub {{ font-family: {SERIF}; font-style: italic; font-size: 1.15rem; color: {MUTED}; margin-bottom: 1rem; }}
.mast .rule {{ border-top: 3px solid {INK}; border-bottom: 1px solid {INK}; height: 5px; margin-bottom: 1rem; }}
.mast .abs {{ font-family: {SERIF}; font-size: 1rem; line-height: 1.7; color: {INK}; max-width: 50rem; margin: 0 0 .6rem; }}
.mast .facts {{ font-family: {MONO}; font-size: .74rem; color: {MUTED}; margin-bottom: 1.4rem; }}
.section {{ font-family: {SERIF}; font-size: 1.4rem; font-weight: 600; color: {INK}; margin: 1.8rem 0 .6rem; }}
.section small {{ font-family: {MONO}; font-size: .72rem; color: {MUTED}; font-weight: 400; margin-left: .8rem; }}
.subhead {{ font-family: {SERIF}; font-style: italic; font-size: 1rem; color: {INK}; margin: .2rem 0 .4rem; }}
[data-testid="stMetric"] {{ border-top: 2px solid {INK}; border-bottom: 1px solid {INK}; padding: .7rem 1rem .6rem; border-left: 1px solid {LINE}; }}
div[data-testid="stColumn"]:first-child [data-testid="stMetric"] {{ border-left: none; padding-left: .2rem; }}
[data-testid="stMetricLabel"] p {{ font-size: .8rem; color: {MUTED}; font-style: italic; font-family: {SERIF}; }}
[data-testid="stMetricValue"] {{ font-family: {MONO}; font-weight: 500; font-size: 1.65rem; color: {INK}; }}
.strip {{ font-family: {SERIF}; font-style: italic; color: {MUTED}; font-size: .95rem; margin: 0 0 .2rem; }}
.strip b {{ font-style: normal; color: {INK}; }}
.cap, .note {{ font-family: {SERIF}; }}
.empty {{ border-top: 3px solid {INK}; border-bottom: 1px solid {INK}; padding: 1.2rem .2rem; font-family: {SERIF}; }}
.empty h3 {{ font-family: {SERIF}; font-weight: 600; }}
.step {{ font-family: {MONO}; font-size: .72rem; color: {ACCENT}; margin: 0; }}
.stitle {{ font-family: {SERIF}; font-size: 1.15rem; font-weight: 600; color: {INK}; margin: 0 0 .5rem; }}
.footer {{ font-family: {SERIF}; font-style: italic; }}
""",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear model; outputs calibrated probabilities. Strong baseline when the signal is mostly additive.",
    "Decision Tree": "Rule-based model (depth-limited to avoid over-fitting). Easy to explain, but unstable and usually lower AUC.",
    "KNN": "Votes among the nearest training students in scaled feature space. Scoring is slower because it searches the training set.",
    "Naive Bayes": "Gaussian Naive Bayes - assumes features are independent given the class. Very fast.",
    "Support Vector Machine": "RBF-kernel SVM; AUC uses the signed distance to the decision boundary. Scoring 20,000 rows takes roughly a minute.",
}

st.set_page_config(page_title="Student Dropout Risk | Model Evaluation", layout="wide")
st.markdown(f"<style>{BASE_CSS}{THEME_CSS[THEME]}</style>", unsafe_allow_html=True)

# matplotlib look that matches the page
plt.rcParams.update({
    "font.family": "serif" if T["serif"] else "sans-serif", "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
})
CM_MAP = LinearSegmentedColormap.from_list("theme", [T["cm_lo"], ACCENT])


def section(title, hint=""):
    small = f"<small>{html.escape(hint)}</small>" if hint else ""
    st.markdown(f'<div class="section">{html.escape(title)}{small}</div>', unsafe_allow_html=True)


def caption(label, text):
    st.markdown(f'<p class="cap"><b>{html.escape(label)}</b> {html.escape(text)}</p>', unsafe_allow_html=True)


def masthead(n_features, n_test, sklearn_version):
    """Header block; its structure differs per theme (not just its colours)."""
    t_ = f"<code>{TARGET}</code>"
    desc = ("Comparative evaluation of five classifiers - Logistic Regression, Decision Tree, K-Nearest Neighbor, "
            "Naive Bayes and Support Vector Machine - predicting whether a student will drop out "
            f"({t_} = 1), scored on a held-out test set with the saved models.")
    if THEME == "journal":
        return (f'<div class="mast"><div class="mast-top"><span>Machine Learning &middot; S1-26_AIMLZG565</span>'
                "<span>Assignment 2</span></div>"
                "<h1>Student Dropout Risk</h1>"
                '<div class="sub">A comparative evaluation of five classification models</div>'
                '<div class="rule"></div>'
                f'<p class="abs"><b>Abstract.</b> {desc}</p>'
                f'<div class="facts">binary classification &nbsp;/&nbsp; {n_features} input features &nbsp;/&nbsp; '
                f"test set n = {n_test:,} &nbsp;/&nbsp; scikit-learn {sklearn_version}</div></div>")
    if THEME == "swiss":
        return ('<div class="mast"><div class="bar"></div>'
                '<div class="meta">ML ASSIGNMENT 2 / S1-26_AIMLZG565</div>'
                "<h1>Student dropout risk</h1>"
                '<div class="sub">Model evaluation &mdash; five classifiers, one held-out test set.</div>'
                f'<div class="data">BINARY<i>/</i>{n_features} FEATURES<i>/</i>5 MODELS<i>/</i>N = {n_test:,}'
                f"<i>/</i>SKLEARN {sklearn_version}</div></div>")
    rows = [("Task", "Binary classification"), ("Target", TARGET), ("Input features", str(n_features)),
            ("Models compared", "5"), ("Test set", f"{n_test:,} students"), ("scikit-learn", sklearn_version)]
    table = "".join(f"<tr><td>{a}</td><td>{html.escape(b)}</td></tr>" for a, b in rows)
    return ('<div class="mast"><div class="l"><div class="eyebrow">Machine Learning &middot; Assignment 2 &middot; Evaluation report</div>'
            f"<h1>Student Dropout Risk</h1><p>{desc}</p></div><table>{table}</table></div>")


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
st.markdown(masthead(len(FEATURES), meta["split"]["n_test"], trained_with), unsafe_allow_html=True)

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

caption("Table 1.", f"Evaluation metrics of {model_name} on {len(df):,} test students (positive class = dropout).")
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
        sns.heatmap(cm, annot=annot, fmt="", cmap=CM_MAP, cbar=False, ax=ax, linewidths=3, linecolor=PAPER,
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, annot_kws={"fontsize": 10})
        ax.set(xlabel="Predicted", ylabel="Actual")
        ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        caption("Figure 1.", f"Confusion matrix of {model_name}; percentages are shares of each actual class.")
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
            ax.plot(fpr, tpr, color=ACCENT, lw=2.2, label=f"{model_name} (AUC {metrics['AUC']:.4f})")
            ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
            ax.plot([0, 1], [0, 1], "--", color=ACCENT2, lw=1.2, label="Random guess")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
            ax.legend(loc="lower right", frameon=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            caption("Figure 2.", f"ROC curve of {model_name}.")
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
        st.dataframe(bench.style.format("{:.4f}").highlight_max(axis=0, color=T["hl"]), width="stretch")
        st.bar_chart(bench[["AUC", "F1", "MCC"]], color=[ACCENT, ACCENT2, MUTED])
    else:
        st.info("model/metrics.csv not found.")

st.markdown('<div class="footer">Machine Learning (S1-26_AIMLZG565) &middot; Assignment 2 &middot; '
            "Saved scikit-learn pipelines are loaded from <code>model/</code>; no retraining occurs in this app.</div>",
            unsafe_allow_html=True)
