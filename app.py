"""
app.py – Streamlit Web Application
====================================
Interactive ML Classification Dashboard with flexible dataset support.

Features:
  • CSV dataset upload (any dataset with any columns)
  • Automatic target column detection (uses last column as target)
  • Model selection dropdown (5 classifiers)
  • Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
  • Confusion matrix heatmap & classification report
  • Side-by-side model comparison
  • Works with any classification dataset!
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Detect target column
# ─────────────────────────────────────────────────────────────────────────────
def detect_target_column(df):
    """
    Automatically detect the target column from the dataset.
    Uses the last column as the target column.
    
    Returns: (target_column_name, features_df, target_series)
    """
    target_col = df.columns[-1]
    features = df.drop(target_col, axis=1)
    target = df[target_col]
    return target_col, features, target

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* --- Global --- */
    .main { background-color: #0e1117; }

    /* --- Metric cards --- */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.15);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #9ca3af;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* --- Section dividers --- */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #818cf8;
        color: #e2e8f0;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }

    /* --- Header banner --- */
    .hero-banner {
        background: linear-gradient(135deg, #312e81 0%, #581c87 50%, #831843 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-banner h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .hero-banner p {
        color: #c4b5fd;
        font-size: 1rem;
    }

    /* --- Comparison table --- */
    .dataframe th {
        background-color: #1e1b4b !important;
        color: #c4b5fd !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load default data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_default_dataset():
    """Load Wine Quality data from UCI (red + white combined)."""
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    try:
        red = pd.read_csv(red_url, sep=";")
        white = pd.read_csv(white_url, sep=";")
    except Exception:
        st.error("❌ Could not fetch dataset from UCI. Please upload a CSV file.")
        return None

    red["wine_type"] = 0
    white["wine_type"] = 1
    df = pd.concat([red, white], ignore_index=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Models
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, solver="lbfgs", random_state=42
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, random_state=42
    ),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(
        n_neighbors=7, weights="distance"
    ),
    "Naive Bayes (Gaussian)": lambda: GaussianNB(),
    "Random Forest (Ensemble)": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
    ),
}


@st.cache_resource
def train_all_models(_X_train, _y_train):
    """Train all 5 models and return them as a dict."""
    trained = {}
    for name, factory in MODEL_REGISTRY.items():
        model = factory()
        model.fit(_X_train, _y_train)
        trained[name] = model
    return trained


def compute_auc(model, X_test, y_test):
    """Compute AUC safely for both binary and multiclass targets."""
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                return roc_auc_score(y_test, y_prob[:, 1])
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                return roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class="ovr",
                    average="weighted",
                    labels=np.unique(y_test),
                )

        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                return roc_auc_score(y_test, y_score[:, 1])
            if y_score.ndim == 2 and y_score.shape[1] > 2:
                return roc_auc_score(
                    y_test,
                    y_score,
                    multi_class="ovr",
                    average="weighted",
                    labels=np.unique(y_test),
                )

        return float("nan")
    except Exception:
        return float("nan")


def get_class_labels(y):
    """Return readable labels for binary wine targets and generic numeric classes."""
    unique_vals = sorted(pd.Series(y).dropna().unique().tolist())
    if unique_vals == [0, 1]:
        return ["Red Wine", "White Wine"]
    return [str(v) for v in unique_vals]


def compute_metrics(model, X_test, y_test):
    """Return dict of all 6 metrics + confusion matrix + classification report."""
    y_pred = model.predict(X_test)
    auc = compute_auc(model, X_test, y_test)
    class_labels = get_class_labels(y_test)
    class_values = np.unique(y_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred, labels=class_values)
    report = classification_report(
        y_test,
        y_pred,
        labels=class_values,
        target_names=class_labels,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=class_values,
        target_names=class_labels,
        zero_division=0,
    )

    return metrics, cm, report, report_text, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/machine-learning.png",
        width=64,
    )
    st.markdown("## 🤖 ML Classification")
    st.markdown("---")

    # --- Dataset upload ---
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader(
        "Upload test CSV (optional)",
        type=["csv"],
        help="Upload a CSV file with any columns. The LAST column will be used as the target variable. "
             "If not uploaded, the built-in Wine Quality dataset is used.",
    )

    st.markdown("---")

    # --- Model selector ---
    st.markdown("### 🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose a model",
        list(MODEL_REGISTRY.keys()),
        index=4,  # default: Random Forest
    )

    st.markdown("---")

    # --- Display options ---
    st.markdown("### ⚙️ Options")
    show_comparison = st.checkbox("Show All-Model Comparison", value=True)
    show_data_preview = st.checkbox("Show Data Preview", value=False)

    st.markdown("---")
    st.caption("Built for BITS WILP ML Assignment 2")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# Hero banner
st.markdown(
    f"""
    <div class="hero-banner">
        <h1>🤖 ML Classification Dashboard</h1>
        <p>Multi-class classification with flexible datasets &nbsp;•&nbsp; 5 ML models &nbsp;•&nbsp; Automatic feature detection</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ────────────────────────────────────────────────────────────────
full_df = load_default_dataset()

if full_df is None:
    st.stop()

# Automatically detect target column
target_col, X_full, y_full = detect_target_column(full_df)

X_train, X_test_default, y_train, y_test_default = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

scaler = StandardScaler()
feature_cols = X_train.columns.tolist()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
)
X_test_scaled_default = pd.DataFrame(
    scaler.transform(X_test_default), columns=feature_cols, index=X_test_default.index
)

# Handle uploaded file vs default test set
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        user_df.columns = [c.strip().replace(" ", "_") for c in user_df.columns]

        # Detect target column in user dataset
        user_target_col, user_X, user_y = detect_target_column(user_df)
        
        if len(user_X.columns) == 0:
            st.error("❌ Uploaded CSV must contain at least one feature column and one target column.")
            st.stop()

        X_user = user_X
        y_user = user_y

        # Ensure same columns
        missing = set(feature_cols) - set(X_user.columns)
        if missing:
            st.error(f"❌ Uploaded CSV is missing columns: {missing}")
            st.stop()

        X_user = X_user[feature_cols]
        X_test_eval = pd.DataFrame(
            scaler.transform(X_user), columns=feature_cols, index=X_user.index
        )
        y_test_eval = y_user
        st.success(f"✅ Using uploaded CSV ({len(user_df)} rows)")
        data_source = "Uploaded CSV"
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()
else:
    X_test_eval = X_test_scaled_default
    y_test_eval = y_test_default
    data_source = "Default Test Set (20% holdout)"

# ── Train models ─────────────────────────────────────────────────────────────
with st.spinner("Training models... (cached after first run)"):
    trained_models = train_all_models(X_train_scaled, y_train)

# ── Dataset Info ─────────────────────────────────────────────────────────────
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("📊 Total Samples", f"{len(full_df):,}")
col_info2.metric("🎯 Features", len(feature_cols))
col_info3.metric("🏷️ Classes", len(y_full.unique()))
col_info4.metric("📁 Data Source", data_source.split("(")[0].strip())

if show_data_preview:
    st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        full_df.head(20).style.format(precision=3),
        width="stretch",
        height=300,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Selected Model Results
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="section-header">📈 Results — {selected_model}</div>',
    unsafe_allow_html=True,
)

model = trained_models[selected_model]
metrics, cm, report_dict, report_text, y_pred = compute_metrics(
    model, X_test_eval, y_test_eval
)

# Metric cards
metric_cols = st.columns(6)
metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metric_colors = ["#818cf8", "#a78bfa", "#c084fc", "#e879f9", "#f472b6", "#fb923c"]

for i, (col, name) in enumerate(zip(metric_cols, metric_names)):
    val = metrics[name]
    display_val = f"{val:.4f}" if not np.isnan(val) else "N/A"
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{name}</div>
            <div class="metric-value" style="background: linear-gradient(135deg, {metric_colors[i]}, {metric_colors[(i+1)%6]});
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {display_val}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Confusion Matrix & Classification Report side-by-side
col_cm, col_report = st.columns([1, 1])

with col_cm:
    st.markdown("#### 🔲 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5.5))
    fig_cm.patch.set_facecolor("#0e1117")
    ax_cm.set_facecolor("#0e1117")

    classes = get_class_labels(y_test_eval)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="#2d2d44",
        ax=ax_cm,
        cbar_kws={"shrink": 0.8},
    )
    ax_cm.set_xlabel("Predicted", fontsize=12, color="white")
    ax_cm.set_ylabel("Actual", fontsize=12, color="white")
    ax_cm.set_title(f"Confusion Matrix – {selected_model}", fontsize=13, color="white", pad=12)
    ax_cm.tick_params(colors="white")

    st.pyplot(fig_cm)
    plt.close(fig_cm)

with col_report:
    st.markdown("#### 📝 Classification Report")
    st.code(report_text, language="text")

# ─────────────────────────────────────────────────────────────────────────────
# All-Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
if show_comparison:
    st.markdown(
        '<div class="section-header">🏆 All-Model Comparison</div>',
        unsafe_allow_html=True,
    )

    comparison_data = {}
    for name, mdl in trained_models.items():
        m, _, _, _, _ = compute_metrics(mdl, X_test_eval, y_test_eval)
        comparison_data[name] = m

    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df.index.name = "Model"

    # Highlight best values
    def highlight_best(s):
        is_best = s == s.max()
        return ["background-color: #312e81; font-weight: bold" if v else "" for v in is_best]

    st.dataframe(
        comparison_df.style
        .format(precision=4)
        .apply(highlight_best, axis=0),
        width="stretch",
    )

    # Bar chart comparison
    st.markdown("#### 📊 Visual Comparison")
    fig_bar, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig_bar.patch.set_facecolor("#0e1117")

    colors_bar = ["#818cf8", "#a78bfa", "#c084fc", "#e879f9", "#f472b6", "#fb923c"]

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("#1e1e2f")

        vals = comparison_df[metric_name].values
        model_short = [n.split("(")[0].strip()[:12] for n in comparison_df.index]
        bars = ax.bar(model_short, vals, color=colors_bar[idx], edgecolor="none", alpha=0.85)
        ax.set_title(metric_name, color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_ylim(0, max(1.0, max(vals) * 1.15) if max(vals) > 0 else 1.0)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

        for spine in ax.spines.values():
            spine.set_color("#2d2d44")

    plt.tight_layout(pad=2.0)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # Winner
    winner_model = comparison_df["F1"].idxmax()
    winner_f1 = comparison_df.loc[winner_model, "F1"]
    st.success(f"🏆 **Best Model (by F1 Score):** {winner_model} — F1 = {winner_f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 1rem 0;">
        Wine Quality Classification Dashboard &nbsp;|&nbsp;
        Dataset: <a href="https://archive.ics.uci.edu/dataset/186/wine+quality"
            style="color: #818cf8; text-decoration: none;">UCI Wine Quality</a> &nbsp;|&nbsp;
        Built with Streamlit & scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
