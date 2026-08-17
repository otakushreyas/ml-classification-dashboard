"""
train_models.py
===============
Script to train all 5 classification models on the Wine Quality dataset,
evaluate them, save the models and generate test_data.csv.

Dataset: Wine Quality (UCI ML Repository)
- Red wine: 1599 samples, 11 physicochemical features
- White wine: 4898 samples, 11 physicochemical features
- Combined: ~6497 samples, 12 features (11 original + wine_type)
- Target: quality (multi-class: 3–9)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------

def load_wine_quality_data():
    """Load wine quality dataset from UCI repository or local cache."""
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    try:
        red = pd.read_csv(red_url, sep=";")
        white = pd.read_csv(white_url, sep=";")
    except Exception:
        # Fallback: try local files
        red = pd.read_csv("data/winequality-red.csv", sep=";")
        white = pd.read_csv("data/winequality-white.csv", sep=";")

    red["wine_type"] = 0   # 0 for red
    white["wine_type"] = 1  # 1 for white

    df = pd.concat([red, white], ignore_index=True)

    # Clean column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Split into features/target, scale, and return train/test sets."""
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    feature_cols = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


# ---------------------------------------------------------------------------
# 2. Define models
# ---------------------------------------------------------------------------

def get_models():
    """Return a dict of model name -> model instance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, multi_class="multinomial", solver="lbfgs", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, weights="distance"
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
        ),
    }


# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Compute all 6 evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)

    # For AUC, we need probability estimates
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return metrics, cm, report


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Wine Quality Classification – Model Training & Evaluation")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading Wine Quality dataset ...")
    df = load_wine_quality_data()
    print(f"      Dataset shape: {df.shape}")
    print(f"      Features: {df.columns.tolist()}")
    print(f"      Target classes: {sorted(df['quality'].unique())}")

    # Prepare
    print("\n[2/5] Preparing data (80/20 split, StandardScaler) ...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
    print(f"      Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Save test data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_df = X_test.copy()
    test_df["quality"] = y_test
    test_csv_path = os.path.join(project_root, "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"      Test data saved to: {test_csv_path}")

    # Train & evaluate models
    print("\n[3/5] Training and evaluating models ...")
    models = get_models()
    results = {}
    model_dir = os.path.join(project_root, "model")
    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\n  → {name}")
        model.fit(X_train, y_train)
        metrics, cm, report = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        for k, v in metrics.items():
            print(f"      {k}: {v:.4f}")

        # Save model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(model_dir, f"{safe_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"      Model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save feature columns
    cols_path = os.path.join(model_dir, "feature_cols.pkl")
    with open(cols_path, "wb") as f:
        pickle.dump(feature_cols, f)

    # Summary table
    print("\n" + "=" * 70)
    print("  Comparison Table")
    print("=" * 70)
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    print(results_df.round(4).to_string())

    # Winner
    winner = results_df["F1"].idxmax()
    print(f"\n  Overall Best Model (by F1): {winner}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    main()
