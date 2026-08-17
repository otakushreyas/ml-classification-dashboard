"""Quick script to generate test_data.csv from UCI Wine Quality dataset."""
import sys
import traceback

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red = pd.read_csv(red_url, sep=";")
    white = pd.read_csv(white_url, sep=";")
    red["wine_type"] = 0
    white["wine_type"] = 1
    df = pd.concat([red, white], ignore_index=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save UNSCALED test data (the Streamlit app will scale it internally)
    test_df = X_test.copy()
    test_df["quality"] = y_test.values
    test_df.to_csv("test_data.csv", index=False)

    with open("gen_status.txt", "w") as f:
        f.write(f"SUCCESS: {len(test_df)} rows, {len(test_df.columns)} cols\n")
        f.write(f"Columns: {test_df.columns.tolist()}\n")

except Exception as e:
    with open("gen_status.txt", "w") as f:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())
    sys.exit(1)
