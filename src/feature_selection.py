from pathlib import Path

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


PROCESSED_DATA_PATH = Path("data/processed/spy_features.csv")
SELECTED_FEATURES_PATH = Path("models/selected_features.csv")


def load_processed_data(path=PROCESSED_DATA_PATH):
    return pd.read_csv(path, parse_dates=["date"])


def get_feature_columns(df):
    excluded_columns = {
        "date",
        "target",
        "next_day_return",
    }

    return [col for col in df.columns if col not in excluded_columns]


def select_features_with_rfe(df, n_features=24):
    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df["target"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = LinearSVC(
        C=0.01,
        penalty="l2",
        dual=False,
        max_iter=10000,
        random_state=42,
    )

    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=1,
    )

    selector.fit(X_scaled, y)

    selected_features = [col for col, selected in zip(feature_cols, selector.support_) if selected]

    return selected_features


def save_selected_features(selected_features, path=SELECTED_FEATURES_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    selected_features_df = pd.DataFrame({"feature": selected_features})
    selected_features_df.to_csv(path, index=False)


def main():
    df = load_processed_data()
    selected_features = select_features_with_rfe(df, n_features=24)
    save_selected_features(selected_features)

    print("Selected features:")
    for feature in selected_features:
        print(f"- {feature}")

    print(f"Saved selected features to {SELECTED_FEATURES_PATH}")


if __name__ == "__main__":
    main()