from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset import CATEGORY_LIST
from .features import load_folders_and_build_features


def train_random_forest(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"[INFO] Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("\n=== Confusion Matrix (RF, after aug) ===")
    print(confusion_matrix(y_val, y_pred, labels=CATEGORY_LIST))

    print("\n=== Classification Report (RF, after aug) ===")
    print(classification_report(y_val, y_pred, target_names=CATEGORY_LIST))

    return clf


def main():
    folders = ["augmentation_data1", "data2"]

    X, y = load_folders_and_build_features(folders)
    print("[INFO] Feature table shape:", X.shape)

    _ = train_random_forest(X, y)


if __name__ == "__main__":
    main()