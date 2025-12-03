from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import CATEGORY_LIST
from .features import load_folders_and_build_features


def train_svm(
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

    clf = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=False,
            random_state=random_state,
        ),
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("\n=== Confusion Matrix (SVM, after aug) ===")
    print(confusion_matrix(y_val, y_pred, labels=CATEGORY_LIST))

    print("\n=== Classification Report (SVM, after aug) ===")
    print(classification_report(y_val, y_pred, target_names=CATEGORY_LIST))

    return clf


def main():
    folders = ["augmentation_data1", "data2"]

    X, y = load_folders_and_build_features(folders)
    print("[INFO] Feature table shape:", X.shape)

    _ = train_svm(X, y)


if __name__ == "__main__":
    main()