from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset import CATEGORY_LIST          # 라벨 순서 통일용
from .features import load_folders_and_build_features


def train_random_forest(X, y, test_size: float = 0.2, random_state: int = 42):
    # confusion matrix / classification report 출력.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,          # 클래스 비율 유지
        random_state=random_state,
    )

    print(f"[INFO] Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_val, y_pred, labels=CATEGORY_LIST))

    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=CATEGORY_LIST))

    return clf


def main():
    # 1) 증강 전 데이터 feature 추출
    X, y = load_folders_and_build_features(["data1", "data2"])

    print("[INFO] Feature table shape:", X.shape)

    # 2) RF 학습 및 평가
    _ = train_random_forest(X, y)


if __name__ == "__main__":
    main()