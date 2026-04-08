from pathlib import Path
import joblib # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def prepare_dataset() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns={"target": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
    csv_path = DATA_DIR / "breast_cancer.csv"
    df.to_csv(csv_path, index=False)
    return df


def train_model() -> dict:
    ensure_directories()
    df = prepare_dataset()

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="malignant")
    recall = recall_score(y_test, predictions, pos_label="malignant")
    f1 = f1_score(y_test, predictions, pos_label="malignant")

    joblib.dump(model, MODELS_DIR / "breast_cancer_model.pkl")

    predictions_df = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": predictions
    })
    predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    with open(RESULTS_DIR / "metrics.txt", "w", encoding="utf-8") as file:
        file.write("Breast Cancer Prediction Metrics\n")
        file.write(f"Accuracy = {accuracy:.4f}\n")
        file.write(f"Precision = {precision:.4f}\n")
        file.write(f"Recall = {recall:.4f}\n")
        file.write(f"F1-score = {f1:.4f}\n")

    cm = confusion_matrix(y_test, predictions, labels=["benign", "malignant"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malignant"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    metrics = train_model()
    print("Model training completed.")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")
