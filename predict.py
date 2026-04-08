from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "breast_cancer_model.pkl"


def predict_sample(sample_index: int = 0) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please run main.py first to train and save the model."
        )

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target.map({0: "malignant", 1: "benign"})

    model = joblib.load(MODEL_PATH)
    sample = X.iloc[[sample_index]]
    prediction = model.predict(sample)[0]
    actual = y.iloc[sample_index]

    print("Sample index:", sample_index)
    print("Actual diagnosis   :", actual)
    print("Predicted diagnosis:", prediction)
    print("\nSample features:")
    print(pd.DataFrame(sample))


if __name__ == "__main__":
    predict_sample(0)
