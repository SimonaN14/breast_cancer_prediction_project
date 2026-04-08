from train_model import train_model


def main() -> None:
    metrics = train_model()
    print("\nTraining finished successfully.")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
