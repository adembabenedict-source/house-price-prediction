import os
import pandas as pd
from src.model import load_model
from src.data import load_sample_dataset, prepare_features

MODEL_PATH = os.path.join('models', 'house_price_model.joblib')


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found. Run 'python src\\train.py' first to create {MODEL_PATH}"
        )

    model = load_model(MODEL_PATH)
    df = load_sample_dataset()
    X, _ = prepare_features(df, target_column='target')

    predictions = model.predict(X[:5])
    print('Sample predictions for the first 5 rows:')
    for index, value in enumerate(predictions, start=1):
        print(f'  row {index}: {value:.4f}')


if __name__ == '__main__':
    main()
