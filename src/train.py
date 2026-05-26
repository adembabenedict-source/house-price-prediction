import os
from sklearn.model_selection import train_test_split
from src.data import load_sample_dataset, prepare_features
from src.model import build_model, evaluate_model, save_model

MODEL_PATH = os.path.join('models', 'house_price_model.joblib')


def main():
    df = load_sample_dataset()
    X, y = prepare_features(df, target_column='target')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print('Training complete. Evaluation metrics:')
    for name, value in metrics.items():
        print(f'  {name}: {value:.4f}')

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(model, MODEL_PATH)
    print(f'Model saved to: {MODEL_PATH}')


if __name__ == '__main__':
    main()
