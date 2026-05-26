from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


def build_model(random_state: int = 42):
    """Create and configure a regression model."""
    return RandomForestRegressor(n_estimators=100, random_state=random_state)


def evaluate_model(model, X_test, y_test):
    """Return evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
    }


def save_model(model, path: str):
    """Save the trained model to disk."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load a trained model from disk."""
    return joblib.load(path)
