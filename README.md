# House Prediction System

A starter Python project for house price prediction using scikit-learn.

## What it includes

- `src/train.py` — trains a regression model using the California housing dataset
- `src/predict.py` — loads the saved model and makes predictions
- `src/data.py` — dataset loading and feature preparation
- `src/model.py` — model creation, saving, and evaluation
- `requirements.txt` — Python dependencies

## Quick start

1. Create a new Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model:

```powershell
python src\train.py
```

3. Make a prediction using the saved model:

```powershell
python src\predict.py
```

## Next steps

- Replace the built-in sample dataset with your own CSV data
- Add feature engineering for your house dataset
- Build a web or API interface for real-time predictions
