import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("🏠 House Price Prediction System")
print("=" * 40)

# 1. Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target * 100000  # Convert to dollars

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# 2. Split data
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")

# 6. Save results
with open('01_model_results.txt', 'w') as f:
    f.write(f"House Price Prediction Results\n")
    f.write(f"==============================\n\n")
    f.write(f"Model: Linear Regression\n")
    f.write(f"Dataset: California Housing (20,640 samples)\n\n")
    f.write(f"Performance Metrics:\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"RMSE: ${rmse:,.2f}\n")
    f.write(f"MAE: ${mae:,.2f}\n\n")
    f.write(f"Features used: {list(X.columns)}\n")

# 7. Plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices ($)')
plt.ylabel('Predicted Prices ($)')
plt.title('Actual vs Predicted House Prices')
plt.savefig('02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Coefficient', y='Feature')
plt.title('Feature Importance - Linear Regression Coefficients')
plt.savefig('03_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Files saved: 01_model_results.txt, 02_actual_vs_predicted.png, 03_feature_importance.png")
print("🚀 Project complete! Ready for GitHub.")