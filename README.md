# Rainfall-Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Load the dataset
df = pd.read_csv("rainfall_data.xlsx", encoding='latin1', on_bad_lines='skip')

# Show first 5 rows
print("\n--- Data Preview ---")
print(df.head())

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Fill or drop missing values
df = df.dropna()

# Show correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Features and Target
X = df[['Temperature', 'Humidity', 'WindSpeed', 'Pressure']]
y = df['Rainfall']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n--- Model Evaluation ---")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Scatter plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.show()

# Example new input
new_data = pd.DataFrame({
    'Temperature': [28],
    'Humidity': [85],
    'WindSpeed': [12],
    'Pressure': [1010]
})

prediction = model.predict(new_data)
print("\nPredicted Rainfall (mm):", prediction[0])
