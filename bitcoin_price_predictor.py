from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

# Download Bitcoin data (BTC-USD ticker)
data = yf.download('BTC-USD', start='2015-01-01', end='2023-01-01')

# Resample data by month, taking the 'Close' price at the end of each month
monthly_data = data['Close'].resample('M').last()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data.values.reshape(-1, 1))

# Create dataset for month-to-month predictions
def create_monthly_dataset(data):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[i])
        y.append(data[i + 1])
    return np.array(X), np.array(y)

# Create X and y datasets
X, y = create_monthly_dataset(scaled_data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fix warning: Reshape y to 1D arrays using ravel
y_train = y_train.ravel()
y_test = y_test.ravel()

# Inverse transform y_test to get the actual Bitcoin prices
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Random Forest model
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
predictions_rf = scaler.inverse_transform(predictions_rf.reshape(-1, 1))

# Support Vector Regressor model
from sklearn.svm import SVR
model_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model_svr.fit(X_train, y_train)
predictions_svr = model_svr.predict(X_test)
predictions_svr = scaler.inverse_transform(predictions_svr.reshape(-1, 1))

# Gradient Boosting Regressor model
from sklearn.ensemble import GradientBoostingRegressor
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_gb.fit(X_train, y_train)
predictions_gb = model_gb.predict(X_test)
predictions_gb = scaler.inverse_transform(predictions_gb.reshape(-1, 1))

# Plot Actual vs Predicted for all models
plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Actual Price')

# Random Forest
plt.plot(predictions_rf, label='Random Forest Predictions')

# Support Vector Regressor
plt.plot(predictions_svr, label='SVR Predictions')

# Gradient Boosting Regressor
plt.plot(predictions_gb, label='Gradient Boosting Predictions')

plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate accuracy for Random Forest
r2_rf = r2_score(actual_prices, predictions_rf)
print("Random Forest Performance:")
print(f"R-squared (R²): {r2_rf}")
print()

# Calculate accuracy for Support Vector Regressor
r2_svr = r2_score(actual_prices, predictions_svr)
print("Support Vector Regressor Performance:")
print(f"R-squared (R²): {r2_svr}")
print()

# Calculate accuracy for Gradient Boosting
r2_gb = r2_score(actual_prices, predictions_gb)
print("Gradient Boosting Performance:")
print(f"R-squared (R²): {r2_gb}")
