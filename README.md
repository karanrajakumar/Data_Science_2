# Data_Science_2
time_series_forecasting

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = '/content/Alcohol_Sales.csv'  # Update this if the path changes in Colab
data = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')

# Display the first few rows of the dataset
print("Dataset Head:\n", data.head())

# Plot the time series data
plt.figure(figsize=(10, 5))
plt.plot(data, label='Alcohol Sales')
plt.title('Alcohol Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Splitting the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fitting the ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # You can tune the order (p, d, q)
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=len(test))

# Evaluate the model
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs forecasted values
plt.figure(figsize=(10, 5))
plt.plot(test, label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('Actual vs Forecasted Values')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

