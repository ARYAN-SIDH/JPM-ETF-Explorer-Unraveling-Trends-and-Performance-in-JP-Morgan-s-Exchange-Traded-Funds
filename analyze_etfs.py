# analyze_etfs.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load your JP Morgan ETF dataset
# Replace 'your_etf_data.csv' with the actual file path or URL
etf_data = pd.read_csv('your_etf_data.csv')

# Data Preprocessing
# (You may need to customize this based on your dataset)
etf_data['Date'] = pd.to_datetime(etf_data['Date'])
etf_data.set_index('Date', inplace=True)

# Calculate Daily Returns
etf_data['Daily_Return'] = etf_data['Close'].pct_change()

# Calculate Cumulative Returns
etf_data['Cumulative_Return'] = (1 + etf_data['Daily_Return']).cumprod() - 1

# Normalize Close Prices for Visualization
scaler = MinMaxScaler()
etf_data['Normalized_Close'] = scaler.fit_transform(etf_data['Close'].values.reshape(-1, 1))

# Visualize Cumulative Returns
plt.figure(figsize=(10, 6))
plt.plot(etf_data.index, etf_data['Cumulative_Return'], label='Cumulative Returns')
plt.title('JP Morgan ETF Cumulative Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Visualize Normalized Close Prices
plt.figure(figsize=(10, 6))
plt.plot(etf_data.index, etf_data['Normalized_Close'], label='Normalized Close Prices')
plt.title('JP Morgan ETF Normalized Close Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Normalized Close Prices')
plt.legend()
plt.show()

# Calculate and Print Key Metrics
average_daily_return = etf_data['Daily_Return'].mean()
volatility = etf_data['Daily_Return'].std()
sharpe_ratio = (average_daily_return / volatility) * (252**0.5)  # Assuming 252 trading days per year

print(f'Average Daily Return: {average_daily_return:.4f}')
print(f'Volatility (Standard Deviation of Daily Returns): {volatility:.4f}')
print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
