import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess data
def preprocess_data(data):
    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    
    # Feature Engineering: Moving Average and Volatility
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Close'].rolling(window=20).std()
    
    # Drop rows with NaN values created by rolling calculations
    data.dropna(inplace=True)
    
    return data

# Function to detect anomalies using multiple methods
def detect_anomalies(data):
    prices = data['Close'].values.reshape(-1, 1)
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    data['Anomaly_ISO'] = iso_forest.fit_predict(prices_scaled)

    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20)
    data['Anomaly_LOF'] = lof.fit_predict(prices_scaled)

    # One-Class SVM
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    data['Anomaly_SVM'] = svm.fit_predict(prices_scaled)

    return data, scaler

# Function to plot the stock prices and anomalies
def plot_anomalies(data, ticker):
    plt.figure(figsize=(14, 12))
    
    # Define colors for better visibility
    colors = {
        'normal': 'blue',
        'anomaly_iso': 'red',
        'anomaly_lof': 'orange',
        'anomaly_svm': 'green'
    }

    # Plot Close Prices
    plt.subplot(4, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price', color=colors['normal'], linewidth=2)
    plt.title(f'{ticker} Stock Price', fontsize=16)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Plot Isolation Forest Anomalies
    plt.subplot(4, 1, 2)
    plt.plot(data.index, data['Close'], label='Close Price', color=colors['normal'], linewidth=2)
    anomalies_iso = data[data['Anomaly_ISO'] == -1]
    plt.scatter(anomalies_iso.index, anomalies_iso['Close'], color=colors['anomaly_iso'], label='Anomaly (ISO)', marker='o', s=100, edgecolor='black', alpha=0.7)
    plt.title('Anomalies Detected by Isolation Forest', fontsize=16)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Plot Local Outlier Factor Anomalies
    plt.subplot(4, 1, 3)
    plt.plot(data.index, data['Close'], label='Close Price', color=colors['normal'], linewidth=2)
    anomalies_lof = data[data['Anomaly_LOF'] == -1]
    plt.scatter(anomalies_lof.index, anomalies_lof['Close'], color=colors['anomaly_lof'], label='Anomaly (LOF)', marker='o', s=100, edgecolor='black', alpha=0.7)
    plt.title('Anomalies Detected by Local Outlier Factor', fontsize=16)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Plot One-Class SVM Anomalies
    plt.subplot(4, 1, 4)
    plt.plot(data.index, data['Close'], label='Close Price', color=colors['normal'], linewidth=2)
    anomalies_svm = data[data['Anomaly_SVM'] == -1]
    plt.scatter(anomalies_svm.index, anomalies_svm['Close'], color=colors['anomaly_svm'], label='Anomaly (SVM)', marker='o', s=100, edgecolor='black', alpha=0.7)
    plt.title('Anomalies Detected by One-Class SVM', fontsize=16)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Fetch stock data for a specific ticker
    ticker = 'DJT'  
    start_date = '2023-10-10'
    end_date = '2024-10-30'
    
    # Get the stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess the data
    processed_data = preprocess_data(stock_data)
    
    # Detect anomalies
    detected_anomalies, scaler = detect_anomalies(processed_data)
    
    # Plot the stock prices and anomalies
    plot_anomalies(detected_anomalies, ticker)
