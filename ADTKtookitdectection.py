import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.detector import ThresholdAD

# define the stock ticker and date range
ticker = 'DJT'  #using a volitaile stock like DJT for example
start_date = '2023-10-10'
end_date = '2024-10-30'

# Download the DJT stock data
data = yf.download(ticker, start=start_date, end = end_date)
data.reset_index(inplace=True)

# Inspect the initial columns
print("Initial DataFrame columns:")
print(data.columns)

#Flatten the Columns if they are a MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [f'{col[0]}_{col[1]}' for col in data.columns]

# Inspect the columns after flattening
print("Flattened DataFrame columns:")
print(data.columns)
# print the first few rows of the DataFrame
print("DataFrame head:")
print(data.head())

# Convert'Date_' column to datetime and set it as the index
data['Date_'] = pd.to_datetime(data['Date_'])  # Use the correct flattened date column
data.set_index('Date_', inplace=True)

# Calculate rolling quartiles
window_size = 20  
data['Q1']=data['Close_DJT'].rolling(window=window_size).quantile(0.25)
data['Q3']=data['Close_DJT'].rolling(window=window_size).quantile(0.75)

# Calculate IQR
data['IQR'] = data['Q3'] -data['Q1']

# Define upper and lower bounds for IQR
data['Lower Bound'] = data['Q1'] - 1.5 * data['IQR']
data['Upper Bound'] = data['Q3'] + 1.5 * data['IQR']

# Print the DataFrame to check if the columns were created successfully
print("Data after calculating Q1, Q3, IQR, Lower Bound, and Upper Bound: ")
print(data[['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound']].head(25))

# Drop NaN values (especially at the beginning due to rolling calculations)
data.dropna(subset=['Q1', 'Q3', 'Lower Bound', 'Upper Bound'], inplace=True)

# Detect anomalies using IQR
data['Anomaly_IQR'] = ((data['Close_DJT'] < data['Lower Bound']) | (data['Close_DJT'] > data['Upper Bound'])).astype(int)

# Validate the series for ThresholdAD
validated_data = validate_series(data['Close_DJT'])

 #Initialize the ThresholdAD detector with the latest bounds
latest_upper_bound = data['Upper Bound'].iloc[-1]
latest_lower_bound = data['Lower Bound'].iloc[-1]
threshold_ad =ThresholdAD(high=latest_upper_bound, low=latest_lower_bound)

# Fit the model and detect anomalies using ThresholdAD
anomalies_threshold_ad = threshold_ad.detect(validated_data)

# Visualization
plt.figure(figsize=(14,7))

# Plot stock prices
plt.plot(data.index, data['Close_DJT'], label='Stock Price', color='blue')

# Plot anomalies detected by IQR
plt.scatter(data[data['Anomaly_IQR'] == 1].index, 
            data[data['Anomaly_IQR'] == 1]['Close_DJT'], 
            color='red', label='Anomalies (IQR)', marker='o')

# Plot anomalies detected by ThresholdAD
plt.scatter(data[anomalies_threshold_ad].index, 
            data[anomalies_threshold_ad]['Close_DJT'], 
            color= 'yellow' , label='Anomalies (ThresholdAD)', marker='x')

# Plot upper and lower bounds
plt.plot(data.index, data['Upper Bound'], color='green', linestyle='--', label='Upper Bound (IQR)')
plt.plot(data.index, data['Lower Bound'], color='orange', linestyle='--', label='Lower Bound (IQR)')

# Title and labels
plt.title(f'Anomaly Detection in {ticker} Stock Prices: Moving IQR and using ADTK toolkit')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.tight_layout()  # Adjust layout for better spacing

# Show the plot
plt.show()
