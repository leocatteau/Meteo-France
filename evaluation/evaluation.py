import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import display
import statsmodels.api as sm

def statistical_comparison(clean_dataset, imputed_dataset, station = None):
    clean_data = pd.DataFrame(clean_dataset, columns=[f'Station_{i}' for i in range(clean_dataset.shape[1])])
    reconstructed_data = pd.DataFrame(imputed_dataset, columns=[f'Station_{i}' for i in range(imputed_dataset.shape[1])])

    if station is not None:
        selected_station = station
    else:
        selected_station = np.random.choice(clean_data.columns)

    original_stats = clean_data[selected_station].describe()
    imputed_stats = reconstructed_data[selected_station].describe()

    stats_comparison = pd.DataFrame({
        'Metric': original_stats.index,
        'Original Data': original_stats.values,
        'Imputed Data (GRIN)': imputed_stats.values
    })
    display(stats_comparison)

    return stats_comparison

def autocorrelation(clean_dataset, imputed_dataset, station = None, lags=100):
    clean_data = pd.DataFrame(clean_dataset, columns=[f'Station_{i}' for i in range(clean_dataset.shape[1])])
    reconstructed_data = pd.DataFrame(imputed_dataset, columns=[f'Station_{i}' for i in range(imputed_dataset.shape[1])])

    if station is not None:
        selected_station = station
    else:
        selected_station = np.random.choice(clean_data.columns)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sm.graphics.tsa.plot_acf(clean_data[selected_station].dropna(), lags=lags, ax=plt.gca(), title="ACF of Original Data")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    sm.graphics.tsa.plot_acf(reconstructed_data[selected_station], lags=lags, ax=plt.gca(), title="ACF of Imputed Data")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def seasonal_analysis(clean_dataset, imputed_dataset, station = None, period = 24):
    clean_data = pd.DataFrame(clean_dataset, columns=[f'Station_{i}' for i in range(clean_dataset.shape[1])])
    reconstructed_data = pd.DataFrame(imputed_dataset, columns=[f'Station_{i}' for i in range(imputed_dataset.shape[1])])

    if station is not None:
        selected_station = station
    else:
        selected_station = np.random.choice(clean_data.columns)

    original_series = clean_data[selected_station]
    imputed_series = reconstructed_data[selected_station]

    original_decompose = seasonal_decompose(original_series.interpolate(), model='additive', period=period)  
    imputed_decompose = seasonal_decompose(imputed_series.interpolate(), model='additive', period=period)

    plt.figure(figsize=(10, 5))
    plt.plot(original_decompose.trend, label='Original Trend', color='blue')
    plt.plot(imputed_decompose.trend, label='Imputed Trend', color='green', linestyle='--')
    plt.title('Trend Comparison: Original vs. Imputation')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(original_decompose.seasonal, label='Original Seasonality', color='blue')
    plt.plot(imputed_decompose.seasonal, label='Imputed Seasonality', color='green', linestyle='--')
    plt.xlim(0, 250)
    plt.title('Seasonality Comparison: Original vs. Imputation')
    plt.legend()
    plt.grid(True)
    plt.show()