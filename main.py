import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import toolkit

# to display all the columns
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

url = 'https://raw.githubusercontent.com/tanmayk26/Air-Pollution-Forecasting/main/LSTM-Multivariate_pollution.csv'
df = pd.read_csv(url, index_col='date')
date = pd.date_range(start='1/2/2010',
                     periods=len(df),
                     freq='H')
df.index = date
# Prints head and tail of the dataframe
print(df)

# Describe the data
print(f'Data basic statistics: \n{df.describe()}')

# Check NA value
print(f'NA value: \n{df.isna().sum()}')
# As we can see there are no null values

# Plotting dependent variable vs time
plt.figure(figsize=(16, 8))
plt.plot(list(df.index.values), df['pollution'])
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.title('Pollution over Time')
plt.legend()
plt.tight_layout()
plt.show()

# ACF Plot
toolkit.Cal_autocorr_plot(df['pollution'], lags=100)

# Correlation Matrix
corr = df.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(corr, annot=True, cmap='BrBG')
plt.title('Correlation Heatmap', fontsize=18)
plt.show()

# split train-test 80-20
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['pollution'], shuffle=False, test_size=0.2)

print(f'Training set size: {len(X_train)} rows and {len(X_train.columns)+1} columns')
print(f'Testing set size: {len(X_test)} rows and {len(X_test.columns)+1} columns')

# Stationarity Tests on raw data
toolkit.ADF_Cal(df['pollution'])
toolkit.kpss_test(df['pollution'])
toolkit.CalRollingMeanVarGraph(df, 'pollution')

# Transforming data to make it stationary
df['diff_order_1'] = toolkit.differencing(df['pollution'], 1)

# Stationarity Tests on transformed data
toolkit.CalRollingMeanVarGraph(df[1:], 'diff_order_1')
print('ADF test on diff_order_1:-')
toolkit.ADF_Cal(df['diff_order_1'][1:])
print('KPSS test on diff_order_1:-')
toolkit.kpss_test(df['diff_order_1'][1:])

# ACF Plot on transformed data
toolkit.Cal_autocorr_plot(df['diff_order_1'][1:], lags=20)

# Transforming data to make it stationary
df['diff_order_2'] = toolkit.differencing(df['diff_order_1'], 2)

# Stationarity Tests on transformed data
toolkit.CalRollingMeanVarGraph(df[2:], 'diff_order_2')
print('ADF test on diff_order_2:-')
toolkit.ADF_Cal(df['diff_order_2'][2:])
print('KPSS test on diff_order_2:-')
toolkit.kpss_test(df['diff_order_2'][2:])

# ACF Plot on transformed data
toolkit.Cal_autocorr_plot(df['diff_order_2'][2:], lags=20)

