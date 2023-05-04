import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import toolkit

# %%
###
# Notes
# roots are inside the unit circle then stable
# close to 1 marginally stable
# close to 9 stable
# cutoff, tailoff: decreasing to 0
# significant corr at lollipops in pacf
# cutoff in acf, tailoff in pacf: ar
# cutoff in pacf, tailoff in acf: ma



###

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


def wind_encode(s):
    if s == "SE":
        return int(1)
    elif s == "NE":
        return int(2)
    elif s == "NW":
        return int(3)
    else:
        return int(4)


df["wnd_dir"] = df["wnd_dir"].apply(wind_encode)
print(df.info())

# Correlation Matrix
corr = df.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(corr, annot=True, cmap='BrBG')
plt.title('Correlation Heatmap', fontsize=18)
plt.show()

# split train-test 80-20
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['pollution'], shuffle=False, test_size=0.2)

print(f'Training set size: {len(X_train)} rows and {len(X_train.columns)+1} columns')
print(f'Testing set size: {len(X_test)} rows and {len(X_test.columns)+1} columns')

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
# toolkit.Cal_autocorr_plot(df['pollution'], lags=100, title='ACF Plot for Pollution')

# ACF/PACF plot raw data
toolkit.ACF_PACF_Plot(df['pollution'], lags=60)

# Stationarity Tests on raw data
toolkit.ADF_Cal(df['pollution'])
toolkit.kpss_test(df['pollution'])
toolkit.CalRollingMeanVarGraph(df, 'pollution')

# STL Decomposition
Pollution = pd.Series(df['pollution'].values, index = date, name = 'pollution')

STL_raw = STL(Pollution, period=24)
res = STL_raw.fit()

T = res.trend
S = res.seasonal
R = res.resid

plt.figure(figsize=(16, 8))
# This
fig = res.plot()
plt.show()

str_trend = max(0, 1-(np.var(R)/np.var(T+R)))
print(f'The strength of trend for this data set is {round(str_trend, 3)}.')

str_seasonality = max(0, 1-(np.var(R)/np.var(S+R)))
print(f'The strength of seasonality for this data set is {round(str_seasonality, 3)}.')

# Seasonal Differencing
s = 24
df['seasonal_d_o_1'] = toolkit.seasonal_differencing(df['pollution'], seasons=s)
#print(df[['pollution', 'seasonal_d_o_1']].head(60))

# Plotting dependent variable vs time
plt.figure(figsize=(16, 8))
plt.plot(list(df.index.values), df['seasonal_d_o_1'])
plt.xlabel('Time')
plt.ylabel('seasonal_d_o_1')
plt.title('Pollution over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Stationarity on seasonaly differenced data
toolkit.ACF_PACF_Plot(df['seasonal_d_o_1'][s:], lags=60)
toolkit.ADF_Cal(df['seasonal_d_o_1'][s:])
toolkit.kpss_test(df['seasonal_d_o_1'][s:])
toolkit.CalRollingMeanVarGraph(df[s:], 'seasonal_d_o_1')

# Deseasoned data

# STL Decomposition

toolkit.STL_decomposition(df[s:], 'seasonal_d_o_1')

#
# # Strongly seasonal. Partially stationarity. looking at local peaks (~) and overall decreasing residuals in ACF, stationarity in ADF
# # and rolling mean being stable for raw data. Need to check order by looking at PACF plot.
# # Depending on that, decide whether to difference or not probably need to use SARIMA. Multiplicative model.
#
# # DASH app. Different Tabs. Enter order. display ACF/PACF, ADF., etc.
#


# Doing a non-seasonal differencing after the seasonal differrencing
# Transforming data to make it stationary
df['diff_order_1'] = toolkit.differencing(df['seasonal_d_o_1'], s)

# Plotting dependent variable vs time
plt.figure(figsize=(16, 8))
plt.plot(list(df.index.values), df['diff_order_1'])
plt.xlabel('Time')
plt.ylabel('diff_order_1')
plt.title('Pollution over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Stationarity Tests on transformed data
toolkit.ACF_PACF_Plot(df['diff_order_1'][s+1:], lags=60)
toolkit.CalRollingMeanVarGraph(df[s+1:], 'diff_order_1')
print('ADF test on diff_order_1:-')
toolkit.ADF_Cal(df['diff_order_1'][s+1:])
print('KPSS test on diff_order_1:-')
toolkit.kpss_test(df['diff_order_1'][s+1:])

# diff

# STL Decomposition
toolkit.STL_decomposition(df[s+1:], 'diff_order_1')

# SARIMA model with MA on the right tail ticks.
# Seasonal diff with season=24 (hrs) followed by first order non-seasonal differencing
# %%
X = df.drop(['diff_order_1'], axis=1)
y = df['diff_order_1']

X = X[s+1:]
y = y[s+1:]
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

print(f'Training set size: {len(X_train)} rows and {len(X_train.columns)+1} columns')
print(f'Testing set size: {len(X_test)} rows and {len(X_test.columns)+1} columns')

# %%

# # from statsmodels.tsa.holtwinters import ExponentialSmoothing
# #
# # model_holt_winters = ExponentialSmoothing(df['diff_order_1'], seasonal_periods=s).fit(optimized=True)
# # #forecasts_holt_winters = model_holt_winters.forecast(len(test))
# #
#
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# model = ExponentialSmoothing(df['diff_order_1'][s+1:]).fit()
#
# # make predictions
# pred = model.predict(start='2023-05-01', end='2023-12-31')

# # df['rev_seasonal_d_o_1'] = [np.nan] * s + toolkit.rev_differencing(df['diff_order_1'][s+1:], df['seasonal_d_o_1'][s], order=1)
# # print(df.head(50))
# # print('-----')
# # print(df.tail(50))
# # df['rev_pollution'] = toolkit.rev_seasonal_differencing(df['rev_seasonal_d_o_1'][s:], df['pollution'][:s], seasons=s)
# # print(df.head(50))
#

#
# # plot results
# plt.figure(figsize=(12,6))
# plt.plot(df.index, df['pollution'], label='Actual')
# plt.plot(pred.index, pred.values, label='Forecast')
# plt.legend()
# plt.title('Holt-Winters Forecast')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.show()

# %%


avg = toolkit.base_method('Average', y, len(y_train))
naive = toolkit.base_method('Naive', y, len(y_train))
drift = toolkit.base_method('Drift', y, len(y_train))
ses_point5 = toolkit.base_method('SES', y, len(y_train), 0.5)



# %%
lags=50
round_off = 3
ry = sm.tsa.stattools.acf(df['diff_order_1'][s+1:], nlags=lags)
toolkit.gpac(ry, j_max=10, k_max=10, round_off=round_off)


#
# # %%
# ############### Feature selection
# X_train = X_train.drop(['pollution', 'seasonal_d_o_1'], axis=1)
# model_ols = sm.OLS(y_train, X_train).fit()
# print(model_ols.summary())
#
# ##########
#
# # %%
#
# X_train = X_train.drop(['temp'], axis=1)
# model_ols = sm.OLS(y_train, X_train).fit()
# print(model_ols.summary())
#
# # %%
# X_train = X_train.drop(['wnd_spd'], axis=1)
# model_ols = sm.OLS(y_train, X_train).fit()
# print(model_ols.summary())
#
# # %%
# X_train = X_train.drop(['snow'], axis=1)
# model_ols = sm.OLS(y_train, X_train).fit()
# print(model_ols.summary())
#
# # %%
# from statsmodels.stats.outliers_influence import variance_inflation_factor
#
# #calculate VIF for each explanatory variable
# vif = pd.DataFrame()
# vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
# vif['variable'] = X_train.columns
#
# #view VIF for each explanatory variable
# print(vif)
