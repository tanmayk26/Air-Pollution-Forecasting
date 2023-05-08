import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import toolkit

# %%
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
# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['pollution'], shuffle=False, test_size=0.2)
#
# print(f'Training set size: {len(X_train)} rows and {len(X_train.columns)+1} columns')
# print(f'Testing set size: {len(X_test)} rows and {len(X_test.columns)+1} columns')

# Plotting dependent variable vs time
# toolkit.plot_graph(x_value=df.index.values, y_value=df['pollution'], xlabel='Time', ylabel='Pollution', title='Pollution over Time')

# ACF/PACF plot raw data
# toolkit.ACF_PACF_Plot(df['pollution'], lags=60)

# Stationarity Tests on raw data
# print('ADF test on pollution:-')
# toolkit.ADF_Cal(df['pollution'])
# print('KPSS test on pollution:-')
# toolkit.kpss_test(df['pollution'])
# toolkit.CalRollingMeanVarGraph(df, 'pollution')

# STL Decomposition
# toolkit.STL_decomposition(df, 'pollution')

# Seasonal Differencing
s = 24
print(f'Performing Seasonal Differencing with interval={s}')

df['seasonal_d_o_1'] = toolkit.seasonal_differencing(df['pollution'], seasons=s)

# Plotting dependent variable vs time
# toolkit.plot_graph(x_value=df.index.values, y_value=df['seasonal_d_o_1'], xlabel='Time', ylabel='seasonal_d_o_1', title='Pollution over Time')

# ACF/PACF plot seasonaly differenced data
# toolkit.ACF_PACF_Plot(df['seasonal_d_o_1'][s:], lags=60)

# Stationarity on seasonaly differenced data
# print('ADF test on seasonal_d_o_1:-')
# toolkit.ADF_Cal(df['seasonal_d_o_1'][s:])
# print('KPSS test on seasonal_d_o_1:-')
# toolkit.kpss_test(df['seasonal_d_o_1'][s:])
# toolkit.CalRollingMeanVarGraph(df[s:], 'seasonal_d_o_1')

# STL Decomposition

# toolkit.STL_decomposition(df[s:], 'seasonal_d_o_1')

# Doing a non-seasonal differencing after the seasonal differrencing
# Transforming data to make it stationary
print('Performing Non-Seasonal Differencing with interval=1')
df['diff_order_1'] = toolkit.differencing(df['seasonal_d_o_1'], s)

# Plotting dependent variable vs time
# toolkit.plot_graph(x_value=df.index.values, y_value=df['diff_order_1'], xlabel='Time', ylabel='diff_order_1', title='Pollution over Time')

# ACF/PACF plot transformed data
toolkit.ACF_PACF_Plot(df['diff_order_1'][s+1:], lags=60)

# Stationarity Tests on transformed data
# toolkit.CalRollingMeanVarGraph(df[s+1:], 'diff_order_1')
# print('ADF test on diff_order_1:-')
# toolkit.ADF_Cal(df['diff_order_1'][s+1:])
# print('KPSS test on diff_order_1:-')
# toolkit.kpss_test(df['diff_order_1'][s+1:])

# STL Decomposition
# toolkit.STL_decomposition(df[s+1:], 'diff_order_1')


# %%

# Train-Test Split
df.index.freq = 'H'
df_train, df_test = train_test_split(df, shuffle=False, test_size=0.20)
print(f'Training set size: {len(df_train)} rows and {len(df_train.columns)+1} columns')
print(f'Testing set size: {len(df_test)} rows and {len(df_test.columns)+1} columns')

# Base Models

holt_winters = toolkit.base_method('Holt-Winters', df['pollution'], len(df_train))
df_err = toolkit.base_method('Average', df['diff_order_1'][s+1:], len(df_train))
naive = toolkit.base_method('Naive', df['diff_order_1'][s+1:], len(df_train))
drift = toolkit.base_method('Drift', df['diff_order_1'][s+1:], len(df_train))
ses_point5 = toolkit.base_method('SES', df['diff_order_1'][s+1:], len(df_train), 0.5)

df_err = pd.concat([df_err, naive], ignore_index=True)
df_err = pd.concat([df_err, drift], ignore_index=True)
df_err = pd.concat([df_err, ses_point5], ignore_index=True)
df_err = pd.concat([df_err, holt_winters], ignore_index=True)

print('===Model Comparison===\n', df_err)


# %%
df_2 = df[s+1:].drop(['pollution', 'seasonal_d_o_1'], axis=1)

# corr = df_2.corr()
# plt.figure(figsize=(16, 8))
# sns.heatmap(corr, annot=True, cmap='BrBG')
# plt.title('Correlation Heatmap', fontsize=18)
# plt.show()

X = df_2.drop(['diff_order_1'], axis=1)
y = df_2['diff_order_1']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
col_list = X_train.columns.to_list()

print(f'Training set size: {len(X_train)} rows and {len(X_train.columns)+1} columns')
print(f'Testing set size: {len(X_test)} rows and {len(X_test.columns)+1} columns')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=col_list)
X_test = pd.DataFrame(X_test, columns=col_list)

X1 = X_train.to_numpy()
H = np.dot(X1.T, X1)
u, s, vh = np.linalg.svd(H)
print('Singular values =', s)
print('Condition number is', round(np.linalg.cond(X1), 2))

from statsmodels.stats.outliers_influence import variance_inflation_factor

#calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['variable'] = X_train.columns

#view VIF for each explanatory variable
print(vif)


# %%
############### Feature selection
X_train_t = sm.add_constant(X_train, prepend=True)
X_test_t = sm.add_constant(X_test, prepend=True)

model_ols = sm.OLS(list(y_train), X_train_t).fit()
print(model_ols.summary())

# %%

X_train_t = X_train_t.drop(['temp'], axis=1)
model_ols = sm.OLS(list(y_train), X_train_t).fit()
print(model_ols.summary())

# %%
X_train_t = X_train_t.drop(['snow'], axis=1)
model_ols = sm.OLS(list(y_train), X_train_t).fit()
print(model_ols.summary())



# Final Model
# %%
X_train = sm.add_constant(X_train, prepend=True)
X_test = sm.add_constant(X_test, prepend=True)

model_ols = sm.OLS(list(y_train), X_train).fit()
print(model_ols.summary())


y_pred = model_ols.predict(X_train)
X_test = X_test[X_train.columns.to_list()]
y_forecast = model_ols.predict(X_test)

df_final = pd.DataFrame(list(zip(pd.concat([y_train, y_test], axis=0), pd.concat([y_pred, y_forecast], axis=0))), columns=['y', 'y_pred'])
toolkit.plot_forecast(df_final, len(y_train), title_body='Forecast using OLS method', xlabel='Time', ylabel='Pollution')
e, e_sq, MSE_train, VAR_train, MSE_test, VAR_test, mean_res_train = toolkit.cal_errors(df_final['y'].to_list(), df_final['y_pred'].to_list(), len(y_train), 0)
lags=50


method_name = 'Multi-linear Regression'
q_value = toolkit.Cal_q_value(e[:len(y_train)], lags, len(y_train), 2)
print('Error values using {} method'.format(method_name))
print('MSE Prediction data: ', round(MSE_train, 2))
print('MSE Forecasted data: ', round(MSE_test, 2))
print('Variance Prediction data: ', round(VAR_train, 2))
print('Variance Forecasted data: ', round(VAR_test, 2))
print('mean_res_train: ', round(mean_res_train, 2))
print('Q-value: ', round(q_value, 2))
l_err = [[method_name, MSE_train, MSE_test, VAR_train, VAR_test, mean_res_train, q_value]]
print(l_err)
df_err2 = pd.DataFrame(l_err, columns=['method_name', 'MSE_train', 'MSE_test', 'VAR_train', 'VAR_test', 'mean_res_train', 'Q-value'])
df_err = pd.concat([df_err, df_err2], ignore_index=True)

title='ACF Plot for errors - OLS method'
toolkit.Cal_autocorr_plot(e[:len(y_train)], lags, title)

print('T-Test')
print(model_ols.pvalues)
print('\nF-Test')
print(model_ols.f_pvalue)


print(df_err)



# %%
lags = 50
round_off = 3
ry = sm.tsa.stattools.acf(y_train, nlags=lags)
toolkit.gpac(ry, j_max=12, k_max=12, round_off=round_off)


# %%
model = sm.tsa.ARIMA(df['pollution'][:len(df_train)], order=(0,0,0), seasonal_order=(0,1,0,24))
model_fit = model.fit()
print(model_fit.summary())
y_result_hat = model_fit.predict()
y_result_h_t = model_fit.forecast(steps=len(df_test))
res_e = df['pollution'][:len(df_train)] - y_result_hat
fore_error = df['pollution'][len(df_train):] - y_result_h_t

print(f'variance of forecast errors is = {np.var(fore_error):.2f}')
print(f'variance of residual errors is = {np.var(res_e):.2f}')
print(f'Variance of forecast errors vs variance of Residual errors: {np.var(fore_error)/np.var(res_e):.2f}')


# %%
model = sm.tsa.ARIMA(df['pollution'][:len(df_train)], order=(0,1,0), seasonal_order=(0,0,1,24))
model_fit = model.fit()
print(model_fit.summary())
y_result_hat = model_fit.predict()
y_result_h_t = model_fit.forecast(steps=len(df_test))
res_e = df['pollution'][:len(df_train)] - y_result_hat
fore_error = df['pollution'][len(df_train):] - y_result_h_t

print(f'variance of forecast errors is = {np.var(fore_error):.2f}')
print(f'variance of residual errors is = {np.var(res_e):.2f}')
print(f'Variance of forecast errors vs variance of Residual errors: {np.var(fore_error)/np.var(res_e):.2f}')


# %%
na = 0
nb = 1
model_name = f'ARMA({na}, {nb})'

theta, sse, var_error, covariance_theta_hat, sse_list = toolkit.lm_step3(df['pollution'], na, nb)

theta2 = np.array(theta).reshape(-1)
for i in range(na + nb):
    if i < na:
        print('The AR coefficient {} is: {:.3f}'.format(i + 1, np.round(theta2[i], 3)))
    else:
        print('The MA coefficient {} is: {:.3f}'.format(i + 1 - na, np.round(theta2[i], 3)))

toolkit.lm_confidence_interval(theta, covariance_theta_hat, na, nb, round_off=round_off)

print("Estimated Covariance Matrix of estimated parameters:\n", np.round(covariance_theta_hat, decimals=round_off))

print("Estimated variance of error:", round(var_error, round_off))

print(toolkit.lm_find_roots(theta, na, round_off=round_off))

toolkit.plot_sse(sse_list, '')


plt.plot(df['pollution'][len(df_train):].index.values, df['pollution'][len(df_train):], label='Test')
plt.plot(df['pollution'][len(df_train):].index.values, y_result_h_t, label='h-Step Predictions')
plt.legend()
plt.title('Test vs H-Step Predictions graph')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()




