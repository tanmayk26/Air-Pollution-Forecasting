import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import statsmodels.api as sm
import copy

from scipy import signal

from statsmodels.tsa.seasonal import STL

# The following function calculates the Rolling Mean and Rolling Variance
def CalRollingMeanVar(dataset, column_name):
    for i in range(1, len(dataset)):
        return np.mean(dataset[column_name].head(i)), np.var(dataset[column_name].head(i))


# The following function calculates the Rolling Mean and Rolling Variance and subsequently plots the graph
def CalRollingMeanVarGraph(dataset, column_name):
    df_plot = pd.DataFrame(columns=['Samples', 'Mean', 'Variance'])
    for i in range(1, len(dataset)):
        df_plot.loc[i] = [i, np.mean(dataset[column_name].head(i)), np.var(dataset[column_name].head(i))]
    plt.subplot(2, 1, 1)
    plt.plot(df_plot['Samples'], df_plot['Mean'], label='Rolling Mean')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean - {}'.format(column_name))
    plt.subplot(2, 1, 2)
    plt.plot(df_plot['Samples'], df_plot['Variance'], label='Rolling Variance')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance - {}'.format(column_name))
    plt.tight_layout()
    plt.show()


# create a differenced series
def differencing(series, order=1):
    diff = []
    for i in range(order):
        diff.append(np.nan)
    for i in range(order, len(series)):
        diff.append(series[i] - series[i - 1])
    return diff


# Augmented Dickey-Fuller test (for stationarity)
# For this test, we state the following Null hypothesis (H0) and alternative hypothesis (H1):
# H0: The time-series has a unit root, meaning it is non-stationary.
# H1: The time-series does not have a unit root, meaning it is stationary.
# When the test statistic is lower than the critical value shown, you reject the null hypothesis and infer that the time series is stationary.
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


# KPSS Test (for stationarity)
# For this test, we state the following Null hypothesis (H0) and alternative hypothesis (H1):
# H0: The process is trend stationary.
# H1: The series has a unit root (series is not stationary).
# the test statistic should be lesser than the provided critical values to not reject null hypothesis.
# regression options
# “c” : The data is stationary around a constant (default).
# “ct” : The data is stationary around a trend.
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


# Auto Correlation Function
def Cal_autocorr(y, lag):
    mean = np.mean(y)
    numerator = 0
    denominator = 0
    for t in range(0, len(y)):
        denominator += (y[t] - mean) ** 2
    for t in range(lag, len(y)):
        numerator += (y[t] - mean)*(y[t-lag] - mean)
    return numerator/denominator


# Auto Correlation Function graph
def Cal_autocorr_plot(y, lags, title='ACF Plot', plot_show='Yes'):
    ryy = []
    ryy_final = []
    lags_final = []
    for lag in range(0, lags+1):
        ryy.append(Cal_autocorr(y, lag))
    ryy_final.extend(ryy[:0:-1])
    ryy_final.extend(ryy)
    lags = list(range(0, lags+1, 1))
    lags_final.extend(lags[:0:-1])
    lags_final = [value*(-1) for value in lags_final]
    lags_final.extend(lags)
    plt.figure(figsize=(12, 8))
    markers, stemlines, baseline = plt.stem(lags_final, ryy_final)
    plt.setp(markers, color='red', marker='o')
    plt.axhspan((-1.96 / np.sqrt(len(y))), (1.96 / np.sqrt(len(y))), alpha=0.2, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.tight_layout()
    if plot_show == 'Yes':
        plt.show()



def average(y, n):
    y_pred = list(np.nan for i in range(0, len(y)))
    for i in range(1, n):
        y_pred[i] = np.mean(y[:i])
    for i in range(n, len(y)):
        y_pred[i] = np.mean(y[:n])
    return y_pred


def naive(y, n):
    y_pred = list(np.nan for i in range(0, len(y)))
    for i in range(1, n):
        y_pred[i] = y[i-1]
    for i in range(n, len(y)):
        y_pred[i] = y[n - 1]
    return y_pred


def drift(y, n):
    y_pred = list(np.nan for i in range(0, len(y)))
    for i in range(2, n):
        y_pred[i] = y[i-1] + ((y[i-1]-y[0]))/(i-1)
    for i in range(n, len(y)):
        y_pred[i] = y[n-1] + (i+1-n)*(y[n-1]-y[0])/(n-1)
    return y_pred


def SES(y, n, alpha):
    y_pred = list(np.nan for i in range(0, len(y)))
    l0 = y[0]
    y_pred[1] = alpha * l0 + (1 - alpha) * l0
    for i in range(2, n):
        y_pred[i] = alpha * y[i-1] + (1-alpha) * y_pred[i-1]
    for i in range(n, len(y)):
        y_pred[i] = alpha * y[n-1] + (1-alpha) * y_pred[n-1]
    return y_pred


def cal_errors(y, y_pred, n, skip_first_n_obs=0):
    e = []
    e_sq = []
    for i in range(0, len(y)):
        if y_pred[i] != np.nan:
            e.append(y[i] - y_pred[i])
            e_sq.append((y[i] - y_pred[i]) ** 2)
        else:
            e.append(np.nan)
            e_sq.append(np.nan)
    MSE_train = np.nanmean(e_sq[skip_first_n_obs:n])
    VAR_train = np.nanvar(e[skip_first_n_obs:n])
    MSE_test = np.nanmean(e_sq[n:])
    VAR_test = np.nanvar(e[n:])
    mean_res_train = np.nanmean(e[skip_first_n_obs:n])
    return e, e_sq, MSE_train, VAR_train, MSE_test, VAR_test, mean_res_train


def plot_forecast(df, n, method_name='', plot_show='Yes', title_body='method & forecast', title_suffix='', xlabel='Time', ylabel='Magnitude'):
    plt.plot(list(df[:n].index.values + 1), df['y'][:n], label='Training dataset')
    plt.plot(list(df[n:].index.values + 1), df['y'][n:], label='Testing dataset', color='orange')
    plt.plot(list(df[n:].index.values + 1), df['y_pred'][n:], label='Forecast',  color='green')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('{} {} {}'.format(method_name, title_body, title_suffix))
    plt.legend()
    plt.tight_layout()
    if plot_show == 'Yes':
        plt.show()

# exclude first two observations for hw2. add a skip_first_n_obs parameter?
def Cal_q_value(e, lags, train_n, skip_first_obs=0):
    acf = 0
    data = [x for x in e[skip_first_obs:] if np.isnan(x) == False]
    for lag in range(1, lags + 1):
        acf += Cal_autocorr(data, lag) ** 2
    q_value = len(data) * (acf)
    return q_value


# LSE Beta B = inv(X.T*X)*X.T*Y
def LSE_Beta(X_train, y_train):
    X_train = sm.add_constant(X_train, prepend=True)
    return np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))


# Backward Step-wise Regression Function: looks at the p-value of each feature. Remove the feature with the highest p-value till there is no feature with p-value>0.05.
def backward_stepwise_regression(X_train, y_train):
    f_X_train = copy.deepcopy(X_train)
    f_y_train = copy.deepcopy(y_train)
    model_comp = {}
    feature_rmv = 'Overall'
    feature_rmv_list = []
    i = 1
    print('\n--- Overall ---\n')
    while i < len(f_X_train):
        model_ols = sm.OLS(f_y_train, f_X_train).fit()
        model_comp[feature_rmv] = [model_ols.aic, model_ols.bic, model_ols.rsquared_adj]
        stop = len(f_X_train.columns) - 2
        while len(f_X_train.columns) > stop:
            current_aic = model_ols.aic
            current_bic = model_ols.bic
            current_adjr2 = model_ols.rsquared_adj
            results = []
            for variable in f_X_train.columns[1:]:
                temp_f_X_train = f_X_train.drop(variable, axis=1)
                temp_model = sm.OLS(f_y_train, temp_f_X_train).fit()
                temp_aic = temp_model.aic
                temp_bic = temp_model.bic
                temp_adjr2 = temp_model.rsquared_adj
                results.append((variable, temp_aic, temp_bic, temp_adjr2))
            results.sort(key=lambda x: (x[1], x[2], -x[3]))
            feature_rmv, best_aic, best_bic, best_adjr2 = results[0]
            if (best_aic < current_aic) and (best_bic < current_bic) and (best_adjr2 > current_adjr2):
                feature_rmv_list.append(feature_rmv)
                print(model_ols.summary())
                f_X_train.drop(feature_rmv, axis=1, inplace=True)
                print(f'\n---Dropping {feature_rmv} ---\n')
            else:
                break
            break
        i += 1
    return model_comp, feature_rmv_list


def Moving_Average_raw(df_new, m, folding_order=0):
    df = copy.deepcopy(df_new)
    k = int((m - 1) / 2)
    even = (m+1)%2
    col_name = '{}-MA'.format(m)
    df[col_name] = np.nan
    for j in range(k, len(df)-k-even):
        for i in range(j-k, j+k+1+even):
            if np.isnan(df[col_name][j]):
                df[col_name][j] = df['Temp'][i]
            else:
                df[col_name][j] += df['Temp'][i]
    df[col_name] = round(df[col_name]/m, 3)
    if folding_order > 0:
        col_name_2 = '{}x{}-MA'.format(folding_order, m)
        df[col_name_2] = np.nan
        for j in range(k+1, len(df)-k-1):
            for i in range(j-1, j+1):
                if np.isnan(df[col_name_2][j]):
                    df[col_name_2][j] = df[col_name][i]
                else:
                    df[col_name_2][j] += df[col_name][i]
        df[col_name_2] = round(df[col_name_2] / folding_order, 3)
        df = df.loc[:, df.columns != col_name]
    return df




def Moving_Average(df_new, m, folding_order=0):
    df = copy.deepcopy(df_new)
    k = int((m - 1) / 2)
    even = (m+1)%2
    col_name = '{}-MA'.format(m)
    df[col_name] = np.nan
    for j in range(k, len(df)-k-even):
        df[col_name][j] = np.nansum(df['Temp'][j-k : j+k+1+even])
    df[col_name] = round(df[col_name]/m, 3)
    if folding_order > 0:
        col_name_2 = '{}x{}-MA'.format(folding_order, m)
        df[col_name_2] = np.nan
        for j in range(k+1, len(df)-k-1):
            df[col_name_2][j] = np.nansum(df[col_name][j - 1: j + 1])
        df[col_name_2] = round(df[col_name_2] / folding_order, 3)
        df = df.loc[:, df.columns != col_name]
    return df






# def plot_STL(df, date, N):
#     series = pd.Series(df['Temp'].values, index=date,
#                      name='daily-temp')
#     STL = STL(series)
#     res = STL.fit()
#     T = res.trend
#     S = res.seasonal
#     R = res.resid
#     plt.figure()
#     plt.plot(T[:N].values, label='trend')
#     plt.plot(S[:N].values, label='Seasonal')
#     plt.plot(R[:N].values, label='residuals')
#     plt.plot(df['Temp'][:N].values, label='original data')
#     plt.legend()
#     plt.show()


#
# # Backward Step-wise Regression Function: looks at the p-value of each feature. Remove the feature with the highest p-value till there is no feature with p-value>0.05.
# def backward_stepwise_regression(X_train, y_train):
#     f_X_train = copy.deepcopy(X_train)
#     f_y_train = copy.deepcopy(y_train)
#     model_comp = {}
#     feature_rmv = 'Overall'
#     i = 1
#     while i < len(f_X_train):
#         model_ols = sm.OLS(f_y_train, f_X_train).fit()
#         model_comp[feature_rmv] = [model_ols.aic, model_ols.bic, model_ols.rsquared_adj]
#         print(model_ols.summary())
#         pvalue_comp = pd.DataFrame(list(zip(f_X_train.columns, model_ols.pvalues)), columns=['Feature', 'pvalue'])
#         pvalue_comp.drop(index=0, inplace=True)
#         pvalue_comp.sort_values(by=['pvalue'], ascending=False, inplace=True)
#         pvalue_comp.reset_index(drop=True, inplace=True)
#         print(pvalue_comp)
#         if pvalue_comp['pvalue'][0] > 0.05:
#             feature_rmv = pvalue_comp['Feature'][0]
#             f_X_train.drop(feature_rmv, axis=1, inplace=True)
#             print(f'\n---Dropping {feature_rmv} ---\n')
#             i += 1
#         else:
#             i += len(f_X_train)
#             print('The optimal feature selection is completed.')
#     return model_comp
#
#


def ar2_loop_method(N, coef, mean=1, std=1):
    np.random.seed(6313)
    e = np.random.normal(mean, std, N)
    y = np.zeros(len(e))
    for i in range(len(e)):
        if i == 0:
            y[0] = e[0]
        elif i == 1:
            y[i] = -coef[0]*y[i-1] + e[i]
        else:
            y[i] = -coef[0]*y[i-1] - coef[1]*y[i-2] + e[i]
    return y, e


def ar2_dslim_method(N, order, coef, mean=1, std=1):
    np.random.seed(6313)
    e = np.random.normal(mean, std, N)
    num = [1] + [0] * order
    den = [1] + coef
    system = (num, den, 1)
    t, y_dlsim = signal.dlsim(system, e)
    return y_dlsim.reshape(-1), e


def ar2_estimated_parameter_vals(y):
    y_t_1 = np.append(0, y[:-1])
    y_t_2 = np.append(0, y_t_1[:-1])
    X = np.vstack((y_t_1, y_t_2)).T
    return list(-np.array(LSE_Beta(X, y)))


def ma2_loop_method(N, coef, mean=1, std=1):
    np.random.seed(6313)
    e = np.random.normal(mean, std, N)
    y = np.zeros(len(e))
    for i in range(len(e)):
        if i == 0:
            y[0] = e[0]
        elif i == 1:
            y[i] = e[i] + coef[0]*e[i-1]
        else:
            y[i] = e[i] + coef[0]*e[i-1] + coef[1]*e[i-2]
    return y, e


def ma2_dslim_method(N, order, coef, mean=1, std=1):
    np.random.seed(6313)
    e = np.random.normal(mean, std, N)
    num = [1] + coef
    den = [1] + [0] * order
    system = (num, den, 1)
    t, y_dlsim = signal.dlsim(system, e)
    return y_dlsim.reshape(-1), e


