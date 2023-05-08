import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import statsmodels.api as sm
import copy

from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import STL


# np.abs(np.roots(ar))
# This process in non-stationary as some roots lie on 1. It should be less than 1

np.random.seed(6313)

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


def rev_differencing(series, first_obs, order=1):
    series.reset_index(inplace=True, drop=True)
    rev_diff = []
    for i in range(order-1):
        rev_diff.append(np.nan)
    rev_diff.append(first_obs)
    for i in range(order, len(series)+1):
        rev_diff.append(series[i-1] + rev_diff[i-1])
    return rev_diff


# create a differenced series
def seasonal_differencing(series, seasons=1):
    diff = []
    for i in range(seasons):
        diff.append(np.nan)
    for i in range(seasons, len(series)):
        diff.append(series[i] - series[i - seasons])
    return diff


def rev_seasonal_differencing(series, first_obs, seasons=1):
    series.reset_index(inplace=True, drop=True)
    rev_diff = list(first_obs)
    for i in range(len(series)):
        rev_diff.append(series[i] + rev_diff[i])
    return rev_diff


# # df['rev_seasonal_d_o_1'] = [np.nan] * s + toolkit.rev_differencing(df['diff_order_1'][s+1:], df['seasonal_d_o_1'][s], order=1)
# # print(df.head(50))
# # print('-----')
# # print(df.tail(50))
# # df['rev_pollution'] = toolkit.rev_seasonal_differencing(df['rev_seasonal_d_o_1'][s:], df['pollution'][:s], seasons=s)
# # print(df.head(50))



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


def base_method(method_name, y, train_n, alpha=0.5):
    print(f'\n=== {method_name} Method ===\n')
    if method_name == 'Average':
        y_pred = average(y, train_n)
    elif method_name == 'Naive':
        y_pred = naive(y, train_n)
    elif method_name == 'Drift':
        y_pred = drift(y, train_n)
    elif method_name == 'SES':
        y_pred = SES(y, train_n, alpha)
    elif method_name == 'Holt-Winters':
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(y).fit()
        #model = ExponentialSmoothing(y, seasonal_periods=24).fit()
        #model = ExponentialSmoothing(y, seasonal_periods=24).fit()
        #model = ExponentialSmoothing(y, seasonal_periods=24).fit()
        #model = ExponentialSmoothing(y, seasonal_periods=24).fit()

        pred_train_holts = model.predict(start=0, end=(train_n - 1))
        pred_test_holts = model.forecast(steps=len(y)-train_n)
        y_pred = pred_train_holts + pred_test_holts
    e_avg, e_sq, MSE_train, VAR_train, MSE_test, VAR_test, mean_res_train = cal_errors(y, y_pred, train_n, 2)
    df = pd.DataFrame(list(zip(y, y_pred, e_avg, e_sq)), columns=['y', 'y_pred', 'e', 'e^2'])
    print(df)
    if method_name == 'SES':
        title_suffix = 'with alpha={}'.format(alpha)
        plot_forecast(df, train_n, method_name, 'Yes', title_suffix)
    else:
        plot_forecast(df, train_n, method_name)
    lags = 50
    q_value = Cal_q_value(e_avg[:train_n], lags, train_n, 2)
    print('Error values using {} method'.format(method_name))
    print('MSE Prediction data: ', round(MSE_train, 2))
    print('MSE Forecasted data: ', round(MSE_test, 2))
    print('Variance Prediction data: ', round(VAR_train, 2))
    print('Variance Forecasted data: ', round(VAR_test, 2))
    print('mean_res_train: ', round(mean_res_train, 2))
    print('Q-value: ', round(q_value, 2))
    l_err = [[method_name, MSE_train, MSE_test, VAR_train, VAR_test, mean_res_train, q_value]]
    print(l_err)
    df_err = pd.DataFrame(l_err, columns=['method_name', 'MSE_train', 'MSE_test', 'VAR_train', 'VAR_test', 'mean_res_train', 'Q-value'])
    return df_err


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


def ar2_loop_method(N, coef, mean=0, std=1):
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


def ar2_dslim_method(N, order, coef, mean=0, std=1):
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


def ma2_loop_method(N, coef, mean=0, std=1):
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


def ma2_dslim_method(N, order, coef, mean=0, std=1):
    np.random.seed(6313)
    e = np.random.normal(mean, std, N)
    num = [1] + coef
    den = [1] + [0] * order
    system = (num, den, 1)
    t, y_dlsim = signal.dlsim(system, e)
    return y_dlsim.reshape(-1), e


def dslim_method(N, num, den, mean=0, std=1, seed=6313):
    np.random.seed(seed)
    e = np.random.normal(mean, std, N)
    system = (num, den, 1)
    t, y_dlsim = signal.dlsim(system, e)
    return y_dlsim.reshape(-1), e


def arma_input():
    N = int(input('Number of observations:'))
    mean_e = int(input('Enter the mean of white noise:'))
    var_e = int(input('Enter the variance of white noise:'))
    na = int(input('Enter AR order:'))
    nb = int(input('Enter MA order:'))
    den = []
    for i in range(1, na + 1):
        den.append(float(input(f'Enter the coefficient {i} of AR process:')))
    num = []
    for i in range(1, nb + 1):
        num.append(float(input(f'Enter the coefficient {i} of MA process:')))
    max_order = max(na, nb)
    ar_coef = [0] * (max_order)
    ma_coef = [0] * (max_order)
    for i in range(na):
        ar_coef[i] = den[i]
    for i in range(nb):
        ma_coef[i] = num[i]
    ar_params = np.array(ar_coef)
    ma_params = np.array(ma_coef)
    ar = np.r_[1, ar_params]
    ma = np.r_[1, ma_params]
    print('AR coef:', ar)
    print('MA coef:', ma)
    return N, na, nb, ar, ma, mean_e, var_e


def ACF_PACF_Plot(y,lags):
    #acf = sm.tsa.stattools.acf(y, nlags=lags)
    #pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


def gpac(ry, show_heatmap='Yes', j_max=7, k_max=7, round_off=3, seed=6313):
    np.random.seed(seed)
    gpac_table = np.zeros((j_max, k_max-1))
    for j in range(0, j_max):
        for k in range(1, k_max):
            phi_num = np.zeros((k, k))
            phi_den = np.zeros((k, k))
            for x in range(0, k):
                for z in range(0, k):
                    phi_num[x][z] = ry[abs(j + 1 - z + x)]
                    phi_den[x][z] = ry[abs(j - z + x)]
            phi_num = np.roll(phi_num, -1, 1)
            det_num = np.linalg.det(phi_num)
            det_den = np.linalg.det(phi_den)
            if det_den != 0 and not np.isnan(det_den):
                phi_j_k = det_num / det_den
            else:
                phi_j_k = np.nan
            gpac_table[j][k - 1] = phi_j_k #np.linalg.det(phi_num) / np.linalg.det(phi_den)
    if show_heatmap=='Yes':
        plt.figure(figsize=(16, 8))
        x_axis_labels = list(range(1, k_max))
        sns.heatmap(gpac_table, annot=True, xticklabels=x_axis_labels, fmt=f'.{round_off}f', vmin=-0.1, vmax=0.1)#, cmap='BrBG'
        plt.title(f'GPAC Table', fontsize=18)
        plt.show()
    #print(gpac_table)
    return gpac_table



def arma_gpac_pacf(ar_order=0, ma_order=0, num=None, den=None, N=1000, mean_e=0, var_e=1, lags=15, j_max=7, k_max=7, round_off=2, seed=6313, user_ip='Yes'):
    # if user_ip=='Yes':
    #     N = int(input('Enter the number of data samples:'))
    #     mean_e = int(input('Enter the mean of white noise:'))
    #     var_e = int(input('Enter the variance of white noise:'))
    #     ar_order = int(input('Enter AR order:'))
    #     ma_order = int(input('Enter MA order:'))
    #     den = []
    #     for i in range(1, ar_order + 2):
    #         den.append(float(input(f'Enter the coefficient {i} of AR process:')))
    #     num = []
    #     for i in range(1, ma_order + 2):
    #         num.append(float(input(f'Enter the coefficient {i} of MA process:')))
    # np.random.seed(seed)
    # max_order = max(ar_order, ma_order)
    # ar_coef = [0] * (max_order + 1)
    # ma_coef = [0] * (max_order + 1)
    # for i in range(ar_order+1):
    #     ar_coef[i] = den[i]
    # for i in range(ma_order+1):
    #     ma_coef[i] = num[i]
    # ar_params = np.array(ar_coef)
    # ma_params = np.array(ma_coef)
    # ar = np.r_[ar_params]
    # ma = np.r_[ma_params]
    # print('AR coef:', ar)
    # print('MA coef:', ma)
    N, na, nb, ar, ma, mean_e, var_e = arma_input()
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    mean_y = mean_e*(1 + np.sum(ar))/(1 + np.sum(ma))
    y = arma_process.generate_sample(N, scale=np.sqrt(var_e)) + mean_y
    print('ARMA Process:', list(np.around(np.array(y[:15]), round_off)))
    ry = arma_process.acf(lags=lags)
    print('ACF:', list(np.around(np.array(ry[:15]),round_off)))
    gpac(ry, j_max=j_max, k_max=k_max, round_off=round_off)
    ACF_PACF_Plot(y, lags=20)


# LM algorithm

def lm_cal_e(y, na, theta, seed=6313):
    np.random.seed(seed)
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):  # matching len of num and den
        for x in range(len(den) - len(num)):
            num = np.append(num, 0)
    elif len(num) > len(den):
        for x in range(len(num) - len(den)):
            den = np.append(den, 0)
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    return e


def lm_step1(y, na, nb, delta, theta):
    n = na + nb
    e = lm_cal_e(y, na, theta)
    sse_old = np.dot(np.transpose(e), e)
    X = np.empty(shape=(len(y), n))
    for i in range(0, n):
        theta[i] = theta[i] + delta
        e_i = lm_cal_e(y, na, theta)
        x_i = (e - e_i) / delta
        X[:, i] = x_i[:, 0]
        theta[i] = theta[i] - delta
    A = np.dot(np.transpose(X), X)
    g = np.dot(np.transpose(X), e)
    return A, g, X, sse_old


def lm_step2(y, na, A, theta, mu, g):
    delta_theta = np.matmul(np.linalg.inv(A + (mu * np.identity(A.shape[0]))), g)
    theta_new = theta + delta_theta
    e_new = lm_cal_e(y, na, theta_new)
    sse_new = np.dot(np.transpose(e_new), e_new)
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    return sse_new, delta_theta, theta_new


def lm_step3(y, na, nb):
    N = len(y)
    n = na+nb
    mu = 0.01
    mu_max = 10 ** 20
    max_iterations = 500
    delta = 10 ** -6
    var_error = 0
    covariance_theta_hat = 0
    sse_list = []
    theta = np.zeros(shape=(n, 1))

    for iterations in range(max_iterations):
        A, g, X, sse_old = lm_step1(y, na, nb, delta, theta)
        sse_new, delta_theta, theta_new = lm_step2(y, na, A, theta, mu, g)
        sse_list.append(sse_old[0][0])
        if iterations < max_iterations:
            if sse_new < sse_old:
                if np.linalg.norm(np.array(delta_theta), 2) < 10 ** -3:
                    theta_hat = theta_new
                    var_error = sse_new / (N - n)
                    covariance_theta_hat = var_error * np.linalg.inv(A)
                    print(f"Convergence Occured in {iterations} iterations")
                    break
                else:
                    theta = theta_new
                    mu = mu / 10
            while sse_new >= sse_old:
                mu = mu * 10
                if mu > mu_max:
                    print('No Convergence')
                    break
                sse_new, delta_theta, theta_new = lm_step2(y, na, A, theta, mu, g)
        if iterations > max_iterations:
            print('Max Iterations Reached')
            break
        theta = theta_new
    return theta_new, sse_new, var_error[0][0], covariance_theta_hat, sse_list


def lm_confidence_interval(theta, cov, na, nb, round_off=4):
    print("Confidence Interval for the estimated parameter(s)")
    lower_bound = []
    upper_bound = []
    for i in range(len(theta)):
        lower_bound.append(theta[i] - 2 * np.sqrt(cov[i, i]))
        upper_bound.append(theta[i] + 2 * np.sqrt(cov[i, i]))
    lower_bound = np.round(lower_bound, decimals=round_off)
    upper_bound = np.round(upper_bound, decimals=round_off)
    for i in range(na+nb):
        if i < na:
            print(f"AR Coefficient {i + 1}: ({lower_bound[i][0]}, {upper_bound[i][0]})")
        else:
            print(f"MA Coefficient {i + 1 - na}: ({lower_bound[i][0]}, {upper_bound[i][0]})")


def lm_find_roots(theta, na, round_off=4):
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):
        for x in range(len(den) - len(num)):
            num = np.append(num, 0)
    elif len(num) > len(den):
        for x in range(len(num) - len(den)):
            den = np.append(den, 0)
    else:
        pass
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    print(np.roots(num))
    print(np.roots(den))
    print("Roots of numerator:", np.round(np.roots(num), decimals=round_off))
    print("Roots of denominator:", np.round(np.roots(den), decimals=round_off))


def plot_sse(sse_list, model_name):
    plt.plot(sse_list)
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.title(f'SSE Learning Rate {model_name}')
    plt.xticks(np.arange(0, len(sse_list), step=1))
    plt.show()


def STL_decomposition(data, column_name):
    series = pd.Series(data[column_name].values, index = data.index.values, name = column_name)
    STL_tf = STL(series, period=24)
    res = STL_tf.fit()
    T = res.trend
    S = res.seasonal
    R = res.resid
    plt.figure(figsize=(16, 8))
    # This
    fig = res.plot()
    plt.show()
    str_trend = max(0, 1-(np.var(R)/np.var(T+R)))
    print(f'The strength of trend for seasonal_d_o_1 is {round(str_trend, 3)}.')
    str_seasonality = max(0, 1-(np.var(R)/np.var(S+R)))
    print(f'The strength of seasonality for seasonal_d_o_1 is {round(str_seasonality, 3)}.')


def plot_graph(x_value, y_value, xlabel='Time', ylabel='Magnitude', title='Samples over Time'):
    # Plotting dependent variable vs time
    plt.figure(figsize=(16, 8))
    plt.plot(list(x_value), y_value)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

