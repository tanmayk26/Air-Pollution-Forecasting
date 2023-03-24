import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
plt.plot(list(df.index.values), df['pollution'])
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.title('Pollution over Time')
plt.legend()
plt.tight_layout()
plt.show()

# ACF Plot
toolkit.Cal_autocorr_plot(df['pollution'], 100)

# Correlation Matrix
corr = df.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(corr, annot=True, cmap='BrBG')
plt.title('Correlation Heatmap', fontsize=18)
plt.show()


