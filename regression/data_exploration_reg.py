#%% Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

#%%Uploading the data

df_train = pd.read_csv(os.path.join('data','trainReg.txt'))
df_test = pd.read_csv(os.path.join('data','testReg.txt'))

#%%Data exploration

#histogram
sns.distplot(df_train['V1'])

#skewness and kurtosis
print("Skewness: %f" % df_train['V1'].skew())
print("Kurtosis: %f" % df_train['V1'].kurt())

#Descriptive statistics summary
statistics = df_train.describe()

#Correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap="YlGnBu")
plt.show()

#5 most correlated variables
k = 5 #number of variables for heatmap
cols = corrmat.nlargest(k, 'V1')['V1'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['V1', 'V2', 'V6', 'V48', 'V21']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()

#%%Verifying the missing values

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data.Total > 0]

#%% Outliers

V1_array = df_train['V1'].values.reshape(-1, 1)
V1_scaled = StandardScaler().fit_transform(V1_array)
low_range = V1_scaled[V1_scaled[:,0].argsort()][:10]
high_range = V1_scaled[V1_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

