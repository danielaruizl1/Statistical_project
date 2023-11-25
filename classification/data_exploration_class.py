#%% Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

#%%Uploading the data

df_train = pd.read_excel(os.path.join('data','TrainClass.xlsx'))
df_train = df_train.set_index('CODIGO_EMPRESA')
df_test = pd.read_excel(os.path.join('data','TestClass.xlsx'))
df_description = pd.read_csv(os.path.join('data',"DataDes.txt"), sep = ":", header = None)
df_description = df_description[2:]
df_description.columns = ["Predictor", "Description"]
df_description = df_description.reset_index(drop=True)

#%%Data exploration

#Correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap="YlGnBu")
plt.show()

#5 most correlated variables
k = 5 #number of variables for heatmap
cols = corrmat.nlargest(k, 'FRACASO')['FRACASO'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Descriptive statistics summary
statistics = df_train.describe()

#%%Verifying the missing values

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data.Total > 0]

#%% Prepare data for modeling

#Normalizing the data
scaler = MinMaxScaler()
df_train[df_train.columns] = scaler.fit_transform(df_train)

#Separate the predictors from the target
df_train_X = df_train.drop(['FRACASO'], axis = 1) 
df_train_y = df_train['FRACASO']

