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

#%% Prepare data for modeling

#Separate the predictors from the target
df_train_X = df_train.drop(['V1'], axis = 1)
df_train_y = df_train['V1']
df_test_X = df_test.drop(['Id'], axis = 1)

#Normalizing the data
scaler = MinMaxScaler()
df_train_X[df_train_X.columns] = scaler.fit_transform(df_train_X)
df_test_X[df_test_X.columns] = scaler.fit_transform(df_test_X)

#%% Feature selection

# Using Recursive Feature Elimination to select the 20 most important features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=20)
fit = rfe.fit(df_train_X, df_train_y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % df_train_X.columns[fit.support_])

selected_features = df_train_X.columns[fit.support_]
X_train_selected = df_train_X[selected_features]
X_test_selected = df_test_X[selected_features]

#%% Using SelectKBest to select the 20 most important features
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=20)
selector.fit(df_train_X, df_train_y)
print("Num Features: %d" % selector.get_support().sum())
print("Selected Features: %s" % df_train_X.columns[selector.get_support(indices=True)])

selected_features = df_train_X.columns[selector.get_support(indices=True)]
X_train_selected = df_train_X[selected_features]
X_test_selected = df_test_X[selected_features]

#%% Using SelectFromModel to select the 20 most important features

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

clf = LassoCV()
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(df_train_X, df_train_y)
print("Num Features: %d" % sfm.get_support().sum())
print("Selected Features: %s" % df_train_X.columns[sfm.get_support(indices=True)])

selected_features = df_train_X.columns[sfm.get_support(indices=True)]
X_train_selected = df_train_X[selected_features]
X_test_selected = df_test_X[selected_features]

#%% Using PCA to select the 10 principal components

from sklearn.decomposition import PCA

pca = PCA(n_components=10)
fit = pca.fit(df_train_X)

X_train_pca = pca.transform(df_train_X)
X_test_pca = pca.transform(df_test_X)

#%% Train models

#%% Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train_selected, df_train_y)
y_pred = model.predict(X_test_selected)
y_pred_r = np.rint(y_pred)
y_pred_r = y_pred_r.astype(int)

#%% Save the results 

results = pd.DataFrame({'Id':df_test['Id'],'y': y_pred_r})
results.to_csv("results.csv",index=False)
