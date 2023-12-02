# Importing the libraries
from sklearn.feature_selection import SelectKBest, f_regression
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
#import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import os

# Parsing the arguments
str2bool = lambda x: (str(x).lower() == 'true')
parser = argparse.ArgumentParser(description='Regression model')
parser.add_argument('--prueba', type=str2bool, default=True, help='Use the test data')
parser.add_argument('--FS_method', type=str, default='SelectFromModel', help='Feature selection method')
parser.add_argument('--PCA', type=str, default=False, help='Use PCA')
parser.add_argument('--model', type=str, default='XgBoost', help='Model to use')
args = parser.parse_args()

if args.prueba:
    data = pd.read_csv(os.path.join('data','trainReg.txt'))
    # Splitting the data into train and test
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    # Separate the predictors from the target
    df_test_X = df_test.drop(['V1'], axis = 1)
    df_test_y = df_test['V1']

else:
    # Uploading the data
    df_train = pd.read_csv(os.path.join('data','trainReg.txt'))
    df_test = pd.read_csv(os.path.join('data','testReg.txt'))
    # Do not use the Id column
    df_test_X = df_test.drop(['Id'], axis = 1)

# Separate the predictors from the target
df_train_X = df_train.drop(['V1'], axis = 1)
df_train_y = df_train['V1']

#Normalizing the data
scaler = MinMaxScaler()
df_train_X[df_train_X.columns] = scaler.fit_transform(df_train_X)
df_test_X[df_test_X.columns] = scaler.fit_transform(df_test_X)

# Feature selection
print("Method: %s" % args.FS_method)

if args.FS_method == 'RFE':
    # Using Recursive Feature Elimination to select the most important features
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=60)
    fit = rfe.fit(df_train_X, df_train_y)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % df_train_X.columns[fit.support_])
    
    selected_features = df_train_X.columns[fit.support_]
    X_train_selected = df_train_X[selected_features]
    X_test_selected = df_test_X[selected_features]

elif args.FS_method == 'SelectKBest':
    # Using SelectKBest to select the most important features
    selector = SelectKBest(f_regression, k=60)
    selector.fit(df_train_X, df_train_y)
    print("Num Features: %d" % selector.get_support().sum())
    print("Selected Features: %s" % df_train_X.columns[selector.get_support(indices=True)])

    selected_features = df_train_X.columns[selector.get_support(indices=True)]
    X_train_selected = df_train_X[selected_features]
    X_test_selected = df_test_X[selected_features]

elif args.FS_method == 'SelectFromModel':
    # Using SelectFromModel to select the most important features
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(df_train_X, df_train_y)
    print("Num Features: %d" % sfm.get_support().sum())
    print("Selected Features: %s" % df_train_X.columns[sfm.get_support(indices=True)])

    selected_features = df_train_X.columns[sfm.get_support(indices=True)]
    X_train_selected = df_train_X[selected_features]
    X_test_selected = df_test_X[selected_features]

else:
    X_train_selected = df_train_X
    X_test_selected = df_test_X

if args.PCA:
    # Using PCA to select the 10 principal components
    pca = PCA(n_components=10)
    fit = pca.fit(df_train_X)
    X_train_pca = pca.transform(df_train_X)
    X_test_pca = pca.transform(df_test_X)

# Training the model

if args.model == 'XgBoost':
    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train_selected, df_train_y)
    params = {'max_depth': 3,'eta': 0.01,"objective": "reg:squarederror", "tree_method": "hist", "device": "cuda"}
    n = 100

    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
    )

    dtest_reg = xgb.DMatrix(X_test_selected)
    y_pred = model.predict(dtest_reg)
    y_pred_r = np.rint(y_pred)
    y_pred_r = y_pred_r.astype(int)

elif args.model == 'GradientBoostingRegressor':
    model = GradientBoostingRegressor()
    model.fit(X_train_selected, df_train_y)
    y_pred = model.predict(X_test_selected)
    y_pred_r = np.rint(y_pred)
    y_pred_r = y_pred_r.astype(int)

elif args.model == 'NN':

    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(60, input_dim=X_train_selected.shape[1], kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    # evaluate model
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    estimator.fit(X_train_selected, df_train_y)
    y_pred = estimator.predict(X_test_selected)
    y_pred_r = np.rint(y_pred)
    y_pred_r = y_pred_r.astype(int)

if args.prueba:
    print(y_pred_r)
    print('Mean Absolute Error:', mean_absolute_error(df_test_y, y_pred_r))

else:
    # Save the results 
    results = pd.DataFrame({'Id':df_test['Id'],'y': y_pred_r})
    results.to_csv(f"results_{args.FS_method}.csv",index=False)





