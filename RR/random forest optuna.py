
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import ADASYN
import optuna
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import ADASYN
import optuna
import joblib

# Read data
data = pd.read_excel('TrainClass.xlsx')
real_data = pd.read_excel('TestClass.xlsx')

# Split data
Y, X = data['FRACASO'], data.drop(['FRACASO', 'CODIGO_EMPRESA'], axis=1)

# Apply ADASYN
adasyn = ADASYN(random_state=0)
X_resampled, Y_resampled = adasyn.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=0)
X = real_data.drop(['CODIGO_EMPRESA'], axis=1)

# Define the objective function for cross-validated AUC
def objective(trial):
    # Define hyperparameters to be optimized
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 1, 10)

    # Create the classifier with suggested hyperparameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

    # Feature selection based on importance
    sfm = SelectFromModel(clf)
    X_train_selected = sfm.fit_transform(X_train, Y_train)

    # Define cross-validation strategy (StratifiedKFold for classification)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Perform cross-validation and get mean AUC
    auc_scores = cross_val_score(clf, X_train_selected, Y_train, cv=cv, scoring='roc_auc')
    mean_auc = auc_scores.mean()

    return mean_auc

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print(f'Best Hyperparameters: {best_params}')

# Train the final model with the best hyperparameters
best_clf = RandomForestClassifier(**best_params, random_state=0)

# Feature selection on the entire dataset
sfm = SelectFromModel(best_clf)
X_train_selected = sfm.fit_transform(X_train, Y_train)
X_test_selected = sfm.transform(X_test)
X_selected = sfm.transform(X)

best_clf.fit(X_train_selected, Y_train)

# Predictions on training set
y_pred_train = best_clf.predict(X_train_selected)

# Calculate AUC for training set
auc_train = roc_auc_score(Y_train, y_pred_train)
print(f'AUC for Training Set: {auc_train:.4f}')

# Predictions on testing set
y_pred_test = best_clf.predict(X_test_selected)

# Calculate AUC for testing set
auc_test = roc_auc_score(Y_test, y_pred_test)
print(f'AUC for Testing Set: {auc_test:.4f}')

# Save the trained model to a file
joblib.dump(best_clf, 'random_forest_model.joblib')

# Load model and generate probability CSV
loaded_model = joblib.load('random_forest_model.joblib')
prob = loaded_model.predict_proba(X_selected)[:, 1]
prob_df = pd.DataFrame(prob, columns=['Probability'])
prob_df.index = prob_df.index + 1
prob_df.index.name = 'Id'
prob_df.to_csv('intento.csv', index=True)