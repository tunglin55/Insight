# Hyperparameter Tuning for Random Forest Regressor 
# This code performs a random search and subsequently a grid search for optmizing the model 

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

#%% Read in design matrix 
route_segment='LY_D'
df = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/design_matrix/D_matrix_' +route_segment+'_bt.csv')
df = df.set_index(pd.DatetimeIndex(df['Datetime']))
df.drop(df.columns[[0]], axis=1, inplace=True)

#%% Setting Up for Regression
# X is the design matrix, Y is the labels 
Y = np.array(df['timeInSeconds'])
X = df.drop('timeInSeconds', axis = 1)
X = np.array(X)

#%% Split into train and test datasets
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#%% Perform random search 
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 50)] # Number of trees in random forest
max_features = ['auto'] # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]# Minimum number of samples required at each leaf node
bootstrap = [True] # Method of selecting samples for training each tree
# Create random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
# Random search of parameters with 3-fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search to the model
rf_random.fit(train_features, train_labels)
rf_random.best_params_

#%% Perform Grid Search  
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [15,20,30,100,None],
    'max_features': ['auto'],
    'min_samples_leaf': [2,3,4,5,6],
    'min_samples_split': [3,4,5,6,8,10],
    'n_estimators': [2000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(train_features, train_labels)
best_grid = grid_search.best_estimator_
grid_search.best_params_

#%% Save Hypertuned Model
filename = '/Users/tung-linwu/Desktop/Insight/models/'+ route_segment +'.sav'
pickle.dump(best_grid, open(filename, 'wb'))

