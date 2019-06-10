# Linear Regression 
# This code performs linear regression and saves the model

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

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

#%% Perform Linear Regression 
sc = StandardScaler()
train_features_Z = sc.fit_transform(train_features)
test_features_Z = sc.transform (test_features)
model = LinearRegression(fit_intercept=True)
model.fit(train_features_Z, train_labels)

#%% Save Linear Model
filename = '/Users/tung-linwu/Desktop/Insight/models/'+ route_segment +'_linear.sav'
pickle.dump(model, open(filename, 'wb'))

