# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 04:54:01 2020

@author: HSH
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('car data.csv')
dataset.isnull().sum()

dataset['Owner'].unique()

from datetime import date
today = date.today()
today = today.strftime("%Y")
print("Today's date:", today)
dataset["current_year"]= dataset["Year"]-2020


dataset=dataset[['current_year','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner','Selling_Price']]
dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)
X.view()
np.set_printoptions(edgeitems=20)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

np.set_printoptions(edgeitems=20)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

np.set_printoptions(edgeitems=20)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
X = np.array(ct.fit_transform(X))




y = np.array(y).reshape((len(y), 1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)






from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)



y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


import seaborn as sns
sns.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred)
