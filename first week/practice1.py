# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:44:48 2020

@author: xiaon
"""

import pandas as pd
auto_data = pd.read_csv('C:\\Users\\xiaon\\auto-mpg.csv')
#show horsepower numeric data
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce')
#drop unvalued data 
auto_data = auto_data.drop(['car name'], axis=1)
#drop horsepower missing data row
auto_data_final = auto_data.dropna(axis=0)

#evaluate model performance metrics.
auto_data_final[auto_data_final.isnull().any(axis=1)]
#auto_data_final.info() #392rows

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
x = auto_data_final.drop('mpg',axis = 1)
y = auto_data_final['mpg']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,test_size=0.2,random_state = 0)
model = SVR(kernel = 'linear',C = 1.0)
model.fit(x_train, y_train)
#print(model.coef_)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
model_mse = mean_squared_error(y_predict, y_test)
print(model_mse)

# Check the correlation matrix to derive horsepower feature by help of other feature
corr = auto_data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(4)
print(corr)












