#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 14 20:23:48 2019

@author: pratik
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset= pd.read_csv("50_Startups.csv")

X= dataset.iloc[: , : -1]
y= dataset.iloc[ : , 4]

# Reducing one state using dummy values 

states= pd.get_dummies( X['State'] , drop_first=True)


# drop the state column 
X=X.drop('State' , axis=1)


# append the states in optimized format
X=pd.concat([X, states] , axis=1)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y , test_size=0.2 , random_state = 0) 



from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train ,y_train)

y_predict = regressor.predict(X_test)



from sklearn.metrics import r2_score
score= r2_score(y_test , y_predict)

print(score)

