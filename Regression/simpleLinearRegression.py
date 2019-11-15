
"""
Created on November 14 19:37:49 2019

@author: pratik
"""
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("linear_dataset.csv")

X=data.iloc[: , :-1]
y=data.iloc[: , 1]

# Splitting Training and test Data set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X, y , test_size=1/3 , random_state=0)

# Implementation of our classifier Linear model

from sklearn.linear_model import LinearRegression 
simpleLinearRegression= LinearRegression()

simpleLinearRegression.fit(X_train , y_train)

y_predict= simpleLinearRegression.predict(X_test)
print(y_predict)



plt.scatter(X_train ,y_train , c="red")
plt.plot( X_train , simpleLinearRegression.predict(X_train))
plt.show()

from sklearn.metrics import r2_score
score=r2_score(y_test , y_predict)
print(score)

