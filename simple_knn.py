from sklearn import datasets 
iris=datasets.load_iris()

from sklearn.model_selection import train_test_split
x=iris.data
y=iris.target

x_train, x_test ,y_train ,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


import random
from scipy.spatial import distance
def euc(a,b):
  return distance.euclidean(a,b)

class simpleKnn():
  def fit(self, x , y):
    self.x=x
    self.y=y

  def predict(self,x_test):
    predictions=[]
    for row in x_test:
      label=self.closest(row)
      predictions.append(label)
    return predictions

  def closest(self,row):
    best_dist=euc(row,self.x[0])
    best_index=0
    for i in range(1, len(x)):
      dista=euc(row,self.x[i])
      if float(dista)<best_dist:
        best_dist=distance
        best_index=i
    return self.y[best_index]

clf2=simpleKnn()
clf2.fit(x_train,y_train)
predict = clf2.predict(x_test)

print(accuracy_score(predict,y_test))

 
