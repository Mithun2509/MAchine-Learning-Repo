from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


data=pd.read_csv('headbrain.csv')
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values

m=len(X)
#can't use rank 1 matrix in scikit learn
X=X.reshape((m,1))
#create the model
reg=LinearRegression()
#fit the training data
reg=reg.fit(X,Y)
#y prediction
Y_pred=reg.predict(X)

#calculating r2 score
r2_score=reg.score(X,Y)

print("R Squared value is ",r2_score)