import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)

data=pd.read_csv('headbrain.csv')
print(data.shape)
print(data.head())

X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values

mean_x=np.mean(X)
mean_y=np.mean(Y)

m=len(X)

num=0
den=0
for i in range(m):
	num+=(X[i]-mean_x)*(Y[i]-mean_y)
	den+=(X[i]-mean_x)**2

b1=num/den
b0=mean_y-(b1*mean_x)
print("\nValue of m and c are ",b1,b0)

max_x=np.max(X)+100
min_x=np.min(X)-100

x=np.linspace(min_x,max_x,1000)
y=b0+b1*x

plt.plot(x,y,color='#58b970',label='Regression Line')
plt.scatter(X,Y,c='#ef5423',label='Scatter Plot')


plt.xlabel('Hwad Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
#plt.show()


ss_t=0
ss_r=0
for i in range(m):
	y_pred=b0+b1*X[i]
	ss_t+=(Y[i]-mean_y)**2
	ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print("R Squared value is ",r2)    	
