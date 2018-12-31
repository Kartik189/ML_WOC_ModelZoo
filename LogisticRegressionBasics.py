import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
def cost(x,theta,y):
	cost=-(np.y.transpose()*np.log(Sigmoid(x*theta))+np.(1-y).transpose()*np.log(1-Sigmoid(x*theta)))/len(y)
	return cost
'''
#Sigmoid Function
def Sigmoid(z):
	y=[]	
	for i in range(1,len(z)):
		y=1/(1+np.exp(-z[i]))
	return y

#Gradient Descent Algorithm
def GradDescent(max_iter,alpha,theta,xtrain,ytrain):
	ctr=0
	while ctr<=max_iter :
		theta=theta-(alpha*np.dot(np.transpose(xtrain),(Sigmoid(np.matmul(theta,np.transpose(xtrain)))-np.transpose(ytrain))))/len(ytrain)
		ctr=ctr+1
	return theta

#Reading csv file
df=pd.read_csv("data.csv")

#Replacing stable as 1 and unstable as 0
df['stabf'].replace('stable',1,inplace=True)
df['stabf'].replace('unstable',0,inplace=True)

#Selecting necessary Features. After testing observation: if we exclude p2 our model's error decreases 
Features=np.asarray(df[['g4','g3','g2','g1','p4','p3','p2','p1','tau4','tau3','tau2','tau1']])
Target=np.asarray(df['stabf'])

#Splitting data into random training and random test sets. 30%test and 70%train
x_train,x_test,y_train,y_test=train_test_split(Features,Target,test_size=0.3,random_state=0)

#Converting Into Arrays For Easier Manipulation
theta1=np.asarray([0.5,0.6,0.7,0.8,0.9,0.2,1,0.5,1,0.2,1,0.4])
xtr=np.asarray(x_train)
ytr=np.asarray(y_train)
xte=np.asarray(x_test)
yte=np.asarray(y_test)
print("dimension of xtr , ytr , xte , yte",np.shape(xtr),np.shape(ytr),np.shape(xte),np.shape(yte))

theta2=GradDescent(200,1e-4,theta1,xtr,ytr)
print("After Applying gradient Descent:: ",theta2)

#Predicting Using Above Trained Algorithm
print(Sigmoid(np.matmul(theta2,np.transpose(xte))),yte)



