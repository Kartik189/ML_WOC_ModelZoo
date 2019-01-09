#importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

#Importing DataFrame
df=pd.read_csv("data.csv")
print(df.head(9))

#Function To Compute Misclassification Error
def mul_error(yhat,ytest):
	error=0.0
	for i in range(0,len(yhat)):
		if (yhat[i]==1 and ytest[i]==0) or (yhat[i]==0 and ytest[i]==1) :
			error+=1.0
	print("Misclassification error=",error/(len(yhat)+1))

#Replacing stable as 1 and unstable as 0
df['stabf'].replace('stable',1,inplace=True)
df['stabf'].replace('unstable',0,inplace=True)
print(df.head(9))

#Selecting necessary Features. After testing observation: if we exclude p2 our model's error decreases 
Features=np.asarray(df[['g4','g3','g2','g1','p4','p3','p1','tau4','tau3','tau2','tau1']])
Target=np.asarray(df['stabf'])

#Splitting data into random training and random test sets. 30%test and 70%train
x_train,x_test,y_train,y_test=train_test_split(Features,Target,test_size=0.3,random_state=0)

#Training Our Model
clf=svm.SVC(C=20.0,kernel='rbf',max_iter=2500)
clf.fit(x_train,y_train)

Yhat=clf.predict(x_test)
print('Actual Values : ',y_test)
print('Predicted Values : ',Yhat)

#Checking Accuracy using misclassification error
mul_error(Yhat,y_test)
