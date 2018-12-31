#Importing Necessary Libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

#Reading csv File On Fuel Consumptions
df=pd.read_csv("FuelConsumptionCo2.csv")

#Checking Data Types
#print(df.dtypes)

#Sorting Out Required Features
#print(df.corr())

#We observe that Enginesize,Cylinders,FuelConsumptions are the necessary features
Features=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
Target=df['CO2EMISSIONS']

#Splitting Our Data Into Train and Test Subset
x_train,x_test,y_train,y_test=train_test_split(Features,Target,test_size=0.3,random_state=0)

#using pipelineto train as well as normalize out polynomial regression model
Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=4)),('mode',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(x_train,y_train)
yhat=pipe.predict(x_test)
print(yhat)

print("R2-score: %.2f" % r2_score(yhat , y_test) )
#Since we get a score of 0.91 it is much better then our linear regression model

