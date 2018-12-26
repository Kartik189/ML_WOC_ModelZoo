#Importing Necessary Libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy
import seaborn as sns


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

#Training Our Model
lm=LinearRegression()
lm.fit(x_train,y_train)
print("INTERCEPT: ",lm.intercept_)
print("COEFFICIENTS: ",lm.coef_)

#Finding Efficiency of our model using R^2
print(lm.score(x_test,y_test))
#Since we get R^2 score of 0.8438 it is a good fit to our data

#Prediction/Testing of out model
Yhat=lm.predict(x_test)
print(Yhat)




