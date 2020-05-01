####Program Written by Muthukumar Srinivasan &Rajagopal Srinivasan
####Date:25th Aprtil 2020

###IMPORT Packages Section

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula as smf
from sklearn.impute import SimpleImputer

###Variables definition
URL="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/diabetes.csv"
split20=0.20

##Read CSV File
dataset=pd.read_csv(URL)
datasetCount=dataset.count

## Read and store only needed values
datasetNew=dataset.iloc[:,[0,1,2,3,4,5,7,9,10,12,13,14,15,16,17,18]]
datasetNewCountr=datasetNew.count

### Split X and Y values
x=datasetNew.iloc[:,:-1].values
y=datasetNew.iloc[:,15].values

####Data Preprocessing
imp=SimpleImputer(missing_values=np.nan,strategy="mean")
x=imp.fit_transform(x)
y=y.reshape(-1,1)
y=imp.fit_transform(y)
y=y.reshape(-1)

#### Build Regression
regressor=LinearRegression()
regressor.fit(x,y)

#####Prediction 
y_pred=regressor.predict(x)


#### Add one extra value to x
x=np.append(arr=np.ones((403,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()

#########Print Outputs
print(dataset)
print(datasetNew)
print(x)
print(y)
print(regressor_OLS.summary())
