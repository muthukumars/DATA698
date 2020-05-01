####Written by Muthukumar Srinivasan and Rajagopal Srinivasan
######25th April 2020

##Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels

from sklearn.model_selection import train_test_split

# explore random forest number of features effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold

###Variables
url="https://raw.githubusercontent.com/dataminingIrbed/data/master/HCV-Egy-Data.csv"

###Read data set
dataset=pd.read_csv(url)

print(dataset)

### identify x and y values
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,28].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# define dataset
x_test, y_test = make_regression(n_samples=1385, n_features=29, n_informative=15,noise=0.1,random_state=2)

# define the model

model = RandomForestRegressor(n_estimators=100)

# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,error_score='raise')

print('-----Significany-Mean -----')
print(mean(n_scores))
print('---------------------------')
print('-----Standard Deviation----')
print(std(n_scores))
print('---------------------------')
