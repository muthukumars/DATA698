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


###Variables
url="https://raw.githubusercontent.com/dataminingIrbed/data/master/HCV-Egy-Data.csv"

###Read data set
dataset=pd.read_csv(url)

print(dataset)

### identify x and y values
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,28].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#############ALL NECESSARY FUNCTIONS ARE BELOW

# get the dataset
def get_dataset():
	x_test, y_test = make_classification(n_samples=1385, n_features=29,
	n_informative=15, n_redundant=5, random_state=3)
	return x_test, y_test


# get a list of models to evaluate
def get_models():
	models = dict()
	models['10'] = RandomForestClassifier(n_estimators=10)
	models['50'] = RandomForestClassifier(n_estimators=50)
	models['100'] = RandomForestClassifier(n_estimators=100)
	models['500'] = RandomForestClassifier(n_estimators=500)
	models['1000'] = RandomForestClassifier(n_estimators=1000)
	return models

# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, x_test, y_test, scoring='accuracy',cv=cv, n_jobs=-1,error_score='raise')
	return scores



# define dataset
x_test, y_test = get_dataset()
print(x_test,y_test)
print(len(x_test))
print(len(y_test))

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()

for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print(name)
	print(mean(scores))
	print(std(scores))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
