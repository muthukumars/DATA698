####DATA 698 ----
## Developed by Muthukumar Srinivasan & Rajagopal Srinivasan
## Dated: 20th April 2020 
###########------

### Import All Libraries
import pandas as pd
import pandas
import statsmodels as statModels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

####Variables definition
#### URL or FileName with Location
url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/abm.xls"
split25=0.25
split40=0.40

####Data Loa=ding and Data  Output
### thefollowing section is to print the data with Header
dataset = pd.read_excel(url)
print(dataset);

####Columna Names Definition
#names = ["time","sex","age","year","thickness","ulcer","status"]

#### Read Data without first row to avoid column names included in the data set
datasetForProcessing=pd.read_excel(url,skiprows=1)
datasetFinal=datasetForProcessing.iloc[:,[0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42]]

NoColumnsFinal=datasetFinal.columns

for i in NoColumnsFinal:
	datasetFinal[i]=datasetFinal[i].astype(float).fillna(0.0)

print(datasetFinal)

#####Define Arrays and store output in Array
X=datasetFinal.iloc[:,0:38]
Y=datasetFinal.iloc[:,39]

###############25% Split
x_train, x_validation, y_train, y_validation =train_test_split(X,Y,test_size=split25,random_state=7)

######Calculate Logistic Regression
#### newton-cg means algorithm to be used for optimization

GNB25=GaussianNB()

GNB25.fit(x_train, y_train)
predictions25=GNB25.predict(x_validation)
accuracyScore25=accuracy_score(y_validation,predictions25)*100
confusionMatrix25=confusion_matrix(y_validation,predictions25)
classificationReport25=classification_report(y_validation,predictions25)

########PRINT OUTPUT###########

print("------For % of percent testing  set---ABM----Gaussian")
print(split25*100)
print("________________________________________")
print("------Accuracy Score-------")
print(accuracyScore25)
print("------Confusion Matrix----")
print(confusionMatrix25)
print("-----ClasificationReport------")
print(classificationReport25)

###############40% Split
x_train, x_validation, y_train, y_validation =train_test_split(X,Y,test_size=split40,random_state=7)


######Calculate Logistic Regression
#### newton-cg means algorithm to be used for optimization

GNB40=GaussianNB()

GNB40.fit(x_train, y_train)
predictions=GNB40.predict(x_validation)
accuracyScore40=accuracy_score(y_validation,predictions)*100
confusionMatrix40=confusion_matrix(y_validation,predictions)
classificationReport40=classification_report(y_validation,predictions)

########PRINT OUTPUT###########

print("------For % of percent testing  set-ABM----Gaussian-")
print(split40*100)
print("________________________________________")
print("------Accuracy Score-------")
print(accuracyScore40)
print("------Confusion Matrix----")
print(confusionMatrix40)
print("-----ClasificationReport------")
print(classificationReport40)

