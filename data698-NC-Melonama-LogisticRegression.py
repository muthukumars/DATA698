####DATA 698 ----
## Developed by Muthukumar Srinivasan & Rajagopal Srinivasan
## Dated: 20th April 2020 
###########------

### Import All Libraries
import pandas as pd
import statsmodels as statModels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

####Variables definition
#### URL or FileName with Location
url="https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/Melanoma.csv"
split25=0.25
split40=0.40

####Data Loading and Data  Output
### thefollowing section is to print the data with Header
outputForPrintOnly = pd.read_csv(url)
print(outputForPrintOnly);

####Columna Names Definition
names = ["time","sex","age","year","thickness","ulcer","status"]

#### Read Data without first row to avoid column names included in the data set
outputForProcessing=pd.read_csv(url,skiprows=1)
print(outputForProcessing)
outputForProcessingStoredInVar=pd.read_csv(url,names=names,skiprows=1)


#####Define Arrays and store output in Array
array=outputForProcessingStoredInVar.values

X=array[:,0:6]
Y=array[:,6]

###############25% Split
x_train, x_validation, y_train, y_validation =train_test_split(X,Y,test_size=split25,random_state=7)


######Calculate Logistic Regression
#### newton-cg means algorithm to be used for optimization

LR25=LogisticRegression(solver='lbfgs')

LR25.fit(x_train, y_train)
predictions25=LR25.predict(x_validation)
accuracyScore25=accuracy_score(y_validation,predictions25)*100
confusionMatrix25=confusion_matrix(y_validation,predictions25)
classificationReport25=classification_report(y_validation,predictions25)

########PRINT OUTPUT###########

print("------For % of percent testing  set-----")
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

LR40=LogisticRegression(solver='lbfgs')

LR40.fit(x_train, y_train)
predictions=LR40.predict(x_validation)
accuracyScore40=accuracy_score(y_validation,predictions)*100
confusionMatrix40=confusion_matrix(y_validation,predictions)
classificationReport40=classification_report(y_validation,predictions)

########PRINT OUTPUT###########

print("------For % of percent testing  set-----")
print(split40*100)
print("________________________________________")
print("------Accuracy Score-------")
print(accuracyScore40)
print("------Confusion Matrix----")
print(confusionMatrix40)
print("-----ClasificationReport------")
print(classificationReport40)

