import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
#Loading data
df=pd.read_csv("pimadata.csv")
df.shape
print(df.head(5))
#Oczyszcanie danych sprawdzamy czy sa puste
print(df.isnull().values.any())

num_obs = len(df)
num_true = len(df.loc[df['diabetes']==1])
num_false = len(df.loc[df['diabetes']==0])
print("Number of True cases: {0} ({1:2.2f})%".format(num_true,(num_true/num_obs)*100))
print("Number of False cases: {0} ({1:2.2f})%".format(num_false,(num_false/num_obs)*100))

feature_column_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age','skin']
predicted_column_name = ['diabetes']
x = df[feature_column_names].values
y =df[predicted_column_name].values
split_test_size = 0.30
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=split_test_size,random_state=42)
#Sprawdzamy czy sie podzielilo rowno
print("{0:0.2f} % in training set".format((len(X_train)/len(df.index))*100))
print("{0:0.2f} % in test set".format((len(X_test)/len(df.index))*100))
#Post split data preperation
#replacing zeros with mean
fill_0 = SimpleImputer(missing_values=0,strategy="mean")
X_train=fill_0.fit_transform(X_train)
X_test=fill_0.fit_transform(X_test)
#Training our model with NAIVE BAYES using GuassianNB
nb_model=GaussianNB()
nb_model.fit(X_train,Y_train.ravel())
#Checking how well trained our train prediciton
nb_predict_train = nb_model.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_train,nb_predict_train)))
#Checking how well trained our test prediciton
nb_predict_test = nb_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test,nb_predict_test)))
#MORE DETAIL
#Confusion matrix - TopLeft - Actual not diabetes and predicted not diabetes; TopRight actual not diabetes but predicted to be diabetes; BottomRight - actual diabetes predicted to be diabetes; BottomLeft - actual diabetes but not diabetes
print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(Y_test,nb_predict_test)))
print("")
#Clasification report - generates stats based on confusion matrix
print("Classification Report")
print(metrics.classification_report(Y_test,nb_predict_test))