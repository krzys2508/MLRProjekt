import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df=pd.read_csv("pimadata.csv")

print(df.head(5))

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

print("{0:0.2f} % in training set".format((len(X_train)/len(df.index))*100))
print("{0:0.2f} % in test set".format((len(X_test)/len(df.index))*100))

fill_0 = SimpleImputer(missing_values=0,strategy="mean")
X_train=fill_0.fit_transform(X_train)
X_test=fill_0.fit_transform(X_test)

nb_model=GaussianNB()
nb_model.fit(X_train,Y_train.ravel())

nb_predict_train = nb_model.predict(X_train)
print("Accuracy against train data: {0:.4f}".format(metrics.accuracy_score(Y_train,nb_predict_train)))

nb_predict_test = nb_model.predict(X_test)
print("Accuracy against test data: {0:.4f}".format(metrics.accuracy_score(Y_test,nb_predict_test)))

print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(Y_test,nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(Y_test,nb_predict_test))
