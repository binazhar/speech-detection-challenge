# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:56:17 2019

@author: Aziz-Dqube
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import missingno as msno

from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier





#load voice data

path = 'C:\\Users\\Aziz-Dqube\\Desktop\\TMW\\Task2\\'
voices_dataSet = pd.read_csv(path+"voices.csv")

plt.rc("font", size=14)
sns.set(style='white')
sns.set(style="whitegrid", color_codes = True)

################################# First Iteration - Build Logistic Regression Model #############################################
print(voices_dataSet.head(10))

#Test for numeric values
print(voices_dataSet[~voices_dataSet.applymap(np.isreal).all(1)])

#Test for null values
null_columns=voices_dataSet.columns[voices_dataSet.isnull().any()]
print(null_columns)


print("Male Vs Female")
print(voices_dataSet['label'].value_counts())
#male      1584
#female    1584
#Name: label, dtype: int64

print(voices_dataSet.info()) 
#meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',  'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label'],
#All attributes are float64 except label which is object

# Individual Plotting
#ax=sns.distplot(voices_dataSet['meanfreq'])
#ax=sns.distplot(voices_dataSet['sd'])
#ax=sns.distplot(voices_dataSet['median'])
#ax=sns.distplot(voices_dataSet['Q25'])
#ax=sns.distplot(voices_dataSet['Q75'])
#ax=sns.distplot(voices_dataSet['IQR'])
#ax=sns.distplot(voices_dataSet['skew'])
#ax=sns.distplot(voices_dataSet['kurt'])

#Pairplot Analysis
#sns.pairplot(voices_dataSet,diag_kind='kde')


#create a copy of orginal data as we are going to modify labels into numeric value
voices_dataSet_copy = voices_dataSet.copy() 

voices_dataSet_copy.label=[1 if each =="female" else 0 for each in voices_dataSet_copy.label]

#verification step
#print(voices_dataSet_copy.head(10))
#print(voices_dataSet_copy.tail(10))
voices_dataSet_copy.info()


#Setting outcome on y-axis and all other attributes on x-axis
y=voices_dataSet_copy.label.values
x_data=voices_dataSet_copy.drop(['label'],axis=1)

#Detremining boundaries of data

#kurt, skew, maxdom & dfrange have higher values than others
print(np.max(x_data))

#kurt have higher values than others
print(np.min(x_data))

#normalization step
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#Split the data into train and test data sets

test_size = 0.30 # creating a 70-30 split of data between train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

#x_train.info()
#x_test.info()
#y_train.info()
#y_test.info()


#Using Logistic Regression model from sklearn
Logistic_model = LogisticRegression()

#Fitting to create sigmoid function
Logistic_model.fit(x_train, y_train)

print("Logistic Regression Classification Test Accuracy {}".format(Logistic_model.score(x_test,y_test)))
print("Logistic Regression Classification Train Accuracy {}".format(Logistic_model.score(x_train,y_train)))

y_predict = Logistic_model.predict(x_test)


######################################### Second Iteration - Finding Correlation between Attributes and Classification ##############################


#correlation matrix.
cor_mat= voices_dataSet_copy.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

#######################################################################
# 1. Centroid and meanfreq are exactly correlated (1). A close inspection of columns 

df = pd.DataFrame()
#df['outcome'] = (voices_dataSet_copy.iloc[:,:-1] == 1).all(1).astype(int) 
df['outcome'] = voices_dataSet_copy.apply(lambda x: 1 if [voices_dataSet_copy['meanfreq']==voices_dataSet_copy['centroid']] else '', axis=1)
#print (df)

# Indicates that both all equal, all 1's. So we can omit centroid column
# Rest of correlation w.r.t label
# Highly Correlated - Positive
# - IQR 
# - sp.ent (Spectral Entropy)
# - sd

# Highly Correlated - Negative
# - Q25
# - meanfun

# Moderately Correlated - Positive 
# - sfm

# Moderately Correlated - Negative 
# - meanfreq
# - median
# - centroid (same as meanfreq)

# Very weak / non extent corelation
# - Q75
# - skew
# - kurt
# - modindx


# Drop columns for feature engineering
voices_dataSet_copy.drop(['centroid','Q75','skew', 'kurt','modindx' ],axis=1,inplace=True)

######################################### Second Iteration - Results with other models ##############################

# K- Nearest Neighbor
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
print("KNN Classification Test Accuracy {}".format(knn_model.score(x_test,y_test)))
print("KNN Classification Train Accuracy {}".format(knn_model.score(x_train,y_train)))

#Support Vector Machine
svc_model = SVC()
svc_model.fit(x_train, y_train)
print("SVC Classification Test Accuracy {}".format(svc_model.score(x_test,y_test)))
print("SVC Classification Train Accuracy {}".format(svc_model.score(x_train,y_train)))


#Decision Tree
dtc_model = DecisionTreeClassifier()
dtc_model.fit(x_train, y_train)
print("Decision Tree Classification Test Accuracy {}".format(dtc_model.score(x_test,y_test)))
print("Decision Tree Classification Train Accuracy {}".format(dtc_model.score(x_train,y_train)))

#Random Forest
rtc_model = RandomForestClassifier()
rtc_model.fit(x_train, y_train)
print("Random Forest Classification Test Accuracy {}".format(rtc_model.score(x_test,y_test)))
print("Random Forest Classification Train Accuracy {}".format(rtc_model.score(x_train,y_train)))


#ax=sns.distplot(voices_dataSet['meanfreq'])
#ax=sns.distplot(voices_dataSet['sd'])
#ax=sns.distplot(voices_dataSet['median'])
#ax=sns.distplot(voices_dataSet['Q25'])
#ax=sns.distplot(voices_dataSet['Q75'])
#ax=sns.distplot(voices_dataSet['IQR'])
#ax=sns.distplot(voices_dataSet['skew'])
#ax=sns.distplot(voices_dataSet['kurt'])

#Pairplot Analysis
#sns.pairplot(voices_dataSet,diag_kind='kde')
