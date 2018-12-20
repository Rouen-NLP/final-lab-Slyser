#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:44:38 2018

@author: leroghug
"""

import os
import pandas as pd

#%% Data Analysis
dirs=[]
[dirs.append(x[0]) for x in os.walk("data/")]
dirs=dirs[1:]

def filecount(folder):
    return len(os.listdir(folder))

n_dirs=len(dirs)
dic_stats=dict()
dic_labels=dict()

i=0
texts=[]
labels=[]

for dir_ in dirs:
    #Create dictionnary of labels
    dic_labels[dir_]=i
    
    #Count number of files in classes
    n_files=filecount(dir_)
    dic_stats[dir_]=n_files
    
    #import data in lists
    files=os.listdir(dir_)
    for file in files:
        with open(dir_ +'/'+file) as f:
            text=f.read()
            texts.append(text)
            labels.append(i)
    i+=1
        

#Pie Plot of n_docs with respect to class
import matplotlib.pyplot as plt

df=pd.DataFrame(pd.Series(dic_stats))
df=df.reset_index()
df.columns=["Type of docs","Numbers of docs"]
df.plot(kind='pie', y = 'Numbers of docs', autopct='%1.1f%%', startangle=90, shadow=False, labels=df['Type of docs'], legend = False, fontsize=9)
plt.show()
plt.savefig('Repartition of documents')



#%% Vectorization:
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

matrix = CountVectorizer()
X = matrix.fit_transform(texts).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)

#%% Classification  
import sklearn.metrics as skm

def classification(classifier,X_train,y_train,X_test):
    clf=classifier
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    return y_pred

def show_results(classifier_name,y_pred,y_test):
    conf_mat=skm.confusion_matrix(y_test,y_pred)
    
    #plt.figure(2,figsize=(10,10))
    plt.matshow(conf_mat)
    plt.colorbar()
    plt.title("Confusion matrix for " + classifier_name)
    plt.show()
    print("Results for " + classifier_name+" \n \n", skm.classification_report(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

gnb=GaussianNB()
y_pred_gnb=classification(gnb,X_train,y_train,X_test)
show_results("Naive Bayes", y_pred_gnb, y_test)

rf = RandomForestClassifier(n_estimators=500)
y_pred_rf=classification(rf,X_train,y_train,X_test)
show_results("Random Forest", y_pred_rf, y_test)
