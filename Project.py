#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:44:38 2018

@author: leroghug
"""

import os
import numpy as np
import pandas as pd

#%% Analyse des données
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
        

#PlotBar of n_docs with respect to class
df=pd.DataFrame(pd.Series(dic_stats))
df=df.reset_index()
df.columns=["Type of docs","Numbers of docs"]
df.plot.bar(x="Type of docs")

#%% Vectorization:
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer()
X = matrix.fit_transform(texts).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)

#%% Classification Gaussian
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

# Accuracy 
from sklearn.metrics import f1_score
f1_score_gnb = f1_score(y_test, y_pred,average="macro")


#%% Classification RF
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
f1_score_rf=f1_score(y_test,y_pred,average="micro")