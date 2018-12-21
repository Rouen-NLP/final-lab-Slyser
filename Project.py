#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn.metrics as skm
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# %% Data Analysis

dirs = []
[dirs.append(x[0]) for x in os.walk("data/")]
dirs = dirs[1:]


def filecount(folder):
    return len(os.listdir(folder))


n_dirs = len(dirs)
dic_stats = dict()
dic_labels = dict()

i = 0
texts = []
labels = []

for dir_ in dirs:
    # Create dictionnary of labels
    dic_labels[dir_] = i

    # Count number of files in classes
    n_files = filecount(dir_)
    dic_stats[dir_] = n_files

    # import data in lists
    files = os.listdir(dir_)
    for file in files:
        with open(dir_ + '/' + file, encoding="utf-8") as f:
            text = f.read()
            texts.append(text)
            labels.append(i)
    i += 1


# %%Pie Plot of n_docs with respect to class

df = pd.DataFrame(pd.Series(dic_stats))
df = df.reset_index()
df.columns = ["Type of docs", "Numbers of docs"]
pie = df.plot(
    kind='pie',
    y='Numbers of docs',
    autopct='%1.1f%%',
    startangle=90,
    shadow=False,
    labels=df['Type of docs'],
    legend=False,
    fontsize=12,
    figsize=(
        18,
         18))
fig = pie.get_figure()
fig.savefig("RepDocs.png")


# %% Vectorization:

matrix = CountVectorizer()
X = matrix.fit_transform(texts).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)
del X

# %% Classification

def classification(classifier, X_train, y_train, X_test):
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def show_results(classifier_name, y_pred, y_test):
    conf_mat = skm.confusion_matrix(y_test, y_pred)

    plt.figure(2,figsize=(20,20))
    plt.matshow(conf_mat)
    plt.colorbar()
    plt.title("Confusion matrix for " + classifier_name)
    print(
        "Results for " +
        classifier_name +
        " \n \n",
        skm.classification_report(
            y_test,
            y_pred))
    plt.savefig("Confusion_Matrix_of_" +
                classifier_name)
    plt.show()



gnb = GaussianNB()
y_pred_gnb = classification(gnb, X_train, y_train, X_test)
show_results("Naive_Bayes", y_pred_gnb, y_test)

rf = RandomForestClassifier(n_estimators=700)
y_pred_rf = classification(rf, X_train, y_train, X_test)
show_results("Random_Forest", y_pred_rf, y_test)
