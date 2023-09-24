#-*- coding = utf-8 -*-
#@Time : 2023-01-05 12:14
#@File : Adaboost.py
#@Software: PyCharm
#@Author:HanYixuan

from DataPreprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import time
from Logger import Logger
import warnings
warnings.filterwarnings("ignore")

# console log
log_path = 'log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)

# START: OWN CODE
# names
dataname="Algerian_Forest_Fire_withoutFWI"
pklname='result/ADB_{dataname}.pkl'.format(dataname=dataname)
figname='img/ADB_{dataname}.jpg'.format(dataname=dataname)

# get dataset from original format
x,y,feature_name,class_labels=Preprocessing(dataname)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3,random_state=22)

# rectify each dataset's index
for i in [Xtrain,Xtest,Ytrain,Ytest]:
    i.index=range(i.shape[0])

print(" using Adaboost to train************************************************")
clf = AdaBoostClassifier(random_state=1)
clf = clf.fit(Xtrain, Ytrain.values.ravel())
print("confusion matrix: ",metrics.confusion_matrix(y_true=Ytest,
    y_pred=clf.predict(Xtest)))
score = cross_val_score(clf, x, y, cv=10)
print("cross_val_score vector: ",score)
score1=score.mean()
print("cross_val_score mean value: ",score1)
print("feature_importances_:",clf.feature_importances_)

print("observe number of estimators' influence***************************************")
score1s = []
for i  in range(0,200,10):
    clf = AdaBoostClassifier(n_estimators=i+1, n_jobs=-1,random_state=1)
    clf.fit(Xtrain, Ytrain.values.ravel())
    score1 = cross_val_score(clf, x, y, cv=10).mean()
    score1s.append(score1)
print("score1s",score1s)
plt.plot(range(1,201,10),score1s)
plt.xlabel('n_estimators')
plt.ylabel('acc')
plt.savefig(figname)
plt.show()
b_n_emr=(score1s.index(max(score1s))*10)+1
print("the best number of estimator and score is {n_emr} and {scr}".format(n_emr=b_n_emr,scr=max(score1s)))

print("using grid-search method*********************************************")
DTrees=[]
for max_d in range(1,10):
    for min_s_l in range(1,20,5):
        for min_s_s in range(1,20,5):
            DTrees.append(DecisionTreeClassifier(max_depth=max_d,min_samples_leaf=min_s_l,min_samples_split=min_s_s))

parameters = {"base_estimator": DTrees,
              "learning_rate":[0.1,0.5,0.7,1]}
clf = AdaBoostClassifier(n_estimators=b_n_emr,random_state=1)
grid = GridSearchCV(clf, parameters, cv=10,verbose=1)
grid.fit(Xtrain, Ytrain.values.ravel())
print("the best parameter combination is",grid.best_params_)
print("the best score is ",grid.best_score_)
print("the cv_results is ",grid.cv_results_)
w = open(pklname, 'wb')
pickle.dump(grid.cv_results_, w)

print(" using ADB with best parameters to train****************")
clf = AdaBoostClassifier(base_estimator=grid.best_params_['base_estimator'],
                         n_estimators=b_n_emr,random_state=1,
                         learning_rate=grid.best_params_['learning_rate'])
clf = clf.fit(Xtrain, Ytrain.values.ravel())
print("confusion matrix: ",metrics.confusion_matrix(y_true=Ytest,
    y_pred=clf.predict(Xtest)))
score = cross_val_score(clf, x, y, cv=10)
print("cross_val_score vector: ",score)
score1=score.mean()
print("cross_val_score mean value: ",score1)
print("feature_importances_:",clf.feature_importances_)
# END: OWN CODE
