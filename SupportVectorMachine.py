#-*- coding = utf-8 -*-
#@Time : 2023-01-04 14:34
#@File : SupportVectorMachine.py
#@Software: PyCharm
#@Author:HanYixuan

from DataPreprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
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
dataname="Algerian_Forest_Fire"
pklname='result/SVM_{dataname}.pkl'.format(dataname=dataname)

# get dataset from original format
x,y,feature_name,class_labels=Preprocessing(dataname,method="standard_scale")
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3,random_state=22)

# rectify each dataset's index
for i in [Xtrain,Xtest,Ytrain,Ytest]:
    i.index=range(i.shape[0])

print(" using SVM to train************************************************")
clf = SVC(random_state=1)
clf = clf.fit(Xtrain, Ytrain.values.ravel())
print("confusion matrix: ",metrics.confusion_matrix(y_true=Ytest,
    y_pred=clf.predict(Xtest)))
score = cross_val_score(clf, x, y, cv=10)
print("cross_val_score vector: ",score)
score1=score.mean()
print("cross_val_score mean value: ",score1)

print(" using grid-search method********************************************")
decision_function_shape=[]
if len(class_labels)>2:
    decision_function_shape=['ovo','ovr']
else:
    decision_function_shape = ['ovo']
print(decision_function_shape)
parameters = [{'C':[0.001,0.003,0.006,0.009,0.01,0.04,0.08,0.1,1, 10, 100],
              'kernel':['linear'],
              'decision_function_shape':decision_function_shape
             },
              {'C': [0.001, 0.003, 0.006, 0.009, 0.01, 0.04, 0.08, 0.1, 1, 10, 100],
              'kernel':['rbf'],
              'gamma':[0.001,0.005,0.1,0.15,0.20,0.23,0.27,1, 10, 100],
              'decision_function_shape':decision_function_shape
              },
              {'C': [0.001, 0.003, 0.006, 0.009, 0.01, 0.04, 0.08, 0.1, 1, 10, 100],
               'kernel': ['poly'],
               'gamma': [0.001, 0.005, 0.1, 0.15, 0.20, 0.23, 0.27, 1, 10, 100],
               'coef0': [0.001, 0.005, 0.1, 0.5, 1, 10, 100],
               'degree': [2, 3, 5],
               'decision_function_shape': decision_function_shape,
               'max_iter':[1000]
               },
              {'C': [0.001, 0.003, 0.006, 0.009, 0.01, 0.04, 0.08, 0.1, 1, 10, 100],
               'kernel': ['sigmoid'],
               'gamma': [0.001, 0.005, 0.1, 0.15, 0.20, 0.23, 0.27, 1, 10, 100],
               'coef0': [0.001, 0.005, 0.1, 0.5, 1, 10, 100],
               'decision_function_shape': decision_function_shape,
               'max_iter':[1000]
               }
              ]
best_parameters={}
best_score={}
cv_results={}
M_score=-1
M_param={}
for parameter in parameters:
    kernel=parameter['kernel'][0]
    print(kernel,"-------------------------------------------")
    clf = SVC(random_state=1)
    grid = GridSearchCV(clf, parameter, cv=10, verbose=1)
    grid.fit(Xtrain, Ytrain.values.ravel())

    best_parameters[kernel]=grid.best_params_
    best_score[kernel]=grid.best_score_
    cv_results[kernel]=grid.cv_results_

    print("{kernel}the best parameter combination is".format(kernel=kernel), grid.best_params_)
    print("{kernel}the best score is ".format(kernel=kernel), grid.best_score_)
    # print("{kernel}the cv_results is ".format(kernel=kernel), grid.cv_results_)

    if grid.best_score_>M_score:
        M_score=grid.best_score_
        M_param=grid.best_params_
w = open(pklname, 'wb')
pickle.dump(cv_results, w)

print("dataname: {dataname}".format(dataname=dataname))
print(" using SVM with best parameters to train*********************************************")
if M_param['kernel']=="linear":
    clf=SVC(random_state=1,C=M_param['C'],decision_function_shape=M_param['decision_function_shape'],kernel='linear')
elif M_param['kernel']=="rbf":
    clf = SVC(random_state=1, C=M_param['C'], decision_function_shape=M_param['decision_function_shape'],
              kernel='rbf',gamma=M_param['gamma'])
elif M_param['kernel']=="poly":
    clf = SVC(random_state=1, C=M_param['C'], decision_function_shape=M_param['decision_function_shape'],
              kernel='poly', gamma=M_param['gamma'],coef0=M_param['coef0'],degree=M_param['degree'],max_iter=1000)
else:
    clf = SVC(random_state=1, C=M_param['C'], decision_function_shape=M_param['decision_function_shape'],
              kernel='sigmoid', gamma=M_param['gamma'], coef0=M_param['coef0'],max_iter=1000)


clf = clf.fit(Xtrain, Ytrain.values.ravel())
print("confusion matrix: ",metrics.confusion_matrix(y_true=Ytest,
    y_pred=clf.predict(Xtest)))
score = cross_val_score(clf, x, y, cv=10)
print("cross_val_score vector: ",score)
score1=score.mean()
print("cross_val_score mean value: ",score1)
# END: OWN CODE

