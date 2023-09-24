#-*- coding = utf-8 -*-
#@Time : 2023-01-03 21:11
#@File : DecisionTree.py
#@Software: PyCharm
#@Author:HanYixuan

from DataPreprocessing import Preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import time
from sklearn.tree import export_graphviz
from Logger import Logger
import warnings

warnings.filterwarnings("ignore")
log_path = 'log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)

# START: OWN CODE
# names
dataname="Algerian_Forest_Fire"
pklname='result/DTree_{dataname}.pkl'.format(dataname=dataname)
dotname="img/{dataname}.dot".format(dataname=dataname)
best_dotname="img/{dataname}_best.dot".format(dataname=dataname)

# get dataset from original format
x,y,feature_name,class_labels=Preprocessing(dataname)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3,random_state=22)

# rectify each dataset's index
for i in [Xtrain,Xtest,Ytrain,Ytest]:
    i.index=range(i.shape[0])

print("dataname: {dataname}".format(dataname=dataname))
print(" using decision tree with default parameters to train****************")
clf = DecisionTreeClassifier(random_state=1)
clf = DecisionTreeClassifier(criterion='gini',max_depth=2,min_impurity_decrease=0,min_samples_leaf=1,splitter='best',random_state=1)
clf = clf.fit(Xtrain, Ytrain)
print("confusion matrix: ",metrics.confusion_matrix(y_true=Ytest,
    y_pred=clf.predict(Xtest)))
score = cross_val_score(clf, x, y, cv=10)
print("cross_val_score vector: ",score)
score1=score.mean()
print("cross_val_score mean value: ",score1)
print("feature_importances_:",clf.feature_importances_)

# visualize decision tree
with open(dotname, 'w') as f:
    f = export_graphviz(clf, feature_names=feature_name,class_names=class_labels, out_file=f)

print(" using grid-search method*********************************************")
parameters = {'splitter': ('best', 'random')
    , 'criterion': ("gini", "entropy")
    , "max_depth": [*range(1, 10)]
    , 'min_samples_leaf': [*range(1, 50, 5)]
    , 'min_impurity_decrease': [*np.linspace(0, 0.5, 20)]
}
clf=DecisionTreeClassifier(random_state=1)
grid = GridSearchCV(clf, parameters, cv=10,verbose=1)
grid.fit(Xtrain, Ytrain)
print("the best parameter combination is",grid.best_params_)
print("the best score is ",grid.best_score_)
print("the cv_results is ",grid.cv_results_)
w = open(pklname, 'wb')
pickle.dump(grid.cv_results_, w)

print(" using decision tree with best parameters to train****************")
clf = DecisionTreeClassifier(criterion=grid.best_params_['criterion'],max_depth=grid.best_params_['max_depth'],
                             min_impurity_decrease=grid.best_params_['min_impurity_decrease'],
                             min_samples_leaf=grid.best_params_['min_samples_leaf'],
                             splitter=grid.best_params_['splitter'],random_state=1)
clf = clf.fit(Xtrain, Ytrain)
print("confusion matrix: ",metrics.confusion_matrix(y_true=Ytest,
    y_pred=clf.predict(Xtest)))
score = cross_val_score(clf, x, y, cv=10)
print("cross_val_score vector: ",score)
score1=score.mean()
print("cross_val_score mean value: ",score1)
print("feature_importances_:",clf.feature_importances_)

# visualize decision tree
with open(best_dotname, 'w') as f:
    f = export_graphviz(clf, feature_names=feature_name,class_names=class_labels, out_file=f)
# END: OWN CODE