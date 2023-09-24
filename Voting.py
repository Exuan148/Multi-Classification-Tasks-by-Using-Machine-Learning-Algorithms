#-*- coding = utf-8 -*-
#@Time : 2023-01-06 21:48
#@File : Voting.py
#@Software: PyCharm
#@Author:HanYixuan
from DataPreprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import SVC
from Logger import Logger
import sys
import os
import time
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
dataname="Credit"
# get dataset from original format
x,y,feature_name,class_labels=Preprocessing(dataname)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3,random_state=22)

# rectify each dataset's index
for i in [Xtrain,Xtest,Ytrain,Ytest]:
    i.index=range(i.shape[0])

print("dataname: {dataname}".format(dataname=dataname))

print(" using voting grid-search method*********************************************")
parameters = {"voting": ['soft','hard']}
dtr_clf=DecisionTreeClassifier(random_state=1,criterion= 'entropy',max_depth= 3,min_impurity_decrease= 0.0,min_samples_leaf= 16,splitter= 'random')
svm_clf=SVC(probability=True, C= 0.006, coef0= 1, decision_function_shape= 'ovo', degree= 5, gamma= 0.2, kernel= 'poly', max_iter= 1000)
clf=VotingClassifier(estimators=[('dtr',dtr_clf),
                             ('svc',svm_clf)])
grid = GridSearchCV(clf, parameters, cv=10,verbose=1)
grid.fit(Xtrain, Ytrain.values.ravel())
print("the best parameter combination is",grid.best_params_)
print("the best score is ",grid.best_score_)
print("the cv_results is ",grid.cv_results_)


print(" using voting with best parameters to train****************")
clf = VotingClassifier(estimators=[('dtr',dtr_clf),
                             ('svc',svm_clf)],
                 voting='soft')
clf = clf.fit(Xtrain, Ytrain.values.ravel())
print("confusion matrix: ",metrics.confusion_matrix(y_true=Ytest,
    y_pred=clf.predict(Xtest)))
score = cross_val_score(clf, x, y, cv=10)
print("cross_val_score vector: ",score)
score1=score.mean()
print("cross_val_score mean value: ",score1)
# END: OWN CODE