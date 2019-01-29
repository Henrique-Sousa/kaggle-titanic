#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:39:51 2019

@author: Henrique Sousa
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

folder = "./data/"
test_path = folder + "test.csv"
train_path = folder + "train.csv"

test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

#train_old = train.copy()
#test_old = test.copy()

#test if there is NaNs
train.isnull().values.any()
test.isnull().values.any()


def preprocess(X):
    #drop from dataset where Embarked==NaN
    X = X[X['Embarked'].notna()]    

    #change age==nan to age==mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X['Age'] = imputer.fit_transform(X['Age'].values.reshape(-1,1))
    
    encoder = LabelEncoder()
    X['Embarked'] = encoder.fit_transform(X['Embarked'])
    X['Sex'] = encoder.fit_transform(X['Sex'])

    X = X.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    return X

train = preprocess(train)
test = preprocess(test)

X_train = train.drop(['PassengerId', 'Survived'], axis=1)
y_train = train['Survived']


#cross validation linear vs poly
lin_svc = SVC(kernel='linear', gamma='auto')
lin_svc.fit(X_train, y_train)
#lin_svc_scores = cross_val_score(linear_svc, X_train, y_train, cv=7)

#poly_svc = SVC(kernel='poly', gamma='auto')
#poly_svc.fit(X_train, y_train)
#poly_svc_scores = cross_val_score(poly_svc, X_train, y_train, cv=7)

#testing models
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

forest = RandomForestClassifier()
forest.fit(X_train, y_train)


lin_svc_pred = lin_svc.predict(X_train)
lin_svc_acc = accuracy_score(y_train, lin_svc_pred)

#poly_svc_pred = poly_svc.predict(X_train)
#poly_svc_cm = confusion_matrix(y_train, poly_svc_pred)

knn_pred = knn.predict(X_train)
knn_acc = accuracy_score(y_train, knn_pred)

gnb_pred = gnb.predict(X_train)
gnb_acc = accuracy_score(y_train, gnb_pred)

tree_pred = tree.predict(X_train)
tree_acc = accuracy_score(y_train, tree_pred)

forest_pred = forest.predict(X_train)
forest_acc = accuracy_score(y_train, forest_pred)

print("Linear SVC accuracy: ", lin_svc_acc)
print("k-NN accuracy: ", knn_acc)
print("Naive Bayes accuracy: ", gnb_acc)
print("Decision Tree accuracy: ", tree_acc)
print("Random Forest accuracy: ", forest_acc)


#cross validation on models

log_reg_cv = LogisticRegression()
log_reg_scores = cross_val_score(log_reg_cv, X_train, y_train, cv=7)

svc_cv = SVC()
svc_scores = cross_val_score(svc_cv, X_train, y_train, cv=7)

knn_cv = KNeighborsClassifier()
knn_scores = cross_val_score(knn_cv, X_train, y_train, cv=7)

gnb_cv = GaussianNB()
gnb_scores = cross_val_score(gnb_cv, X_train, y_train, cv=7)

tree_cv = DecisionTreeClassifier()
tree_scores = cross_val_score(tree_cv, X_train, y_train, cv=7)

forest_cv = RandomForestClassifier(n_estimators=10, criterion="gini")
forest_scores = cross_val_score(forest_cv, X_train, y_train, cv=7)

print("Logistic Regression CV mean accuracy: ", log_reg_scores.mean())
print("SVC CV mean accuracy: ", svc_scores.mean())
print("k-NN CV mean accuracy: ", knn_scores.mean())
print("Naive Bayes CV mean accuracy: ", gnb_scores.mean())
print("Decision Tree CV mean accuracy: ", tree_scores.mean())
print("Random Forest CV mean accuracy: ", forest_scores.mean())



#=============================================

##grid search test for linearity
#param_grid = {'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}
#param_grid = {'kernel':['linear', 'poly']}
#grid_search_svc = SVC()
#grid_search = GridSearchCV(grid_search_svc, param_grid, cv=7)
#grid_search.fit(X_train, y_train)

#=================================================

##grid search for parameter tuning on random forest
param_grid = {'criterion':['entropy', 'gini'], 'n_estimators':list(range(1,20))}
grid_search_forest = RandomForestClassifier()
grid_search = GridSearchCV(grid_search_forest, param_grid, cv=7)
grid_search.fit(X_train, y_train)

#=================================================


#test fare: nan -> mean
fare_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
test['Fare'] = fare_imputer.fit_transform(test['Fare'].values.reshape(-1,1))


log_reg_clf = LogisticRegression()
log_reg_clf.fit(X_train, y_train)

forest_clf = RandomForestClassifier(n_estimators=14)
forest_clf.fit(X_train, y_train)


forest_pred = forest_clf.predict(test.drop(['PassengerId'], axis=1))
log_reg_pred = log_reg_clf.predict(test.drop(['PassengerId'], axis=1))

#kaggle submissions
ids = test['PassengerId']

log_reg_surv = pd.DataFrame(log_reg_pred, columns=['Survived'])
log_reg_submission = pd.concat([ids, log_reg_surv], axis=1)
log_reg_submission.to_csv('log_reg_1.csv', index=False)

forest_surv = pd.DataFrame(forest_pred, columns=['Survived'])
forest_submission = pd.concat([ids, forest_surv], axis=1)
forest_submission.to_csv('random_forest.csv', index=False)

