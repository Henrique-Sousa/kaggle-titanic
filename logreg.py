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
from sklearn.linear_model import LogisticRegression


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


#logistic regression
X_train = train.drop(['PassengerId', 'Survived'], axis=1)
y_train = train['Survived']

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#test fare: nan -> mean
fare_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
test['Fare'] = fare_imputer.fit_transform(test['Fare'].values.reshape(-1,1))

y_pred = classifier.predict(test.drop(['PassengerId'], axis=1))


ids = test['PassengerId']
surv = pd.DataFrame(y_pred, columns=['Survived'])
submission = pd.concat([ids, surv], axis=1)
submission.to_csv('logistic_regression_1.csv', index=False)