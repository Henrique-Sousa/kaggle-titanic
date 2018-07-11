import pandas as pd

train_url = ("./data/train.csv")
train = pd.read_csv(train_url)

test_url = ("./data/test.csv")
test = pd.read_csv(test_url)

#print(train.head())
#print(test.head())

train["Age"] = train["Age"].fillna(train["Age"].median())
train["family_size"] = train["SibSp"]+train["Parch"]+1
features_train = train[["Age", "family_size"]].values
labels_train = train["Survived"].values
#print features_train
#print labels_train


test["Age"] = test["Age"].fillna(test["Age"].median())
test["family_size"] = test["SibSp"]+test["Parch"]+1
features_test = test[["Age", "family_size"]].values
#print features_train
#print labels_train

from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

import numpy as np

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred, PassengerId, columns = ["Survived"])

my_solution.to_csv("my_svm_solution_one.csv", index_label = ["PassengerId"])	
print pred
