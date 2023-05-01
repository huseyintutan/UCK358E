#!/usr/bin/env python3

# @Author: Huseyin Tutan 110190021

# IMPORT THE LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorama

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from colorama import Fore
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.model_selection import CVScores
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# IMPORT THE DATA
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# MISS VALUES

data['Age'].fillna(data['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)


# CODE SECTION

data2 = [data, test]
for dataset in data2:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'
    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'

y = data["Survived"]
features = ["Pclass", "Sex", "Age", "Embarked", "travelled_alone"]
X = pd.get_dummies(data[features])
X_test = pd.get_dummies(test[features])

# PARAMETER SELECTOR

# param_grid = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [3, 4, 5],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'n_jobs': [-1],
#     'random_state':[0, 100, 500, 5000]
# }

# model = RandomForestClassifier(random_state=42)

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# grid_search.fit(X, y)

# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

model = RandomForestClassifier(n_estimators=100,
                               max_depth=4,
                               min_samples_split=10,
                               min_samples_leaf=2,
                               random_state=5000,
                               n_jobs=-1)
# model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=1)
# model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma=0.1, random_state=1))
# model = GaussianNB()
# model = LogisticRegression()


model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame(
    {'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission1.csv', index=False)

# COMPUTE CROSS-VALIDATION SCORE

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())

# COMPUTE ACCURACY SCORE

train_predictions = model.predict(X)
accuracy = accuracy_score(y, train_predictions)
print("Train accuracy:", accuracy)

