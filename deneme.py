#!/usr/bin/env python3


# @Author: Huseyin Tutan

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

# read in the data
#===========================================================================
train_data = pd.read_csv('train.csv')
test_data  = pd.read_csv('test.csv')
solution   = pd.read_csv('submission1.csv')

#===========================================================================
# select some features
#===========================================================================
features = ["Pclass", "Sex", "SibSp", "Parch"]

X       = pd.get_dummies(train_data[features])
y       = train_data["Survived"]
final_X_test  = pd.get_dummies(test_data[features])

# perform the classification and the fit
classifier = RandomForestClassifier(random_state=4)
classifier.fit(X, y)

predictions = classifier.predict(final_X_test)

K_splits = 11

# calculate the scores
CV_scores = cross_val_score(classifier, X, y, cv=K_splits)

# Print Section & Visualize

print("The mean accuracy score of the train data is %.5f" % classifier.score(X, y))
print("The mean  cross-validation   score is %.5f ± %0.2f" % (CV_scores.mean(), CV_scores.std() * 2))
print("The test (i.e. leaderboard)  score is %.5f" % accuracy_score(solution['Survived'],predictions))

print("The individual cross-validation scores are: \n",CV_scores)
print("The minimum cross-validation score is %.3f" % min(CV_scores))
print("The maximum cross-validation score is %.3f" % max(CV_scores))

cv = StratifiedKFold(n_splits=K_splits)
visualizer = CVScores(classifier, cv=cv, scoring='f1_weighted',size=(1200, 400))
visualizer.fit(X, y)
visualizer.show()