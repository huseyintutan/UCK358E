#!/usr/bin/env python3


# @Author: Huseyin Tutan

# Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the data

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

from sklearn.ensemble import RandomForestClassifier
y = data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(data[features])
X_test = pd.get_dummies(test[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

