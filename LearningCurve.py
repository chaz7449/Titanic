import numpy as np
import pandas as pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
import plotly.offline as py
from plotly.graph_objs import *


titanicData = pandas.read_csv("~/Documents/PycharmProjects/Titanic/train.csv")
titanicData["Age_New"] = titanicData["Age"].fillna(titanicData["Age"].median())
titanicData.loc[titanicData["Sex"] == "male", "Sex_New"] = 0
titanicData.loc[titanicData["Sex"] == "female", "Sex_New"] = 1
titanicData["Embarked_New"] = titanicData["Embarked"].fillna('S')
titanicData.loc[titanicData["Embarked_New"] == 'S', "Embarked_New"] = 0
titanicData.loc[titanicData["Embarked_New"] == 'C', "Embarked_New"] = 1
titanicData.loc[titanicData["Embarked_New"] == 'Q', "Embarked_New"] = 2
feature = ["Pclass", "Sex_New", "Age_New", "SibSp", "Parch", "Fare", "Embarked_New"]
target = "Survived"

reg = LinearRegression()
reg.fit(titanicData[feature], titanicData[target])
print(reg._residues)
sum((reg.predict(titanicData[feature]) - titanicData[target]).apply(lambda x: pow(x, 2)))

train_sizes, train_scores, test_scores = learning_curve(reg, titanicData[feature], titanicData[target], n_jobs=3,
                                                        train_sizes=np.linspace(.1, 1.0, 100))


