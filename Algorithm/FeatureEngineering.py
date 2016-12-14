import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv("~/Documents/PycharmProjects/Titanic/train.csv")

# Name
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def get_surname(name):
    surname_search = re.search('([A-Za-z]+),', name)
    if surname_search:
        return surname_search.group(1)
    return ""

train["Title"] = train["Name"].apply(lambda x: get_title(x))
train["Surname"] = train["Name"].apply(lambda x: get_surname(x))

pd.crosstab(train["Sex"], train["Title"])
rare_title = ('Dona', 'Lady', 'Countess','Capt', 'Col', 'Don',
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
train["Title"] = train["Title"].apply(lambda x: "Rare Title" if x in rare_title else x)
train.loc[train["Title"] == "Mlle", "Title"] = "Miss"
train.loc[train["Title"] == "Ms", "Title"] = "Miss"
train.loc[train["Title"] == "Mme", "Title"] = "Mrs"

# feature chart
fig = plt.figure(figsize=(9, 9))
fig_dims = (3, 2)
plt.subplot2grid(fig_dims, (0, 0))
train['Survived'].value_counts().plot(kind='bar', title='Death and Survival Counts')
plt.subplot2grid(fig_dims, (0, 1))
train['Pclass'].value_counts().plot(kind='bar', title='Passenger Class Counts')
plt.subplot2grid(fig_dims, (1, 0))
train['Sex'].value_counts().plot(kind='bar', title='Gender Counts')
plt.xticks(rotation=0)
plt.subplot2grid(fig_dims, (1, 1))
train['Embarked'].value_counts().plot(kind='bar', title='Ports of Embarkation Counts')
plt.subplot2grid(fig_dims, (2, 0))
train['Age'].hist()
plt.title('Age Histogram')

# family
train["Fsize"] = train["SibSp"] + train["Parch"] + 1
train["Family"] = train["Surname"] + "_" + train["Fsize"].map(str)

pd.crosstab(train["Fsize"], train["Survived"]).plot(kind='bar')
train.loc[train["Fsize"] == 1, "FsizeD"] = "singleton"
train.loc[(train["Fsize"] < 5) & (train["Fsize"] > 1), "FsizeD"] = "small"
train.loc[train["Fsize"] > 4, "FsizeD"] = "large"
mosaic(train, ['FsizeD', 'Survived'])

# Missing value
train.loc[61, ]
train.boxplot(column='Fare', by=['Embarked', "Pclass"])
train.loc[61, "Embarked"] = "C"
train.loc[829, "Embarked"] = "S"

sum(train["Age"].isnull())

train['AgeFill'] = train['Age']
train['AgeFill'] = train['AgeFill'] \
                        .groupby([train['Sex'], train['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))

fig = plt.figure(figsize=(9, 6))
fig_dims = (1, 2)
plt.subplot2grid(fig_dims, (0, 0))
plt.hist(train['AgeFill'], normed=1)
plt.subplot2grid(fig_dims, (0, 1))
plt.hist(train.loc[~train['Age'].isnull(), "Age"], normed=1)

train = pd.get_dummies(train, columns=["Embarked", "FsizeD", "Sex", "Title"])

feature = ["Pclass", "AgeFill", "SibSp", "Parch", "Fare", "Fsize", "Embarked_C", "Embarked_Q", "Embarked_S",
            "FsizeD_large", "FsizeD_singleton", "FsizeD_small", "Sex_female", "Sex_male",
            "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare Title"]
target = "Survived"

log = LogisticRegression()
log.fit(train[feature], train[target])
train_prediction = log.predict(train[feature])
accuracy = sum(train_prediction == train["Survived"]) / len(train_prediction)
print(accuracy)

reg = LinearRegression()
reg.fit(train[feature], train[target])
train_prediction = reg.predict(train[feature])
train_prediction[train_prediction > .6] = 1
train_prediction[train_prediction <= .6] = 0
accuracy = sum(train_prediction == train["Survived"]) / len(train_prediction)
print(accuracy)


alg = LogisticRegression()
kf = KFold(n_splits=3, random_state=1)
predictions = []
for trainIdx, testIdx in kf.split(train):
    train_predictors = (train[feature].iloc[trainIdx, :])
    train_target = train["Survived"].iloc[trainIdx]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(train[feature].iloc[testIdx, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions == train["Survived"]) / len(predictions)
print(accuracy)

trainData, testData = train_test_split(train, test_size=0.2)
log = LogisticRegression()
log.fit(trainData[feature], trainData[target])
test_prediction = log.predict(testData[feature])
accuracy = sum(test_prediction == testData["Survived"]) / len(test_prediction)
print(accuracy)

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_val_score(alg, train[feature], train["Survived"], cv=3)
print(scores.mean())

alg = LogisticRegression()
scores = cross_val_score(alg, train[feature], train["Survived"], cv=3)
print(scores.mean())



alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
alg.fit(train[feature], train[target])
featurePoints = dict(zip(feature, alg.feature_importances_))
ind = np.arange(feature.__len__())
width = 0.35
fig = plt.figure(figsize=(8, 8))
plt.bar(ind, alg.feature_importances_)
plt.xticks(ind + width/2., feature, rotation=90)


train.loc[train["Sex"] == "male", "Sex_val"] = 0
train.loc[train["Sex"] == "female", "Sex_val"] = 1

train.loc[train["Embarked"] == "C", "Embarked_val"] = 0
train.loc[train["Embarked"] == "Q", "Embarked_val"] = 1
train.loc[train["Embarked"] == "S", "Embarked_val"] = 2

train.loc[train["FsizeD"] == "singleton", "FsizeD_val"] = 0
train.loc[train["FsizeD"] == "small", "FsizeD_val"] = 1
train.loc[train["FsizeD"] == "large", "FsizeD_val"] = 2

train.loc[train["Title"] == "Master", "Title_val"] = 0
train.loc[train["Title"] == "Miss", "Title_val"] = 1
train.loc[train["Title"] == "Mr", "Title_val"] = 2
train.loc[train["Title"] == "Mrs", "Title_val"] = 3
train.loc[train["Title"] == "Rare Title", "Title_val"] = 3

feature = ["Pclass", "AgeFill", "SibSp", "Parch", "Fare", "Fsize", "Embarked_val",
            "FsizeD_val", "Sex_val", "Title_val"]
target = "Survived"

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
alg.fit(train[feature], train[target])
featurePoints = dict(zip(feature, alg.feature_importances_))
ind = np.arange(feature.__len__())
width = 0.35
fig = plt.figure(figsize=(8, 8))
plt.bar(ind, alg.feature_importances_)
plt.xticks(ind + width/2., feature, rotation=60)

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_val_score(alg, train[feature], train["Survived"], cv=3)
print(scores.mean())

feature = ["Pclass", "AgeFill", "Sex_val", "Title_val", "Fare"]

log = LogisticRegression()
scores = cross_val_score(log, train[feature], train[target], cv=3)
print(scores.mean())

rdf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_val_score(rdf, train[feature], train[target], cv=3)
print(scores.mean())

feature = ["Pclass", "AgeFill", "Fare", "Sex_female", "Sex_male",
            "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare Title"]


log = LogisticRegression()
scores = cross_val_score(log, train[feature], train[target], cv=3)
print(scores.mean())

rdf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=5, min_samples_leaf=4)
scores = cross_val_score(rdf, train[feature], train[target], cv=3)
print(scores.mean())

train.loc[train["Title"] != "Mr", "Title_val"] = 0
train.loc[train["Title"] == "Mr", "Title_val"] = 1
