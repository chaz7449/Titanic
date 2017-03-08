import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from statsmodels.graphics.mosaicplot import mosaic
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

train = pd.read_csv("~/Documents/Projects/Titanic/Data/train.csv")
test = pd.read_csv("~/Documents/Projects/Titanic/Data/test.csv")

fullData = train.append(test)
fullData = fullData.reset_index()

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

fullData["Title"] = fullData["Name"].apply(lambda x : get_title(x))
fullData["Surname"] = fullData["Name"].apply(lambda x: get_surname(x))

pd.crosstab(fullData["Sex"], fullData["Title"])
rare_title = ('Dona', 'Lady', 'Countess','Capt', 'Col', 'Don',
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
fullData["Title"] = fullData["Title"].apply(lambda x: "Rare Title" if x in rare_title else x)
fullData.loc[fullData["Title"] == "Mlle", "Title"] = "Miss"
fullData.loc[fullData["Title"] == "Ms", "Title"] = "Miss"
fullData.loc[fullData["Title"] == "Mme", "Title"] = "Mrs"

# feature chart
fig = plt.figure(figsize=(9, 9))
fig_dims = (3, 2)
plt.subplot2grid(fig_dims, (0, 0))
fullData['Survived'].value_counts().plot(kind='bar', title='Death and Survival Counts');
plt.subplot2grid(fig_dims, (0, 1))
fullData['Pclass'].value_counts().plot(kind='bar', title='Passenger Class Counts');
plt.subplot2grid(fig_dims, (1, 0))
fullData['Sex'].value_counts().plot(kind='bar', title='Gender Counts');
plt.xticks(rotation=0)
plt.subplot2grid(fig_dims, (1, 1))
fullData['Embarked'].value_counts().plot(kind='bar', title='Ports of Embarkation Counts');
plt.subplot2grid(fig_dims, (2, 0))
fullData['Age'].hist();
plt.title('Age Histogram')

# family
fullData["Fsize"] = fullData["SibSp"] + fullData["Parch"] + 1
fullData["Family"] = fullData["Surname"] + "_" + fullData["Fsize"].map(str)

pd.crosstab(fullData["Fsize"], fullData["Survived"]).plot(kind='bar');
fullData.loc[fullData["Fsize"] == 1, "FsizeD"] = "singleton"
fullData.loc[(fullData["Fsize"] < 5) & (fullData["Fsize"] > 1), "FsizeD"] = "small"
fullData.loc[fullData["Fsize"] > 4, "FsizeD"] = "large"
mosaic(fullData, ['FsizeD', 'Survived']);

fullData.loc[fullData["Embarked"].isnull(), ]
fullData.boxplot(column='Fare', by=['Embarked', "Pclass"]);
fullData.loc[61, "Embarked"] = "C"
fullData.loc[829, "Embarked"] = "C"

fullData.loc[fullData["Sex"] == "male", "Sex_val"] = 0
fullData.loc[fullData["Sex"] == "female", "Sex_val"] = 1

fullData.loc[fullData["Embarked"] == "C", "Embarked_val"] = 0
fullData.loc[fullData["Embarked"] == "Q", "Embarked_val"] = 1
fullData.loc[fullData["Embarked"] == "S", "Embarked_val"] = 2

fullData.loc[fullData["FsizeD"] == "singleton", "FsizeD_val"] = 0
fullData.loc[fullData["FsizeD"] == "small", "FsizeD_val"] = 1
fullData.loc[fullData["FsizeD"] == "large", "FsizeD_val"] = 2

fullData.loc[fullData["Title"] == "Master", "Title_val"] = 0
fullData.loc[fullData["Title"] == "Miss", "Title_val"] = 1
fullData.loc[fullData["Title"] == "Mr", "Title_val"] = 2
fullData.loc[fullData["Title"] == "Mrs", "Title_val"] = 3
fullData.loc[fullData["Title"] == "Rare Title", "Title_val"] = 4

fullData['Fare'] = fullData['Fare'] \
                        .groupby([fullData['Embarked'], fullData['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))

plt.hist(fullData.loc[~fullData['Age'].isnull(), "Age"], normed=1);

'''
feature = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Fsize", "Embarked_val",
            "FsizeD_val", "Sex_val", "Title_val"]

knn = KNN(3).complete(fullData[feature])
knn = pd.DataFrame(knn,columns=feature)
plt.hist(knn['Age'], normed=1);

nnm = NuclearNormMinimization().complete(fullData[feature])
nnm = pd.DataFrame(nnm,columns=feature)
plt.hist(nnm['Age'], normed=1);

softimpute = SoftImpute().complete(fullData[feature])
softimpute = pd.DataFrame(softimpute,columns=feature)
plt.hist(softimpute['Age'], normed=1);

fullData['AgeFill'] = knn['Age']
'''


fullData['AgeFill'] = fullData['Age']
fullData['AgeFill'] = fullData['AgeFill'] \
                        .groupby([fullData['Sex'], fullData['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))
fig = plt.figure(figsize=(9, 6))
fig_dims = (1, 2)
plt.subplot2grid(fig_dims, (0, 0))
plt.hist(fullData['AgeFill'], normed=1)
plt.subplot2grid(fig_dims, (0, 1))
plt.hist(fullData.loc[~fullData['Age'].isnull(), "Age"], normed=1)

feature = ["Pclass", "AgeFill", "SibSp", "Parch", "Fare", "Fsize", "Embarked_val",
            "FsizeD_val", "Sex_val", "Title_val"]
target = "Survived"

from sklearn import preprocessing
normalizeDate = preprocessing.scale(fullData[feature])
normalizeDate = pd.DataFrame(normalizeDate, columns=feature)
fullData[feature] = normalizeDate[feature]

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier

trainData = fullData.loc[0:train.shape[0]-1, ]
testData  = fullData.loc[train.shape[0]:, ]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
alg.fit(trainData[feature], trainData[target])
featurePoints = dict(zip(feature, alg.feature_importances_))
ind = np.arange(feature.__len__())
width = 0.35
fig = plt.figure(figsize=(8, 8))
plt.bar(ind, alg.feature_importances_)
plt.xticks(ind + width/2., feature, rotation=60)

feature = ["Pclass", "AgeFill", "Sex_val", "Title_val", "Fare"]
log = LogisticRegression()
scores = cross_val_score(log, trainData[feature], trainData[target], cv=3)
print(scores.mean())

rdf = RandomForestClassifier(n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_val_score(rdf, trainData[feature], trainData[target], cv=5)
print(scores.mean())

sgd = SGDClassifier(loss="perceptron")
scores = cross_val_score(sgd, trainData[feature], trainData[target], cv=5)
print(scores.mean())

rdf = RandomForestClassifier(n_estimators=150, min_samples_split=4, min_samples_leaf=2)
rdf.fit(trainData[feature], trainData[target])
predictions = rdf.predict(testData[feature])
predictions = predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)
