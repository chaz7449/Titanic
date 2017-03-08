import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

#loading Data, then combine train and test data
trainData = pd.read_csv("~/Documents/Projects/Titanic/Data/train.csv")
testData = pd.read_csv("~/Documents/Projects/Titanic/Data/test.csv")

fullData = trainData.append(testData)
fullData = fullData.reset_index()

'''
Data Exploration

Include null data:
age : 263
cabin : 1014
Embarked : 2
Fare : 1
'''

fullData.info()
fullData.describe()

#charting
fullData["Survived"].value_counts(normalize=True).plot(kind='bar', title = 'Survived')
fullData["Survived"].value_counts().plot(kind='bar', title = 'Survived')
fullData["Age"].hist(bins=20)
fullData["Embarked"].value_counts(normalize=True).plot(kind='bar', title='Embarked')
fullData["Embarked"].value_counts().plot(kind='bar', title='Embarked')
fullData["Fare"].hist(bins=20)
fullData["Parch"].value_counts().plot(kind="bar")
fullData["Pclass"].value_counts(normalize=True).plot(kind='bar')
fullData["Sex"].value_counts(normalize=True).plot(kind='bar')
fullData['SibSp'].value_counts().plot(kind='bar')
sns.FacetGrid(fullData, hue="Survived", size=6) \
   .map(sns.kdeplot, "Age") \
   .add_legend()
sns.FacetGrid(fullData, hue="Survived", size=6) \
    .map(plt.hist, "Age", bins=20) \
    .add_legend()
sns.FacetGrid(fullData, col="Survived", size=6) \
    .map(plt.hist, "Age", bins=20) \
    .add_legend()
sns.boxplot(x="Survived", y= "Age", data=fullData)
sns.stripplot(x="Survived", y="Age", data=fullData, jitter=True, edgecolor="gray")
sns.violinplot(x="Survived", y="Age", data=fullData, size=6)
pd.crosstab(fullData["Embarked"], fullData["Survived"]).plot(kind='bar')
pd.crosstab(fullData["Pclass"], fullData["Survived"]).plot(kind='bar')
pd.crosstab(fullData["Sex"], fullData["Survived"]).plot(kind='bar')
pd.crosstab([fullData["Embarked"], fullData["Pclass"]], fullData["Survived"], normalize='index').plot(kind='bar')
pd.crosstab([fullData["Sex"], fullData["Pclass"]], fullData["Survived"]).plot(kind='bar')
sns.FacetGrid(fullData, row="Pclass", col="Survived", size=6) \
    .map(plt.hist, "Fare", bins=20) \
    .add_legend()
sns.swarmplot(x="Pclass", y="Fare", hue="Survived", data=fullData);
sns.swarmplot(x="Parch", y="Fare", data=fullData);
sns.boxplot(x="Sex", y="Age", hue="Survived", data=fullData);
sns.violinplot(x="Sex", y="Age", hue="Survived", data=fullData, split=True)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=fullData, split=True)
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=fullData);
sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=fullData);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=fullData,
              palette={"male": "g", "female": "m"},
              markers=["^", "o"], linestyles=["-", "--"]);
sns.factorplot(x="Pclass", y="Age", hue="Survived", data=fullData);
sns.factorplot(x="Pclass", y="Fare", hue="Survived", data=fullData);
sns.barplot(x="Pclass", y="SibSp", hue="Survived", data=fullData);
sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=fullData);

'''
Data preprocessing
'''
fullData.isnull().sum()

#Handle Embarked missing value
fullData.loc[fullData["Embarked"].isnull(), ]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=fullData)
fullData.loc[fullData["Embarked"].isnull(), "Embarked"] = "C"

#Handle Fare missing vlaue
fullData.loc[fullData["Fare"].isnull(), ]
fullData.boxplot(column='Fare', by=['Embarked', "Pclass"]);
fullData['Fare'] = fullData["Fare"] \
                    .groupby([fullData["Embarked"], fullData["Pclass"]]) \
                    .apply(lambda x: x.fillna(x.median()))

#Handle Age missing value
sns.distplot(fullData.loc[~fullData["Age"].isnull(), "Age"])

# method 1 : by mean and std
age_mean = fullData["Age"].mean()
age_std = fullData["Age"].std()
count_nan_age = fullData['Age'].isnull().sum()
rand_train = np.random.randint(age_mean - age_std, age_mean + age_std, size=count_nan_age)

fullData["AgeM1"] = fullData["Age"].copy()
fullData["AgeM1"].isnull().sum()
fullData.loc[fullData["AgeM1"].isnull(), "AgeM1"] = rand_train
fullData["AgeM1"].isnull().sum()
sns.distplot(fullData["AgeM1"])

# method2 :  by Pclass, Sex
fullData["AgeM2"] = fullData["Age"].copy()
fullData["AgeM2"] = fullData["AgeM2"] \
                    .groupby([fullData["Pclass"], fullData["Sex"]]) \
                    .apply(lambda x : x.fillna(x.median()))
fullData["AgeM2"].isnull().sum()
sns.distplot(fullData["AgeM2"])
fullData.isnull().sum()

'''
Feature Engineering
'''
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
pd.crosstab(fullData["Sex"], fullData["Title"])

#family size
fullData["Fsize"] = fullData["SibSp"] + fullData["Parch"] + 1
fullData["Family"] = fullData["Surname"] + "_" + fullData["Fsize"].map(str)

pd.crosstab(fullData["Fsize"], fullData["Survived"]).plot(kind='bar');
fullData.loc[fullData["Fsize"] == 1, "FsizeD"] = "singleton"
fullData.loc[(fullData["Fsize"] < 5) & (fullData["Fsize"] > 1), "FsizeD"] = "small"
fullData.loc[fullData["Fsize"] > 4, "FsizeD"] = "large"
pd.crosstab(fullData["FsizeD"], fullData["Survived"]).plot(kind='bar');

sns.distplot(fullData.loc[fullData["Fsize"]==1, "AgeM1"])


fullData.loc[(fullData["Age"]>18) & (fullData["Sex"] == "female") & (fullData["Parch"] > 0), "Survived"].value_counts().plot(kind= 'bar')

fullData.info()
fullData["IsAlone"] = 0
fullData.loc[fullData["Fsize"]==1, "IsAlone"] = 1
pd.crosstab(fullData["IsAlone"], fullData["Survived"]).plot(kind='bar')

fullData["NameLength"] = fullData["Name"].apply(len)

'''
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(fullData["Embarked"].unique())
fullData["Embarked"] = le.transform(fullData["Embarked"])
fullData["Embarked"] = le.inverse_transform(fullData["Embarked"])
'''

# Model selection
fullData = pd.get_dummies(fullData, columns=["Embarked", "FsizeD", "Sex", "Title"])

trainData = fullData.loc[0:trainData.shape[0]-1]
testData  = fullData.loc[trainData.shape[0]:]

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

feature = ["Pclass", "AgeM1", "SibSp", "Parch", "Fare", "Fsize", "Embarked_C", "Embarked_Q", "Embarked_S",
            "FsizeD_large", "FsizeD_singleton", "FsizeD_small", "Sex_female", "Sex_male",
            "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare Title", "IsAlone"]
target = "Survived"

stdsc = StandardScaler()
stdsc.fit(trainData[["AgeM1", "Fare", "NameLength", "AgeM2"]]);
trainData[["AgeM1", "Fare", "NameLength", "AgeM2"]].values = stdsc.transform(trainData[["AgeM1", "Fare", "NameLength", "AgeM2"]])
testData[["AgeM1", "Fare", "NameLength", "AgeM2"]].values = stdsc.transform(testData[["AgeM1", "Fare", "NameLength", "AgeM2"]])

stdTrain = stdsc.transform(trainData[["AgeM1", "Fare", "NameLength", "AgeM2"]])

def splitXYData(features, target, data):
    return data[features], data[target]

x,y = splitXYData(feature, target, trainData)

print(cross_val_score(LogisticRegression(), x, y, cv=5, scoring="f1").mean())
print(cross_val_score(DecisionTreeClassifier(criterion="entropy", max_depth=4), x, y, cv=5, scoring="f1").mean())
print(cross_val_score(SVC(kernel='linear'), x, y, cv=5, scoring="f1").mean())
print(cross_val_score(KNeighborsClassifier(n_neighbors=5), x, y, cv=5, scoring="f1").mean())
print(cross_val_score(RandomForestClassifier(n_estimators=150, min_samples_split=3, min_samples_leaf=3), x, y, cv=5, scoring="f1").mean())
print(cross_val_score(SGDClassifier(loss="log"), x, y, cv=5, scoring="f1").mean())
print(cross_val_score(GaussianNB(), x, y, cv=5, scoring="f1").mean())

parameters = {'kernel':('linear', 'rbf'), 'C':np.arange(1, 10, 0.5)}
clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=2, scoring="f1")
clf.fit(x, y)
best = clf.best_estimator_
best.fit(x, y)
best.score(x, y)
print(cross_val_score(best, x, y, cv=5).mean())

from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf3 = SVC(kernel='linear')
clf4 = KNeighborsClassifier(n_neighbors=5)
eclf = VotingClassifier([('lg', clf1), ('tree', clf2), ('svc', clf3)])
eclf.fit(x, y)
print(cross_val_score(eclf, x, y, cv=5).mean())

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=500, max_samples=0.8, max_features=0.8)
print(cross_val_score(bag, x, y, cv=5, scoring="f1").mean())

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=500, learning_rate=1.0)
print(cross_val_score(ada, x, y, cv=5).mean())

parameters = {'penalty': ('l1', 'l2'), 'C' : np.arange(0.1, 1, 0.1)}
log = GridSearchCV(LogisticRegression(), parameters, cv=5, n_jobs=4, scoring="f1")
log.fit(x, y)
log = log.best_estimator_
print(cross_val_score(log, x, y, cv=5, scoring="f1").mean())

parameters = {'max_features': np.arange(0.1, 1, 0.1), 'max_depth': np.arange(2, 8, 1), \
                'min_samples_split': np.arange(2, 10, 1), 'max_leaf_nodes': np.arange(2, 10, 1), \
                'criterion': ('gini', 'entropy')}
tree = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_jobs=4)
tree.fit(x, y)
tree = tree.best_estimator_
print(cross_val_score(tree, x, y, cv=5, scoring="f1").mean())

parameters = {'kernel':('linear', 'rbf'), 'C':np.arange(1, 2, 0.5)}
svm = GridSearchCV(SVC(), parameters, cv=5, n_jobs=4, scoring="f1")
svm.fit(x, y)
svm = svm.best_estimator_
print(cross_val_score(svm, x, y, cv=5, scoring="f1").mean())

parameters = {'n_neighbors': np.arange(2, 7, 1), 'algorithm': ('ball_tree', 'kd_tree', 'brute')}
knn = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, n_jobs=4, scoring="f1")
knn.fit(x, y)
knn = knn.best_estimator_
print(cross_val_score(knn, x, y, cv=5).mean())

import time

parameters = {'criterion': ('gini', 'entropy'), \
                'max_features': np.arange(0.1, 1, 0.1), 'max_depth': np.arange(2, 8, 1), \
                'min_samples_split': np.arange(2, 10, 1)}
rf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=8)
start_time = time.clock()
rf.fit(x, y)
print(time.clock() - start_time, "minute")
rf = rf.best_estimator_
print(cross_val_score(rf, x, y, cv=5, scoring="f1").mean())

gaus = GaussianNB()
print(cross_val_score(gaus, x, y, cv=5).mean())


logTrain = log.predict(x)
treeTrain = tree.predict(x)
svmTrain = svm.predict(x)
rfTrain = rf.predict(x)

x_second = np.concatenate((logTrain.reshape(-1, 1), treeTrain.reshape(-1, 1), svmTrain.reshape(-1, 1), rfTrain.reshape(-1, 1)), axis=1)

import xgboost as xgb
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_second, y)
gbm.score(x_second, y)

from sklearn.metrics import f1_score
print(f1_score(y, gbm.predict(x_second)))

reg = LinearRegression()
reg.fit(x_second, y)
reg_prediction = reg.predict(x_second)
reg_prediction[reg_prediction > 0.5] = 1
reg_prediction[reg_prediction <= 0.5] = 0
print(f1_score(y, reg_prediction))

xTest, yTest = splitXYData(feature, target, testData)

logTest = log.predict(xTest)
treeTest = tree.predict(xTest)
svmTest = svm.predict(xTest)
rfTest = rf.predict(xTest)

xTest_second = np.concatenate((logTest.reshape(-1, 1), treeTest.reshape(-1, 1), svmTest.reshape(-1, 1), rfTest.reshape(-1, 1)), axis=1)
predictions = gbm.predict(xTest_second)
reg_predictions = reg.predict(xTest_second)
reg_prediction[reg_prediction > 0.5] = 1
reg_prediction[reg_prediction <= 0.5] = 0

StackingSubmission = pd.DataFrame({ 'PassengerId': testData["PassengerId"],
                            'Survived': predictions })
StackingSubmission.to_csv("aaa.csv", index=False)


bag = BaggingClassifier(base_estimator=svm, n_estimators=500, max_samples=0.8, max_features=0.8)
print(cross_val_score(bag, x, y, cv=5, scoring="f1").mean())
