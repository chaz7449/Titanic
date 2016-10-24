import pandas as pandas
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

trainData = pandas.read_csv("~/Documents/PycharmProjects/Titanic/train.csv")
print(trainData.describe(include="all"))
print(trainData.describe())

trainData["Age_New"] = trainData["Age"].fillna(trainData["Age"].median())

trainData.loc[trainData["Sex"] == "male", "Sex_New"] = 0
trainData.loc[trainData["Sex"] == "female", "Sex_New"] = 1

trainData["Embarked_New"] = trainData["Embarked"].fillna('S')
trainData.loc[trainData["Embarked_New"] == 'S', "Embarked_New"] = 0
trainData.loc[trainData["Embarked_New"] == 'C', "Embarked_New"] = 1
trainData.loc[trainData["Embarked_New"] == 'Q', "Embarked_New"] = 2

feature = ["Pclass", "Sex_New", "Age_New", "SibSp", "Parch", "Fare", "Embarked_New"]

# regression accuracy = 0.81
train, test = train_test_split(trainData, test_size=0.2)
reg_m1 = LinearRegression()
reg_m1.fit(train[feature], train["Survived"])
reg_m1_prediction = reg_m1.predict(test[feature])
reg_m1_prediction[reg_m1_prediction > .5] = 1
reg_m1_prediction[reg_m1_prediction <= .5] = 0
accuracy_m1 = sum(reg_m1_prediction == test["Survived"]) / len(reg_m1_prediction)

# regression with KF = 3  accuracy = 0.78
kf = KFold(trainData.shape[0], n_folds=3, random_state=1)
reg_m2 = LinearRegression()
predictions = []
for train, test in kf:
    reg_m2.fit(trainData[feature].iloc[train, :], trainData["Survived"].iloc[train])
    predictions.append(reg_m2.predict(trainData[feature].iloc[test, :]))

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy_m2 = sum(predictions == trainData["Survived"]) / len(predictions)

# regression with KF = 5  accuracy = 0.79
kf = KFold(trainData.shape[0], n_folds=5, random_state=1)
reg_m3 = LinearRegression()
predictions = []
for train, test in kf:
    reg_m3.fit(trainData[feature].iloc[train, :], trainData["Survived"].iloc[train])
    predictions.append(reg_m3.predict(trainData[feature].iloc[test, :]))

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy_m3 = sum(predictions == trainData["Survived"]) / len(predictions)

# logistic accuracy = 0.80
train, test = train_test_split(trainData, test_size=0.2)
reg_m4 = LogisticRegression()
reg_m4.fit(train[feature], train["Survived"])
reg_m4_prediction = reg_m4.predict(test[feature])
reg_m4_prediction[reg_m4_prediction > .5] = 1
reg_m4_prediction[reg_m4_prediction <= .5] = 0
accuracy_m4 = sum(reg_m4_prediction == test["Survived"]) / len(reg_m4_prediction)

# logistic with KF = 3  accuracy = 0.79
kf = KFold(trainData.shape[0], n_folds=3, random_state=1)
reg_m5 = LogisticRegression()
predictions = []
for train, test in kf:
    reg_m5.fit(trainData[feature].iloc[train, :], trainData["Survived"].iloc[train])
    predictions.append(reg_m5.predict(trainData[feature].iloc[test, :]))

predictions = np.concatenate(predictions, axis=0)
accuracy_m5 = sum(predictions == trainData["Survived"]) / len(predictions)

alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, trainData[feature], trainData["Survived"], cv=3)
print(scores.mean())

# random forest
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
scores = cross_validation.cross_val_score(alg, trainData[feature], trainData["Survived"], cv=3)
print(scores.mean())
# accuracy = 0.80

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, trainData[feature], trainData["Survived"], cv=3)
print(scores.mean())
# accuracy = 0.82

# Create new feature
trainData["FamilySize"] = trainData["SibSp"] + trainData["Parch"]
trainData["NameLength"] = trainData["Name"].apply(lambda x : len(x))

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

titles = trainData["Name"].apply(get_title)
print(pandas.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v

print(pandas.value_counts(titles))

trainData["Title"] = titles

family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Get the family ids with the apply method
family_ids = trainData.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[trainData["FamilySize"] < 3] = -1

# Print the count of each unique id.
print(pandas.value_counts(family_ids))

trainData["FamilyId"] = family_ids

feature = ["Pclass", "Sex_New", "Age_New", "SibSp", "Parch", "Fare", "Embarked_New" , "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(trainData[feature], trainData["Survived"])

scores = -np.log10(selector.pvalues_)
plt.bar(range(len(feature)), scores)
plt.xticks(range(len(feature)), feature, rotation='vertical')
plt.show()

feature = ["Pclass", "Sex_New", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
scores = cross_validation.cross_val_score(alg, trainData[feature], trainData["Survived"], cv=3)
print(scores.mean())
# accuracy = 0.811

# ensemble
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex_New", "Age_New", "Fare", "Embarked_New", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex_New", "Fare", "FamilySize", "Title", "Age_New", "Embarked_New"]]
]

kf = KFold(trainData.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = trainData["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(trainData[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(trainData[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions == trainData["Survived"]]) / len(predictions)
print(accuracy)
# accuracy = 0.82

