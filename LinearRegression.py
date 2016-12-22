import pandas as pandas
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

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

alg = LinearRegression(normalize=True)
alg.fit(titanicData[feature], titanicData[target])
#print(alg._residues)
#titanicData_prediction = alg.predict(titanicData[feature])
#titanicData_prediction[titanicData_prediction > .5] = 1
#titanicData_prediction[titanicData_prediction <= .5] = 0
#accuracy_m1 = sum(titanicData_prediction == titanicData["Survived"]) / len(titanicData_prediction)

# region learning curve for threshold
thresholds = np.linspace(0, 1.0, 100)
errors = []
for threshold in thresholds:
    titanicData_prediction = alg.predict(titanicData[feature])
    titanicData_prediction[titanicData_prediction > threshold] = 1
    titanicData_prediction[titanicData_prediction <= threshold] = 0
    accuracy = sum(titanicData_prediction == titanicData["Survived"]) / len(titanicData_prediction)
    errors.append(1 - accuracy)

trace2 = go.Scatter(
    x=thresholds,
    y=errors,
    line=go.Line(color='rgb(0,100,80)'),
    mode='lines',
    name='Training Score',
)
data = go.Data([trace2])
layout = go.Layout(
    xaxis=go.XAxis(
        showgrid=False,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False,
        title='Training examples',
        rangemode='tozero'
    ),
    yaxis=go.YAxis(
        showgrid=False,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False,
        title='Error'
    ),
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='regress_threshold.html')

trainData, testData = train_test_split(titanicData, test_size=0.2)
train_alg = LinearRegression(normalize=True)
train_alg.fit(trainData[feature], trainData[target])

train_errors = []
for threshold in thresholds:
    train_prediction = train_alg.predict(trainData[feature])
    train_prediction[train_prediction > threshold] = 1
    train_prediction[train_prediction <= threshold] = 0
    accuracy = sum(train_prediction == trainData["Survived"]) / len(train_prediction)
    train_errors.append(1 - accuracy)

test_alg = LinearRegression(normalize=True)
test_alg.fit(testData[feature], testData[target])

test_errors = []
for threshold in thresholds:
    test_prediction = test_alg.predict(testData[feature])
    test_prediction[test_prediction > threshold] = 1
    test_prediction[test_prediction <= threshold] = 0
    accuracy = sum(test_prediction == testData["Survived"]) / len(test_prediction)
    test_errors.append(1 - accuracy)

trace1 = go.Scatter(
    x=thresholds,
    y=train_errors,
    line=go.Line(color='rgb(0,100,80)'),
    mode='lines',
    name='Training error',
)
trace2 = go.Scatter(
    x=thresholds,
    y=test_errors,
    line=go.Line(color='rgb(255,20,10)'),
    mode='lines',
    name='Validation error',
)
data = go.Data([trace1, trace2])
layout = go.Layout(
    xaxis=go.XAxis(
        showgrid=False,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False,
        title='Training examples',
        rangemode='tozero'
    ),
    yaxis=go.YAxis(
        showgrid=False,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False,
        title='Error'
    ),
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='regress_trainTest_threshold.html')
# endregion

# region one hot encoding
titanicData[['Embarked_S', 'Embarked_C', 'Embarked_Q']] = pandas.get_dummies(titanicData['Embarked_New'])
feature = ["Pclass", "Sex_New", "Age_New", "SibSp", "Parch", "Fare", "Embarked_S", "Embarked_C", "Embarked_Q"]
target = "Survived"
trainData, testData = train_test_split(titanicData, test_size=0.2)
alg = LinearRegression()
alg.fit(trainData[feature], trainData[target])
testData_prediction = alg.predict(testData[feature])
testData_prediction[testData_prediction > 0.5] = 1
testData_prediction[testData_prediction <= 0.5] = 0
accuracy = sum(testData_prediction == testData["Survived"]) / len(testData_prediction)
# endregion

# region Ridge
rig = Ridge(alpha=0, normalize=True)
rig.fit(titanicData[feature], titanicData[target])
titanicData_prediction = rig.predict(titanicData[feature])
titanicData_prediction[titanicData_prediction > 0.5] = 1
titanicData_prediction[titanicData_prediction <= 0.5] = 0
accuracy = sum(titanicData_prediction == titanicData[target]) / len(titanicData_prediction)
print(accuracy)

n_alphas = 200
alphas = np.linspace(-10, 10, n_alphas)
clf = Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(titanicData[feature], titanicData[target])
    coefs.append(clf.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
# endregion

# region Lasso
n_alphas = 200
alphas = np.linspace(-10, 10, n_alphas)
clf = Lasso(fit_intercept=False, max_iter=10000)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(titanicData[feature], titanicData[target])
    coefs.append(clf.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
# endregion

# region feature
titanicData["Age_New"] = titanicData["Age"].groupby([titanicData["Sex"], titanicData["Pclass"]])\
    .apply(lambda x: x.fillna(x.median()))
titanicData.loc[titanicData["Sex"] == "male", "Sex_New"] = 0
titanicData.loc[titanicData["Sex"] == "female", "Sex_New"] = 1
titanicData["Embarked_New"] = titanicData["Embarked"].fillna('S')
titanicData.loc[titanicData["Embarked_New"] == 'S', "Embarked_New"] = 0
titanicData.loc[titanicData["Embarked_New"] == 'C', "Embarked_New"] = 1
titanicData.loc[titanicData["Embarked_New"] == 'Q', "Embarked_New"] = 2
titanicData["FamilySize"] = titanicData["SibSp"] + titanicData["Parch"]
titanicData["FamilySize_2"] = titanicData["FamilySize"].apply(lambda x: pow(x, 2))

feature = ["Pclass", "Sex_New", "Age_New", "FamilySize", "Fare", "Embarked_New"]
target = "Survived"

alg = LinearRegression()
alg.fit(titanicData[feature], titanicData[target])
titanicData_prediction = alg.predict(titanicData[feature])
titanicData_prediction[titanicData_prediction > .6] = 1
titanicData_prediction[titanicData_prediction <= .6] = 0
accuracy = sum(titanicData_prediction == titanicData["Survived"]) / len(titanicData_prediction)
print(accuracy)

poly = PolynomialFeatures()
poly.fit(titanicData)

# endregion
