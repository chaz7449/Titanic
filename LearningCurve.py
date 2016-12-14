import numpy as np
import pandas as pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression

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

reg = LogisticRegression()
train_sizes, train_scores, test_scores = learning_curve(reg, titanicData[feature], titanicData[target], n_jobs=2,
                                                        cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0),
                                                        train_sizes=np.linspace(.1, 1.0, 100))

train_scores_mean = 1 - np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = 1 - np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
train_scores_upper = (train_scores_mean + train_scores_std).tolist()
train_scores_lower = (train_scores_mean - train_scores_std).tolist()
train_scores_lower = train_scores_lower[::-1]
train_sizes_rev = train_sizes[::-1]
test_scores_upper = (test_scores_mean + test_scores_std).tolist()
test_scores_lower = (test_scores_mean - test_scores_std).tolist()
test_scores_lower = test_scores_lower[::-1]

# plot
trace1 = go.Scatter(
    x=train_sizes+train_sizes_rev,
    y=train_scores_upper+train_scores_lower,
    fill='tozeroy',
    fillcolor='rgba(0,100,80,0.2)',
    line=go.Line(color='transparent'),
    showlegend=False,
)
trace2 = go.Scatter(
    x=train_sizes,
    y=train_scores_mean,
    line=go.Line(color='rgb(0,100,80)'),
    mode='lines',
    name='Training Score',
)
trace3 = go.Scatter(
    x=train_sizes + train_sizes_rev,
    y=test_scores_upper + test_scores_lower,
    fill='tozeroy',
    fillcolor='rgba(255,20,10,0.2)',
    line=go.Line(color='transparent'),
    showlegend=False,
)
trace4 = go.Scatter(
    x=train_sizes,
    y=test_scores_mean,
    line=go.Line(color='rgb(255,20,10)'),
    mode='lines',
    name='Cross-validation score',
)

data = go.Data([trace1, trace2, trace3, trace4])

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
py.plot(fig, filename='learning_curve')

