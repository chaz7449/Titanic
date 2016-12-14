import pandas as pd
import numpy as np
import plotly.offline as py
import cufflinks as cf
from sklearn.ensemble import RandomForestClassifier

allData = pd.read_csv("~/Documents/PycharmProjects/Titanic/train.csv")
allData.head()
allData.dtypes
allData.info()
allData.describe()

#cf.go_offline()
#cf.set_config_file(offline=True)
#survived = allData['Pclass'].value_counts().iplot(kind='bar', title='Death and Survival Counts', asFigure=True)
#py.plot(survived)

py.plot(allData[['Pclass', 'Survived']].apply(pd.Series.value_counts).iplot(kind='bar', asFigure=True))
py.plot(allData[['Sex', 'Embarked']].apply(pd.Series.value_counts).iplot(kind='bar', asFigure=True))
py.plot(allData['Age'].iplot(kind='histogram', title='Age', asFigure=True))

pclass_xt = pd.crosstab(allData['Pclass'], allData['Survived'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
py.plot(pclass_xt_pct.iplot(kind='bar', barmode='stack',
                            title='Survival Rate by Passenger Classes',
                            yTitle='Survival Rate', xTitle='Passenger Class',
                            asFigure=True))

sexes = sorted(allData['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
allData['Sex_Val'] = allData['Sex'].map(genders_mapping).astype(int)
sex_val_xt = pd.crosstab(allData['Sex_Val'], allData['Survived'])
sex_val_xt_pct = sex_val_xt.div(sex_val_xt.sum(1).astype(float), axis=0)
py.plot(sex_val_xt_pct.iplot(kind='bar', barmode='stack',
                            title='Survival Rate by Gender',
                            yTitle='Survival Rate', xTitle='Sex',
                            asFigure=True))

females_df = allData[allData['Sex'] == 'female']
females_xt = pd.crosstab(females_df['Pclass'], allData['Survived'])
females_xt_pct = females_xt.div(females_xt.sum(1).astype(float), axis=0)
py.plot(females_xt_pct.iplot(kind='bar', barmode='stack',
                            title='Female Survival Rate by Passenger Class',
                            yTitle='Survival Rate', xTitle='Passenger Class',
                            asFigure=True))

males_df = allData[allData['Sex'] == 'male']
males_xt = pd.crosstab(males_df['Pclass'], allData['Survived'])
males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis=0)
py.plot(males_xt_pct.iplot(kind='bar', barmode='stack',
                            title='Male Survival Rate by Passenger Class',
                            yTitle='Survival Rate', xTitle='Passenger Class',
                            asFigure=True))

allData["Embarked"] = allData["Embarked"].fillna('S')
embarked_locs = sorted(allData['Embarked'].unique())
embarked_locs_mapping = dict(zip(embarked_locs, range(0, len(embarked_locs) + 1)))
allData['Embarked_Val'] = allData['Embarked'].map(embarked_locs_mapping).astype(int)
embarked_val_xt = pd.crosstab(allData['Embarked_Val'], allData['Survived'])
embarked_val_xt_pct = embarked_val_xt.div(embarked_val_xt.sum(1).astype(float), axis=0)

py.plot(males_xt_pct.iplot(kind='bar', barmode='stack',
                            title='Male Survival Rate by Port of Embarkation',
                            yTitle='Survival Rate', xTitle='Port of Embarkation',
                            asFigure=True))
allData = pd.concat([allData, pd.get_dummies(allData['Embarked_Val'], prefix='Embarked_Val')], axis=1)

allData[allData['Age'].isnull()][['Sex', 'Pclass', 'Age']].head()
allData['AgeFill'] = allData['Age']
allData['AgeFill'] = allData['AgeFill'] \
                        .groupby([allData['Sex_Val'], allData['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))
df1 = allData[allData['Survived'] == 0]['Age']
df2 = allData[allData['Survived'] == 1]['Age']
max_age = max(allData['AgeFill'])
py.plot(pd.DataFrame({'Died': df1, 'Survived': df2}) \
        .iplot(kind='histogram', barmode='stack', bins=max_age / 10,
               title='Survivors by Age Groups Histogram',
               yTitle='Count', xTitle='Age', asFigure=True))

py.plot(allData.AgeFill[allData.Pclass == 1].iplot(kind='histogram', histnorm='probability density', asFigure=True))

allData['FamilySize'] = allData['SibSp'] + allData['Parch']
py.plot(allData['FamilySize'] \
        .iplot(kind='histogram',
               title='Family Size Histogram', asFigure=True))

family_sizes = sorted(allData['FamilySize'].unique())
family_size_max = max(family_sizes)
df1 = allData[allData['Survived'] == 0]['FamilySize']
df2 = allData[allData['Survived'] == 1]['FamilySize']
py.plot(pd.DataFrame({'Died': df1, 'Survived': df2}) \
        .iplot(kind='histogram', barmode='stack', bins=family_size_max + 1,
               title='Survivors by Family Size',
               yTitle='Count', xTitle='FamilySize', asFigure=True))

feature = ['Pclass', 'Fare', 'Sex_Val', 'Embarked_Val_0', 'Embarked_Val_1',
           'Embarked_Val_2', 'AgeFill', 'FamilySize']

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(allData[feature], allData['Survived'])
score = clf.score(allData[feature], allData['Survived'])

