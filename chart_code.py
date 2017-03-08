import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from IPython.display import display

trainData = pd.read_csv("~/Documents/Projects/Titanic/Data/train.csv")
testData = pd.read_csv("~/Documents/Projects/Titanic/Data/test.csv")

fullData = trainData.append(testData)
fullData = fullData.reset_index()
display(trainData.head())

# Name
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

#surname
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

fullData.info()

fullData.describe()

fullData.plot(kind="scatter", x='Age', y='Fare');

sns.jointplot(x='Age', y='Fare', data=fullData, size=5)

sns.FacetGrid(fullData, hue="Survived", size=5) \
   .map(plt.scatter, "Age", "Fare") \
   .add_legend()

sns.boxplot(x="Survived", y= "Age", data=fullData)
sns.boxplot(x="Survived", y= "Fare", data=fullData)
sns.stripplot(x="Survived", y="Fare", data=fullData, jitter=True, edgecolor="gray")
sns.violinplot(x="Survived", y="Fare", data=fullData, size=6)

sns.FacetGrid(fullData, hue="Survived", size=6) \
   .map(sns.kdeplot, "Age") \
   .add_legend()
sns.FacetGrid(fullData, hue="Survived", size=6) \
   .map(sns.kdeplot, "Fare") \
   .add_legend()
sns.pairplot(fullData.drop("index", axis=1), hue="Survived", size=3)
sns.pairplot(fullData.drop("index", axis=1), hue="Survived", size=3, diag_kind="kde")

from pandas.tools.plotting import radviz
radviz(fullData.drop("index", axis=1), "Survived")
radviz(fullData[["Survived", "Fare", "Age"]], "Survived")

sns.FacetGrid(fullData, hue="Survived", size=5) \
   .map(plt.hist, "Sex") \
   .add_legend()

pd.crosstab(fullData["Pclass"], fullData["Survived"]).plot(kind='bar')
pd.crosstab(fullData["Pclass"], fullData["Survived"], normalize='index').plot(kind='bar')
pd.crosstab(fullData["Pclass"], fullData["Survived"]).apply(lambda r: r/r.sum(), axis=1).plot(kind='bar')

fullData[["Survived", "Fare", "Age", "Pclass"]]
sns.pairplot(fullData[["Fare", "Pclass", "Survived"]], hue="Survived", size=3);
