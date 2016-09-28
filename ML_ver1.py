import pandas as pandas
from sklearn.linear_model import LinearRegression

titanic_train = pandas.read_csv("~/Documents/PycharmProjects/Titanic/train.csv")
print(titanic_train.describe(include="all"))

titanic_train["Age_transform"] = titanic_train["Age"].fillna(titanic_train["Age"].median())
titanic_train.loc[titanic_train["Sex"] == "male", "Sex_transform"] = 0
titanic_train.loc[titanic_train["Sex"] == "female", "Sex_transform"] = 1
titanic_train["Embarked_transform"] = titanic_train["Embarked"].fillna('S')
titanic_train.loc[titanic_train["Embarked_transform"] == 'S', "Embarked"] = 0
titanic_train.loc[titanic_train["Embarked_transform"] == 'C', "Embarked"] = 1
titanic_train.loc[titanic_train["Embarked_transform"] == 'Q', "Embarked"] = 2

choose_feature = ["Pclass", "Sex_transform", "Age_transform", "SibSp", "Parch", "Fare", "Embarked_transform"]

regr = LinearRegression()


