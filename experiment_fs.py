import pandas as pd
import numpy as np
from scipy.stats import entropy
import fcbf.feature_selection as fs

if __name__ == '__main__':
    data = pd.read_csv('../dataset/titanic/train.csv')
    Y = data['Survived']
    X1 = data['SibSp']
    X2 = data['Parch']
    X3 = data['Sex']
    X4 = data['Pclass']
    X5 = data['Embarked']

    # print(fs.ig_su(X1, Y))
    # print(fs.ig_su(X2, Y))
    # print(fs.ig_su(X3, Y))
    # print(fs.ig_su(X4, Y))
    # print(fs.ig_su(X5, Y))

    ignore_features = ['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin']

    data.drop(labels=ignore_features, axis=1, inplace=True)

    # fs.fcbf(data, 'Survived', threshold=0, base=2, is_debug=False)
    # fs.fcbf(data, 'Survived', threshold=0.2, base=2, is_debug=False)

    X = data.drop('Survived', axis=1)
    y = data['Survived']
    fs.fcbf(X, y, threshold=0, base=2, is_debug=True)