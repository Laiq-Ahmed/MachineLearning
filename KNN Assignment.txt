import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# Question 0
def answer_zero():
    return len(cancer.feature_names)

# Question 1
def answer_one():
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df

# Question 2
def answer_two():
    cancerdf = answer_one()
    return cancerdf['target'].value_counts()

# Question 3
def answer_three():
    cancerdf = answer_one()
    X = cancerdf[cancer.feature_names]
    y = cancerdf['target']
    return X, y

# Question 4
from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test

# Question 5
from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, y_train = answer_four()[:2]
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    return knn

# Question 6
def answer_six():
    cancerdf = answer_one()
    knn = answer_five()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    return knn.predict(means)

# Question 7
def answer_seven():
    X_test = answer_four()[1]
    knn = answer_five()
    return knn.predict(X_test)

# Question 8
def answer_eight():
    X_test, y_test = answer_four()[1:]
    knn = answer_five()
    return knn.score(X_test, y_test)
