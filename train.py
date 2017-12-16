import numpy as np
from ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    clf = AdaBoostClassifier(n_weakers_limit=1000)
    X, y = load_breast_cancer(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # print(X_train)
    # print(y_train)
    # X_train = np.array([
    #     [1, 1],
    #     [1, 0],
    #     [0, 1],
    #     [0, 0]
    # ])

    # y_train = np.array([
    #     0,
    #     1,
    #     1,
    #     0
    # ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    skclf = SkAdaBoostClassifier()
    skclf.fit(X_train, y_train)
    print(accuracy_score(y_test, skclf.predict(X_test)))


if __name__ == '__main__':
    main()
