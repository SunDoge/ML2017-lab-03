import numpy as np
from ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier


def main():
    clf = AdaBoostClassifier(n_weakers_limit=10)
    X_train = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ])

    y_train = np.array([
        0,
        1,
        1,
        0
    ])

    clf.fit(X_train, y_train)
    print(clf.predict(X_train))


if __name__ == '__main__':
    main()
