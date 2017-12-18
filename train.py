import numpy as np
from ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import io, transform, util, color
import matplotlib.pyplot as plt
from feature import NPDFeature
from joblib import Parallel, delayed


def test_xor():
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

    clf = AdaBoostClassifier(n_weakers_limit=1000)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(classification_report(y_train, y_pred))

    skclf = SkAdaBoostClassifier()
    skclf.fit(X_train, y_train)
    print(classification_report(y_train, skclf.predict(X_train)))


def test_breast_cancer():
    clf = AdaBoostClassifier(n_weakers_limit=50)
    X, y = load_breast_cancer(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    skclf = SkAdaBoostClassifier()
    skclf.fit(X_train, y_train)
    print(classification_report(y_test, skclf.predict(X_test)))


def resize_image(img):
    img = transform.resize(img, (24, 24))
    img = color.rgb2gray(img)
    img = util.img_as_ubyte(img)
    return img


def get_features(img):
    img = resize_image(img)
    features = NPDFeature(img).extract()
    return features


def test_image():
    path = 'datasets/original/'
    face = io.imread_collection(path + 'face/*.jpg')
    nonface = io.imread_collection(path + 'nonface/*.jpg')
    labels = ['face', 'nonface']

    X = []
    y = []

    # face_list = [get_features(i) for i in face]
    # nonface_list = [get_features(i) for i in nonface]
    # face_list = Parallel(n_jobs=4)(delayed(get_features)(i) for i in face)
    # nonface_list = Parallel(n_jobs=4)(
    #     delayed(get_features)(i) for i in nonface)

    # X += face_list
    # y += list(np.zeros(len(face_list), dtype=int))
    # X += nonface_list
    # y += list(np.ones(len(nonface_list), dtype=int))

    # AdaBoostClassifier.save(X, 'X.pkl')
    # AdaBoostClassifier.save(y, 'y.pkl')
    X = AdaBoostClassifier.load('X.pkl')
    y = AdaBoostClassifier.load('y.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(y), test_size=0.33, random_state=42)

    print('start training')
    clf = AdaBoostClassifier(n_weakers_limit=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))

    # skclf = SkAdaBoostClassifier(n_estimators=10)
    # skclf.fit(X_train, y_train)
    # print(classification_report(y_test, skclf.predict(X_test), target_names=labels))
    # print(np.array(X))
    # print(np.array(y))


def main():
    # test_xor()
    # test_breast_cancer()
    test_image()


if __name__ == '__main__':
    main()
