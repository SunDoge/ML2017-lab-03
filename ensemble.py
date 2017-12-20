import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier=DecisionTreeClassifier, n_weakers_limit=1000):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.learning_rate = 0.5
        self.le = LabelEncoder()

    def is_good_enough(self):
        '''Optional'''
        pass

    def _boost(self, X, y):
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples = X.shape[0]
        # Label encode y
        # y = self.le.fit_transform(y)
        self.le.fit(y)

        self.n_classes = len(self.le.classes_)
        epsilon = 0.01
        # clear weights
        self.w = np.ones(n_samples) / n_samples
        self.models = []
        self.alphas = []
        # print(self.w)
        # max_depth = None
        for iboost in range(self.n_weakers_limit):

            clf = self.weak_classifier(max_depth=1)
            clf.fit(X, y, sample_weight=self.w)
            y_pred = clf.predict(X)

            incorrect = y_pred != y

            error = np.mean(
                np.average(incorrect, weights=self.w, axis=0))

            if error <= 0:
                print(error)
                alpha = 1
            else:

                alpha = self.learning_rate * (
                    np.log((1. - error) / (error)) +
                    np.log(self.n_classes - 1.))

                self.w *= np.exp(alpha * incorrect *
                                 ((self.w > 0) |
                                  (alpha < 0)))

                self.models.append(clf)
                self.alphas.append(alpha)

            if error == 0:
                break

            self.w /= self.w.sum()
            print("round %d/%d, error=%f" %
                  (iboost + 1, self.n_weakers_limit, error))

        return self

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        n_classes = self.n_classes
        classes = self.le.classes_[:, np.newaxis]

        pred = sum((clf.predict(X) == classes).T * alpha
                   for clf, alpha in zip(self.models,
                                         self.alphas))
        # for clf, alpha in zip(self.models, self.alphas):
        #     print(clf.predict(X) == classes).T * alpha
        print(self.alphas)

        pred /= sum(self.alphas)

        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            # return self.le.classes_.take(pred > 0, axis=0)
            return pred

        # return self.le.classes_.take(np.argmax(pred, axis=1), axis=0)
        return np.argmax(pred, axis=1)

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''

        n_classes = self.n_classes
        classes = self.le.classes_[:, np.newaxis]

        pred = sum((clf.predict(X) == classes).T * alpha
                   for clf, alpha in zip(self.models,
                                         self.alphas))
        # for clf, alpha in zip(self.models, self.alphas):
        #     print(clf.predict(X) == classes).T * alpha
        print(self.alphas)

        pred /= sum(self.alphas)

        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.le.classes_.take(pred > 0, axis=0)

        return self.le.classes_.take(np.argmax(pred, axis=1), axis=0)

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
