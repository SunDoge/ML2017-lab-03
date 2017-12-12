import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier


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

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_features = X.shape[0]
        # clear weights
        self.w = np.ones(n_features) / n_features
        self.models = []
        self.alphas = []
        # print(self.w)
        # max_depth = None
        for iboost in range(self.n_weakers_limit):
            weak_classifier = self.weak_classifier()
            weak_classifier.fit(X, y, sample_weight=self.w)
            y_pred = weak_classifier.predict(X)

            error = self.w.dot(y_pred)
            alpha = np.log((1 - error) / error)
            self.w = w * np.exp(-alpha * y * y_pred)
            self.w = self.w / self.w.sum()

            self.models[iboost] = weak_classifier
            self.alphas[iboost] = alpha

        return self

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
