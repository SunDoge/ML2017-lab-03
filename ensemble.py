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
        self.le = LabelEncoder()

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
        # Label encode y
        y = self.le.fit_transform(y)
        epsilon = 0.1
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
            # print(iboost)
            # print(y_pred)

            error = self.w.dot(y_pred != y)
            # print(error)
            # error = 0.1 * self.w.min() if error == 0 else error

            alpha = 0.5 * np.log((1 - error + epsilon) / (error + epsilon))
            # print(alpha)

            self.w = self.w * np.exp(-alpha * y * y_pred)
            self.w = self.w / self.w.sum()

            self.models.append(weak_classifier)
            self.alphas.append(alpha)

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
        y_pred = np.zeros(X.shape[0])
        for alpha, clf in zip(self.alphas, self.models):
            y_pred += alpha * clf.predict(X)
            # print(alpha)

        y_pred = np.sign(y_pred)
        # print(y_pred)
        return self.le.inverse_transform(y_pred.astype(int))

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


if method == 'gd':            
    grad = gradient(X_batch, y_batch, w)
    w -= learning_rate * grad
    
elif method == 'NAG':
    grad = gradient(X_batch, y_batch, w - gamma * v)
    v = gamma * v + learning_rate * grad
    w -= v
    
elif method == 'RMSProp':
    grad = gradient(X_batch, y_batch, w)
    G = gamma * G + (1 - gamma) * np.square(grad)
    w -= learning_rate * 100 * grad / np.sqrt(G + epsilon)
    
elif method == 'AdaDelta':
    grad = gradient(X_batch, y_batch, w)
    G = gamma * G + (1 - gamma)* np.square(grad)
    dw = -np.sqrt(delta + epsilon) / np.sqrt(G + epsilon) * grad
    w += dw
    delta = gamma * delta + (1 - gamma) * np.square(delta)
    
elif method == 'Adam':
    t = i + 1
    grad = gradient(X_batch, y_batch, w)
    m = beta * m + (1 - beta) * grad
    G = gamma * G + (1 - gamma) * np.square(grad)
    alpha = learning_rate * 100 * np.sqrt(1 - gamma ** t) / (1 - beta ** t)
    w -= alpha * m / np.sqrt(G + epsilon)