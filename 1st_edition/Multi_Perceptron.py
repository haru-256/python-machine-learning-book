#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:41:15 2017

@author: yohei
"""

import numpy as np
import pandas as pd


class Multi_Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.coef_ = np.zeros((X.shape[1], len(np.unique(y))))
        self.intercept_ = np.zeros(1)
        self.errors_ = []
        T = pd.get_dummies(y).values
        for _ in range(self.n_iter):
            errors = 0
            for xi, ti in zip(X, T):
                predict = np.zeros(len(np.unique(y)))
                predict[self.predict_1(xi)] = 1
                update = self.eta * (ti - predict)
                self.coef_ += update * xi
                self.intercept_ += update
                errors += int(update != 0.0)  # int(True)=1, int(False)=0
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.coef_) + self.intercept_  # calculate w.T x + b

    def predict_1(self, X):
        return np.argmax(self.net_input(X))

    def predict(self, X):
        """Return class label after unit step"""
        return np.argmax(self.net_input(X), axis=1)
        # np.where(条件式, Trueの時返す値, Falseの時返す値)  返り値の型:array
