#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:31:00 2017

@author: yohei
"""

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six  # Python 2.7 との互換性を持たせるため
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    # このクラスに基本的な機能をつけるためにBaseEstimatorクラスとClassifierMixinクラスを
    # 継承する．この２つのクラスによって，set_paramsメソッドやget_params, scoreメソッドが
    # 使えるようになる．
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='classlabel')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], (optional ,default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """

    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        # named_classifiers([..]) リストの中身は識別器で,
        # この関数はそれを識別器の名前(sklearnが独自に決めた識別器のクラスの小文字)と，
        # 識別器オブジェクトに分けられ，タプルのリストとなっている.
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            # clone() 関数は同じパラメータのどのデータにもfitしていない識別器を返す
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.

        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            # predictions : [n_samples, len(classifiers_)]
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            # arr=predictionsのaxis=1(列方向)に沿って func1d関数を適用する.
            # 各サンプルのクラス確率に重みをつけて足し合わせた値が最大となる列番号を返す．
            maj_vote = np.apply_along_axis(
                func1d=lambda x:
                np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions)
        # デコードを行う
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """

        # np.asarray() 入力をarrayに変換する
        # probas は３d-arrayである.  shape: (n_classifiers_, n_samples, n_classes)
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        # 識別器ごとに重み付き平均を取る
        # avg_probaは二次元配列となる. shape: (n_samples, n_classes)
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            # super((スーパークラスを調べたいサブクラスのクラス名), (インスタンス(self)) )
            # これによりスーパークラスを呼び出せる.
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            # copy() は同じ要素を持つオブジェクトを作りだす．
            # しかし，copy()で作り出したものを変更してもオリジナルの方は変更しない
            # = によるコピーだと変更してしまう．スライスによるコピーも同じ
            # だが．　リストに対しての[:]のみcopy()と同じ働きをする.
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                # step.get_params(deep=True)により各識別器のパラメータをキーとした
                # ディクショナリとして返される.キーの値にはパラメータの値を持つ.
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['{0}__{1}'.format(name, key)] = value
            return out
