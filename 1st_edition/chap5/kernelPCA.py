#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:46:00 2017

@author: yohei
"""
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import exp
from numpy.linalg import eigh  # 実対称行列の固有値を求めるのでeigではなくてこちらを使う．
import numpy as np


class KPCA(object):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
      Tuning parameter of the RBF kernel

    n_components: int
      Number of principal components to return

    kernel: str
      name of Kernel('rbf')

    Attributes
    -----------
    eigvals_ : 1d-array
        グラム行列の固有値を昇順に並べたもの
    eigvecs_ : 2d-array
        グラム行列の固有ベクトルを縦に連結したもの
        eigvals_[i] は eigvecs_[:, i] に対応している
    """

    def __init__(self, gamma, n_components, kernel='rbf'):
        self.gamma = gamma
        self.n_components = n_components
        self.kernel = kernel

    def fit(self, X):
        """Fit training data.
        training dataからグラム行列Kを作り固有値(self.eigvals_)，
        固有ベクトル(self.eigvecs_)を求める．

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object

        """
        self.X_ = X
        sq_dists = pdist(
            self.X_, metric='sqeuclidean')  # pdist(pairwise distancesの略)

        # Convert pairwise distances into a square matrix. 上三角行列を正方行列に変換する
        mat_sq_dists = squareform(sq_dists)

        # Compute the symmetric kernel matrix.
        if self.kernel == 'rbf':
            K = exp(-self.gamma * mat_sq_dists)
        else:
            raise NameError('今はカーネルはRBFしかありません')
        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # Obtaining eigenpairs from the centered kernel matrix
        # numpy.linalg.eigh returns them in sorted order 昇順
        self.eigvals_, self.eigvecs_ = eigh(K)

        return self

    def transform(self, X):
        """データXを変換する．
        データXから主成分を取り出す．
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Data vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_kpca: {array-like}, shape = [n_samples, n_components]
            変換されたデータ行列．
        """

        X_kpca = np.empty((X.shape[0], 2))
        for j in range(self.n_components):
            for i, xi in enumerate(X):
                if self.kernel == 'rbf':
                    # ユークリッド距離の２乗を計算する
                    sqeuclidean = cdist(
                        self.X_, xi[np.newaxis, :], metric='sqeuclidean')
                    # RBFカーネルを作る
                    kernel = exp(-self.gamma * sqeuclidean)
                    # 変換を行う
                else:
                    raise NameError("今はカーネルはRBFしかありません")
                X_kpca[i, j] = 1 / np.sqrt(self.eigvals_[-j - 1]) * \
                    np.dot(self.eigvecs_[:, -j - 1].T, kernel)

        return X_kpca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
