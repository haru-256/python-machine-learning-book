#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Stu Sep 23 16:46:00 2017

@author: yohei
"""
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import exp
from numpy.linalg import eigh  # 実対称行列の固有値を求めるのでeigではなくてこちらを使う．
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
      Tuning parameter of the RBF kernel

    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset

    """
    # Calculate pairwise squared Euclidean(ユークリッド距離の２乗) distances
    # in the MxN dimensional datasetself.
    """
    scipy.spatial.distances.pdistはn次元空間でのデータ点ごとの距離を返す関数
    行方向にデータが積まれているとする．
    戻り値は距離行列の上三角行列のみである．これは距離行列が対象行列であり対角成分が
    ０のためである．距離行列にするにはscipy.spatial.distances.squareform()を使えば良い
    引数のmetricにはどのような距離を求めるかの指定もできる．
    """
    sq_dists = pdist(X, metric='sqeuclidean')  # pdist(pairwise distancesの略)

    # Convert pairwise distances into a square matrix. 上三角行列を正方行列に変換する
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.linalg.eigh returns them in sorted order 昇順
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_kpca = np.empty((X.shape[0], 2))
    for j in range(n_components):
        for i, xi in enumerate(X):
            """
            scipy.spatial.distance.cdistは２つのベクトル間の距離を計算する．
            しかし，引数には2D-arrayが２つとれ，その２つの距離計算を行ってくれる．
            一方が行列で，もう一方がベクトルの場合，列の長ささえ合っていれば良い．
            詳しくはドキュメント参照
            https://docs.scipy.org/doc/scipy-0.19.1/reference/generated
                 /scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
            """
            sqeuclidean = cdist(X, xi[np.newaxis, :], metric='sqeuclidean')
            kernel = exp(-gamma * sqeuclidean)
            X_kpca[i, j] = 1 / np.sqrt(eigvals[-j - 1]) * \
                np.dot(eigvecs[:, -j - 1].T, kernel)

    return X_kpca
