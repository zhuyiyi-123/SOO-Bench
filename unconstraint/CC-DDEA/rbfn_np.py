# -*- coding: utf-8 -*-
import functools
import math
from typing import Callable, Union
import numpy as np
from sklearn.cluster import KMeans
import rbf


class RBFN:
    '''radial basis function neural network
    Metric:
        2-Norm

    How it train:
        Use k-means to get the centers points, and use the pseudoinverse to do interpolation
    '''

    def __init__(self,
                 hidden_features: int,
                 norm_func: Callable[[np.ndarray], np.ndarray] = functools.partial(
                     np.linalg.norm, ord=2, axis=-1),
                 basis_func: Union[Callable[[np.ndarray],
                                      np.ndarray], str] = 'gaussian',
                 ):
        '''
        Args:
            hidden_features: a postive integer, stands for the number of center point of the hidden layer
        '''
        self.hidden_features = hidden_features
        if callable(norm_func):
            self.norm_func = norm_func
        else:
            raise ValueError('RBF layer: norm_func must be a callable object')
        if callable(basis_func):
            self.basis_func = basis_func
        elif isinstance(basis_func, str):
            if basis_func in rbf.funcs:
                self.basis_func = rbf.funcs[basis_func]
            else:
                raise ValueError(
                    f'RBF layer: No builtin basis function named {basis_func}')
        else:
            raise ValueError('RBF layer: error type of basis_func')

        # paramters of RBF layer
        # sigmas is a vector, sigmas[i] is the sigma of centers, it is the spread radius
        self.sigmas = None
        self.centers = None
        # paramters of linear layer
        self.weight = None
        self.bias = None

    def __calc_sigmas(self) -> np.ndarray:
        '''
        compute the hyperparameter `sigma` of kernel function
        '''
        c_ = np.expand_dims(self.centers, 1)
        ds = self.norm_func(c_ - self.centers)
        sigma = 2 * np.mean(ds, axis=1)
        sigma = np.sqrt(0.5) / sigma
        return sigma

    def rbf(self, X: np.ndarray) -> np.ndarray:
        '''
        This is the first layer of the radial basis function neural network
        '''
        x = np.expand_dims(X, 1)
        r = self.norm_func(x - self.centers)
        eps_r = self.sigmas * r
        return self.basis_func(eps_r)

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Principle:
            1. kmeans: compute the first layer
            2. least squares method by pseudoinverse: compute the second layer
        '''
        n = len(X)
        # TODO https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
        self.centers = KMeans(n_clusters=self.hidden_features, n_init=10).fit(
            X).cluster_centers_
        self.sigmas = self.__calc_sigmas()
        # RBF layer
        X = self.rbf(X)
        # linear layer
        X = np.c_[X, np.ones(n)]
        w = np.linalg.pinv(X) @ y.reshape([-1, 1])
        self.weight = w[:self.hidden_features, :]
        self.bias = w[self.hidden_features, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Return:
            a column vector(2-d)
        '''
        if X.ndim == 1:
            X = X.reshape((1, -1))
        X = self.rbf(X)
        return X @ self.weight + self.bias
