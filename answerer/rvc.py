# -*- coding: utf-8 -*-
# @Author: dmytro
# @Date:   2017-01-05 21:19:46
# @Last Modified by:   Dmytro Kalpakchi
# @Last Modified time: 2017-01-15 22:27:08

import numpy as np
from numpy.linalg import inv, pinv
import logging
import cProfile
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel
from scipy.special import expit

logger = logging.getLogger("RVC")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def gaussian_mixture_data(N):
    cross = np.random.multivariate_normal([1, 2], np.array([[1, -0.5], [-0.5, 1]]), N // 2)
    circle1 = np.random.multivariate_normal([3, 3.5], np.array([[1, -0.5], [-0.5, 1]]), N // 4)
    circle2 = np.random.multivariate_normal([3.2, 1], np.array([[1, 0.2], [0.2, 1]]), N - N // 4 - N // 2)
    x = np.vstack((cross, circle1, circle2))
    t = np.hstack((np.zeros(N // 2), np.ones(N - N // 2)))
    return x, t

# not sure if it's linear just remember it's a kernel satisfying Mercer's condition
def linear_kernel(x, y):
    x, y = np.array(x), np.array(y)
    return np.dot(x, y.T) + 1


def estimate_gamma_cls(data, kernel, minr, maxr, step):
    k = 5
    gamma_test = np.arange(minr, maxr, step)
    maxGamma = 0
    max_err = float('inf')
    for g in gamma_test:
        logger.info("Started CV with gamma={}".format(g))
        # rvc = RVC(lambda x, y: kernel(x, y, gamma=g))
        rvc = SVC(kernel=kernel, gamma=g)
        kf = KFold(n_splits=k)
        err = 0
        for train, test in kf.split(data[0]):
            X_train, X_test, t_train, t_test = data[0][train], data[0][test], data[1][train], data[1][test]
            try:
                rvc.fit(X_train, t_train)
                y = rvc.predict(X_test)
                err = max(err, np.sum(y != t_test))
            except np.linalg.LinAlgError:
                err = float('inf')
                break
        if err < max_err:
            maxGamma = g
            max_err = err
    return maxGamma

def plot_svm_boundary(svc, x, y, t):
    sv_ind = svc.support_
    grid_x = np.arange(x.min() - 0.1, x.max() + 0.1, 0.025)
    grid_x1, grid_x2 = np.meshgrid(grid_x, grid_x)
    grid_points = np.column_stack((np.ravel(grid_x1), np.ravel(grid_x2)))
    y1 = svc.predict(grid_points)
    y_grid = np.reshape(y1, grid_x1.shape)
    plt.title("SVM: error={}% vectors={}".format(np.sum(y != t) * 100 / len(y), svc.support_.shape[0]))
    plt.contour(grid_x1, grid_x2, y_grid, levels=[0.5], colors="k", linestyles='dashed')
    plt.scatter(x[sv_ind,0], x[sv_ind,1], s=80, facecolors='none', edgecolors='g')

    # Plot also the training points
    c1, c2 = t == 0, t == 1
    plt.scatter(x[c1,0], x[c1,1], marker="x")
    plt.scatter(x[c2,0], x[c2,1], marker="o")
    plt.savefig("svm.pdf")
    plt.show()


# Inheritance from BaseEstimator is realized in order for OneVsRestClassifier to function properly
# Although {fit, predict}_one_vs_rest are also realized to check
class RVC(BaseEstimator):
    def __init__(self, kernel, min_grad=1e-12, tol=1e12, alpha=None):
        """
        Args:
            kernel (callable): A kernel function to be used for the set of basis functions
            tol (float, optional): Pruning tolerance
            alpha (numpy.array): Description
        """
        self.__alpha = alpha
        self.kernel = kernel
        self.tol = tol
        self.min_grad = min_grad

    @property
    def alpha(self):
        return self.__alpha

    @property
    def weights(self):
        return self.__mc_weights if len(np.unique(self.__t)) > 2 else self.__weights

    def design_matrix(self, x):
        return np.insert(self.kernel(x, self.__X), 0, 1, axis=1)

    def sigmoid(self, y):
        # return 1 / (1 + np.exp(-y))
        return expit(y)

    def __objective(self, w):
        y = self.sigmoid(np.dot(self.design_matrix(self.__X), w))
        A = np.diag(self.__alpha)
        return np.sum(self.__t * np.log(y)) + np.sum((1 - self.__t) * np.log(1 - y)) - 0.5 * np.dot(np.dot(w.T, A), w)

    def __fit_init(self, X, t):
        assert len(X) == len(t), "Not all data is labeled"
        self.__X, self.__t = X, t
        self.__N = len(X)
        self.__alpha = np.ones(self.__N + 1)
        self.__weights = np.zeros(self.__N + 1) # singular matrix with all values >= 1

    def fit(self, X, t):
        """
        Args:
            X (numpy.array): Matrix of N D-dimensional data points with dimensionality N x D
            t (numpy.array): Vector of class labels, each t is either 0 or 1
        Returns: None
        """
        self.__fit_init(X, t)
        for it in range(1000):
            # pruning
            ind = self.__alpha < self.tol
            Phi = self.design_matrix(self.__X)[:,ind]
            A = np.diag(self.__alpha[ind])

            # Finding the mode of a posterior using IRLS
            # IRLS for RVM is from Bishop book
            # NOTE: instead of maximizing a posterior, we're minimizing the negated posterior
            for iw in range(100):
                y = self.sigmoid(Phi.dot(self.__weights[ind]))
                B = np.eye(len(y)) * y * (1 - y)
                neg_grad = A.dot(self.__weights[ind]) - Phi.T.dot(self.__t - y)
                neg_H = Phi.T.dot(B).dot(Phi) + A # sometimes can be singular
                w_old = np.copy(self.__weights)
                self.__weights[ind] = self.__weights[ind] - inv(neg_H).dot(neg_grad)
                if np.linalg.norm(neg_grad) < self.min_grad: # gradient based check
                    break
            logger.debug("IRLS converged after {} iterations".format(iw + 1))

            y = self.sigmoid(np.dot(Phi, self.__weights[ind]))
            B = np.eye(len(y)) * y * (1 - y)
            Sigma = inv(Phi.T.dot(B).dot(Phi) + A)

            gamma = 1 - self.__alpha[ind] * np.diag(Sigma)
            old_alpha = np.copy(self.__alpha)
            self.__alpha[ind] = gamma / (self.__weights[ind] * self.__weights[ind])

            if np.linalg.norm(self.__alpha - old_alpha) < 1e-8:
                break
        logger.debug("RVM converged after {} iterations".format(it + 1))
        logger.info("Final amount of relevance vectors: {}".format(len(self.__weights[self.__alpha < self.tol])))

    def fit_one_vs_rest(self, X, t):
        self.__mc_weights, self.__mc_ind = [], []
        for k in np.unique(t):
            t_new = np.copy(t)
            pind, nind = t_new == k, t_new != k
            t_new[pind], t_new[nind] = 1, 0
            self.fit(X, t_new)
            self.__mc_weights.append(np.copy(self.__weights))
            self.__mc_ind.append(np.copy(self.__alpha < self.tol))
        self.__t = t

    def predict(self, x):
        print("Predict is invoked with {}".format(x))
        ind = self.__alpha < self.tol
        return (self.sigmoid(np.dot(self.design_matrix(x)[:,ind], self.__weights[ind])) > 0.5).astype(np.int)

    def predict_proba(self, x):
        ind = self.__alpha < self.tol
        y = self.sigmoid(np.dot(self.design_matrix(x)[:,ind], self.__weights[ind]))
        return np.dstack((1 - y, y))[0]

    def predict_one_vs_rest(self, x):
        y = []
        for ind, weights in zip(self.__mc_ind, self.__mc_weights):
            y.append(self.sigmoid(np.dot(self.design_matrix(x)[:,ind], weights[ind])))
        return np.argmax(np.dstack(y)[0], axis=1)

    def plot_decision_border(self):
        self.plot_data_points("classified with RVM")
        rv_ind = self.__alpha < self.tol
        grid_x = np.arange(self.__X.min() - 0.1, self.__X.max() + 0.1, 0.025)
        grid_x1, grid_x2 = np.meshgrid(grid_x, grid_x)
        grid_points = np.column_stack((np.ravel(grid_x1), np.ravel(grid_x2)))
        Phi = self.design_matrix(grid_points)[:,rv_ind]
        y = self.sigmoid(Phi.dot(self.__weights[rv_ind]))
        y_grid = np.reshape(y, grid_x1.shape)
        plt.contour(grid_x1, grid_x2, y_grid, levels=[0.5], colors="k", linestyles='dashed')
        plt.scatter(self.__X[rv_ind[1:],0], self.__X[rv_ind[1:],1], s=80, facecolors='none', edgecolors='g')

    def plot_data_points(self, title):
        plt.title(title)
        c1, c2 = self.__t == 0, self.__t == 1
        plt.scatter(self.__X[c1,0], self.__X[c1,1], marker="x")
        plt.scatter(self.__X[c2,0], self.__X[c2,1], marker="o")

    def rv(self):
        return self.__X[self.rv_ind()]

    def rv_ind(self):
        """     
        Returns:
            numpy.array: the array of relevance vectors' boolean indices in the original data set
        """
        return (self.__alpha < self.tol)[1:]


if __name__ == '__main__':
    N = 200
    # x, t = generate_data(N)
    # x, t = load_pima() # estimated by CV, gamma = 0.05
    # x, t = load_iris(return_X_y=True)
    x, t = load_usps()
    # x, t = gaussian_mixture_data(N)
    x_new = np.empty((1, x.shape[1]))
    t_new = np.empty((1,))
    for ti in np.unique(t):
        ind = (t == ti)
        x_new = np.vstack((x_new, x[ind,:][:np.sum(ind) // 3]))
        t_new = np.hstack((t_new, t[ind][:np.sum(ind) // 3]))

    x = x_new[1:]
    t = t_new[1:]
    print(x.shape)

    # scaling
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
      
    # gamma = estimate_gamma_cls((x, t), rbf_kernel, 0.01, 1, 0.01)
    # print(gamma)
    # gamma = 0.05
    # rvc = RVC(lambda x, y: rbf_kernel(x, y, gamma=gamma))
    rvc = OneVsRestClassifier(RVC(rbf_kernel))
    rvc.fit(x, t)
    for est in rvc.estimators_:
        print("Amount of relevance vectors:", est.rv().shape[0])

    y = rvc.predict(x)
    print("RVM train error:", np.sum(y != t))

    # rvc.plot_data_points("original")
    # plt.savefig("original.pdf")
    # plt.show()
    # rvc.plot_decision_border()
    # plt.title("RVM: error={}% vectors={}".format(np.sum(y != t) * 100 / len(y), rvc.rv().shape[0]))
    # plt.savefig("rvm.pdf")
    # plt.show()

    x_test, t_test = load_usps(test=True)
    x_test = scalar.fit_transform(x_test)
    
    y = rvc.predict(x_test)
    print("RVM test error:", np.sum(y != t_test))

    svc = OneVsRestClassifier(SVC(kernel=rbf_kernel)) # 0.01 by CV
    svc.fit(x, t)
    # print("Amount of support vectors:", svc.support_.shape[0])
    for est in svc.estimators_:
        print("Amount of support vectors: {}, by class: {}".format(est.support_.shape[0], est.n_support_))

    y = svc.predict(x)
    print("SVM train error:", np.sum(y != t))

    # plot_svm_boundary(svc, x, y, t)
    
    y = svc.predict(x_test)
    print("SVM test error:", np.sum(y != t_test))
