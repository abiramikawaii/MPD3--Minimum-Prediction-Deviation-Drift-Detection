import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.trees.hoeffding_tree import HoeffdingTree



class MPD3:
    def __init__(self, bootstrap, quantile_p ,threshold):
        """
        :param estimator: an initialized sklearn estimator, must have a .predict_proba method
        :param bootstrap: number of samples to bootstrap.

        notes to ourselves:

        steps:

        Step 1 - ensemble_bootstrap
        a. bootstrap the dataset n times
        b. train n models on the n bootstrapped data

        Step 2 - MPD
        a. Give a sample, x, to n-models, and get n predictions on x.
        b. Calculate the mpd value of x, using the equations.
        """

        self.b = bootstrap
        self.q = quantile_p
        self.T = threshold

        self.bootstrap_clfs = []
        self.total_probs_ = np.array([])

        # self.mpd_list = []  # will be used to record the mpd values of each sample in X
        # self.data_mpd_mean = []
        # self.self.total_probs_ = []

    def ensemble_bootstrap(self,train_X,train_y):
        """
        :param train_X: An array of feature samples to train
        :param train_y: A vector of labels to train
        
        This function resamples the X and y data and trains n models on them.
        
        :return:
        :bootstrap_clfs: An ensemble of classifiers
        """
        X_b = train_X
        y_b = train_y
        bootstrap_clfs = []
        X_sparse_matrix = coo_matrix(X_b)
        for i in range(self.b):
            X, X_sparse, y = resample(X_b, X_sparse_matrix, y_b,replace = True)
            self.clf_b  = DecisionTreeClassifier()
            self.clf_b = self.clf_b.fit(X,y)
            bootstrap_clfs.append(self.clf_b)
        self.bootstrap_clfs = bootstrap_clfs
        return self.bootstrap_clfs

    def MPD_score(self,X_batches,ensemble):
        """
        :param X_batches: An array of features samples to test
        This function takes x, to n-models, and get n predictions on x. And calculate the mpd value of x, using the equations.
        
        :return: array of mpd values for the input array
        """
        Test_X = X_batches
        total_probs = []
        for clf_e in ensemble:
            probs_classes = clf_e.predict_proba(Test_X)
            probs = probs_classes[:,0]
            total_probs.append(probs)
        self.total_probs_ = np.array(total_probs)
        #mpd.......
        mpd = []
        U_0_X = (self.total_probs_-0)**2
        U_0_X = U_0_X.sum(axis=0)
        U_0_X = np.sqrt(U_0_X/100)
        U_1_X = (self.total_probs_-1)**2
        U_1_X = U_1_X.sum(axis=0)
        U_1_X = np.sqrt(U_1_X/100)
        mpd = np.minimum(U_0_X, U_1_X)
        return mpd

    
    def drift_check(self,mpd_value):
        m = np.quantile(mpd_value,self.q)
        if m > self.T:
            return True
        else:
            return False



