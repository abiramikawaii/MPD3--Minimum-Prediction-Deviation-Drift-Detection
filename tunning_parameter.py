import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from MPD3 import MPD3


class Tunning:
    def __init__(self, X, y):
        self.X_array = X
        self.y_array = y
        
    def parameter_q_and_t(self):
        accuracy_of_combinations  = []
        combination = []
        quantile_percent = [0.50,0.75,1.0]
        threshold = [0.5,0.6,0.7]
        test_X,test_y = get_data_batches(self.X_array,self.y_array)
        ensemble_clf = DecisionTreeClassifier()
        clf = HoeffdingTree()
        bootstrap_count = 100
        for q in quantile_percent:
            for t in threshold:
                Train_X = test_X[0]
                Train_y = test_y[0].flatten()
                clf= clf.fit(Train_X,Train_y)
                MPD3_detector = MPD3(bootstrap_count,q,t)
                ensemble = MPD3_detector.ensemble_bootstrap(Train_X,Train_y)
                batch_accuracy = []
                result = []
                for i in range(len(test_X)-1):
                    index = i+1
                    prediction = clf.predict(test_X[index])
                    batch_accuracy.append(accuracy_score(test_y[index],prediction))
                    mpd_value = MPD3_detector.MPD_score(test_X[index],ensemble)    

                    if MPD3_detector.drift_check(mpd_value):
                        Train_X = test_X[index]
                        Train_y = test_y[index].flatten()
                        clf = clf.partial_fit(Train_X,Train_y)
                        ensemble = MPD3_detector.ensemble_bootstrap(Train_X,Train_y)

                mean_accuracy = np.average(batch_accuracy)
                accuracy_of_combinations.append(mean_accuracy)
                combination.append([q,t])
        index_of_max_acc = np.argmax(accuracy_of_combinations)
        final_q,final_t = combination[index_of_max_acc]
        return final_q,final_t

def get_data_batches(X_batch,y_batch):
	X_data = X_batch
	y_data = y_batch
	n =int(len(X_batch)/15)
	test_X_batches = []
	test_y_batches = []
	test_X_batches = [np.array(X_data[i:i+n]) for i in range(0,X_data.shape[0],n)]
	test_y_batches = [np.array(y_data[i:i+n]) for i in range(0,y_data.shape[0],n)]
	return test_X_batches,test_y_batches

	
        












