# MPD3-Minimum  Prediction  Deviation Drift Detection

This repository contains supplemental material of the algorithm to detect concept drift named "Minimum Prediction Deviation Drift Detector" (MPD3). `MPD3.py` makes use of the minimum prediction deviation, a metric which quantifies the uncertainty a classifier's prediction on a particular sample is. The basic approach is to continuously monitor the MPD score for a number of predictions. If the 75\% quantile rise rises above a threshold it indicates that the model is no longer performing with a reliable consistency. We then we consider the data to have drifted. We can then either retrain the model (after obtaining labeled data) or take other reactive measures to prevent the performance degradation of the model. 

![outline](https://user-images.githubusercontent.com/43429949/87806321-4d7caf80-c874-11ea-9dfa-a80db8c38c20.png)

**Dataset:**
* The streaming datasets used in this work from:
  * https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow
* In addition to these streaming datasets a "semi-stationary" phishing dataset is also used.

**Install:**
* You have to install scikit-multiflow in addition to commonly used python libraries. (sklearn, pandas, numpy, matplotlib)
  * https://scikit-multiflow.github.io/
  
**Code overview:**
To detect the drift in data in case of batch data first we load the data from the data folder in `testing_batch.ipynb`file. Here we use streaming and semi-stationary data to detect drift in it. In streaming, we have an artificial and real-world dataset. After loading the data, the data is converted into a set of data batches since our aim here is to find drift in batch data. Before detecting the drift in the batches we need to find value for the parameters q (Quantile percent) and T (Threshold) that is best opted for respective dataset being used. To find the parameters we use `tunning_parameter.py` python file by inputting the first batch of the data. After that, we initialize the bootstrap count (the count of bootstraps), the ensemble classifier, and the base classifier. For the ensemble classifier we use `DecisionTreeClassifier` and for the base classifier, we use `HoeffdingTree`. `MPD3.py` is used for detecting the drift. After that, we create a new instance of the class `MPD3` called `MPD3_detector` using bootstrap count, q, and T as parameters. We start the detection process by training the base classifier using the first batch data. After that, we compute `ensemble` which is an ensemble of the classifier on the first batch using `ensemble_bootstrap()` function. In a loop, we move through the remaining batches in the Data. We compute the MPD score `mpd_value` using the `MPD_score()` function by giving `ensemble` as a parameter and the batch accuracy of the corresponding batch. The MPD score `mpd_value` is a vector of length of the corresponding batch. If the quantile percent `q` of `mpd_value` is greater than T, we trigger a drift alert. If there is drift, we retrain the base classifier and update the `ensemble` with the latest batch data. If there is no drift, we continue the loop.
