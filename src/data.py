# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:02:01 2020

@author: ivanlimaWin
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
'''
    Method used to create Datasets for the models.
'''
def create_dataset(samples, features, classes, random_seed):
  dataset, labels = make_classification(n_samples=samples, n_features=features, n_informative=2, n_redundant=0, n_repeated=0, n_classes=classes, n_clusters_per_class=1, random_state=random_seed)
  data_train, data_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.5, random_state=random_seed)

  return data_train, data_test, labels_train, labels_test
print(labels_train)