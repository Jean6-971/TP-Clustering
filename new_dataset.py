# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:03:24 2022

@author: jeanv
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

import scipy.cluster.hierarchy as shc
from scipy.io import arff
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan


# Donnees dans datanp

path = './dataset-rapport/'
filename = "x2.txt"
databrut = pd.read_csv(path+filename, sep = " ", encoding = "ISO-8859-1", skipinitialspace=True)
data = databrut
datanp = databrut.to_numpy()

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

########################################

tps1 = time.time()
clusterer = hdbscan.HDBSCAN()
clusterer.fit(datanp)
labels = clusterer.labels_
tps2 = time.time()

########################################

# tps1 = time.time()
# model = cluster.DBSCAN(eps= 0.032, min_samples= 8)
# model = model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_

# Affichage clustering
# =============================================================================
plt.scatter(f0, f1, c=labels, s=8)
plt.title(" Resultat du clustering HDBSCAN ")
plt.show()
print (" runtime = ", round ((tps2 - tps1)*1000, 2), "ms")
# =============================================================================
