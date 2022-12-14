# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:03:24 2022
@author: jeanv
"""

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan

# Donnees dans datanp

path = "./artificial/"
databrut = arff.loadarff(open(path+"hypercube.arff", "r"))
datanp = [[x[0],x[1]] for x in databrut [0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

tps1 = time.time()
clusterer = hdbscan.HDBSCAN()
clusterer.fit(datanp)
labels = clusterer.labels_
tps2 = time.time()

# Affichage clustering
# =============================================================================
plt.scatter(f0, f1, c=labels, s=8)
plt.title(" Resultat du clustering HDBSCAN ")
plt.show()
print (" runtime = ", round ((tps2 - tps1)*1000, 2), "ms")
# =============================================================================