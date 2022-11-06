# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:03:24 2022

@author: jeanv
"""

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn.metrics import silhouette_score
import hdbscan

# Donnees dans datanp

path = "./artificial/"
dataset_name = "long2"
databrut = arff.loadarff(open(path + dataset_name + ".arff", "r"))
datanp = [[x[0], x[1]] for x in databrut[0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

mins = [k for k in range(2, 150)]
silhouette = []
print(len(datanp))
for min in mins:
    # set distance_threshold (0 ensures we compute the full tree )
    tps1 = time.time()
    model = hdbscan.HDBSCAN(min_cluster_size=min,metric="")
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    print("min_cluster_size= " + str(min) + " nb labels = " + str(
        len(labels)))
    try:
        _ = silhouette_score(datanp, labels[:])
    except:
        print("Exception")
        _ = 0
    silhouette.append(_)
# Affichage silhouette en fonction d'eps
plt.plot(mins, silhouette)
plt.title("HDBSCAN Score Silhouette en fonction de min_cluster_size \nDataset: " + dataset_name + ".arff")
plt.show()
