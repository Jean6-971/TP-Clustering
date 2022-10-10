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


# Donnees dans datanp

path = "./artificial/"
databrut = arff.loadarff(open(path+"R15.arff", "r"))
datanp = [[x[0],x[1]] for x in databrut [0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]


# Distances k plus proches voisins
# Donnees dans X
k=5
neigh = NearestNeighbors(n_neighbors= k)
neigh.fit(datanp)
distances, indices = neigh.kneighbors(datanp)

# retirer le point " origine "
newDistances = np.asarray([np.average(distances[i][1:]) for i in range (0, distances.shape[0])])
trie = np.sort(newDistances)
plt.title(" Plus proches voisins (5)")
plt.plot(trie);
plt.show()


silhouette = []
davies_bouldin = []
calinski_harabasz = []
range_min_samples = range(1,20)
for min_samples in range_min_samples : 
    tps1 = time.time()
    model = cluster.DBSCAN(eps= 0.3, min_samples= min_samples)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    tps_dbscan = tps2 - tps1
    
    tps1 = time.time()
    silhouette.append(silhouette_score(datanp, labels))
    tps2 = time.time()
    tp_silhouette = tps2 - tps1
    tps1 = time.time()
    davies_bouldin.append(davies_bouldin_score(datanp, labels))
    tps2 = time.time()
    tp_bouldin = tps2 - tps1
    tps1 = time.time()
    calinski_harabasz.append(calinski_harabasz_score(datanp, labels)/500)
    tps2 = time.time()
    tp_calinski = tps2 - tps1
    print ("min sample =",min_samples, 
           ",runtime dbscan = ", round((tps_dbscan)*1000,2),"ms",
           ",runtime silhouette = ", round((tp_silhouette)*1000,2),"ms",
           ",runtime bouldin = ", round((tp_bouldin)*1000,2),"ms", 
           ",runtime calinski = ", round((tp_calinski)*1000,2),"ms",)

print(silhouette)
plt.plot(range_min_samples,silhouette,'r')
plt.plot(range_min_samples,davies_bouldin,'g')
plt.plot(range_min_samples,calinski_harabasz,'b')
plt.legend(["silhouette", "davies_bouldin", "calinski_harabasz"])

plt.show() 

# Affichage clustering
# =============================================================================
# plt.scatter(f0, f1, c=labels, s=8)
# plt.title(" Resultat du clustering ")
# plt.show()
# print (" runtime = ", round ((tps2 - tps1)*1000, 2), "ms")
# =============================================================================
