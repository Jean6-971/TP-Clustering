# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:27:13 2022

@author: jeanv
"""

import scipy . cluster . hierarchy as shc
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster

# Donnees dans datanp

path = "./artificial/"
databrut = arff.loadarff(open(path+"R15.arff", "r"))
datanp = [[x[0],x[1]] for x in databrut [0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

print (" Dendrogramme ’single ’ donnees initiales ")
linked_mat = shc.linkage(datanp, 'single')
plt.figure(figsize = (12, 12))
shc.dendrogram(linked_mat,
orientation = 'top',
distance_sort = 'descending',
show_leaf_counts = False)
plt.show()

range_threshold = range(0,1000)
for distance_threshold in range_threshold :
    # set distance_threshold (0 ensures we compute the full tree )
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(distance_threshold/100, linkage ='single', n_clusters = None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_

# Affichage clustering
plt.scatter(f0, f1, c=labels, s=8)
plt.title(" Resultat du clustering ")
plt.show()
print ("nb clusters =",k ,", nb feuilles = ", leaves, " runtime = ", round ((tps2 - tps1)*1000, 2), "ms")

# =============================================================================
# # set the number of clusters
# k = 4
# tps1 = time.time()
# model = cluster.AgglomerativeClustering(linkage = 'single', n_clusters = k)
# model = model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# kres = model.n_clusters_
# leaves = model.n_leaves_
# 
# # Affichage clustering
# plt.scatter(f0, f1, c=labels, s=8)
# plt.title(" Resultat du clustering ")
# plt.show()
# print ("nb clusters =",k ,", nb feuilles = ", leaves, " runtime = ", round ((tps2 - tps1)*1000, 2), "ms")
# =============================================================================
