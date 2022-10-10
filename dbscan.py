# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:03:24 2022

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


tps1 = time.time()
model = cluster.DBSCAN(eps= 0.7, min_samples= 20)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Affichage clustering
plt.scatter(f0, f1, c=labels, s=8)
plt.title(" Resultat du clustering ")
plt.show()
print (" runtime = ", round ((tps2 - tps1)*1000, 2), "ms")