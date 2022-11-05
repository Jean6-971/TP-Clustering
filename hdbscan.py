# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:03:24 2022
"""

import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.metrics import silhouette_score
import hdbscan

path = "./artificial/"
dataset_name = "smile1"
databrut = arff.loadarff(open(path + dataset_name+".arff", "r"))
datanp = [[x[0], x[1]] for x in databrut[0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

# List of eps
min_samples = [k for k in range(2, 10)]
for mins in min_samples:
    distances = [k / 100 for k in range(1,10)]
    silhouette = []
    print(len(datanp))
    for eps in distances[:]:
        # set distance_threshold (0 ensures we compute the full tree )
        tps1 = time.time()
        #model = hdbscan.HDBSCAN(eps=eps, min_samples=mins)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        print("eps = " + str(eps) + " nb labels = " + str(
            len(labels)))
        try:
            _ = silhouette_score(datanp, labels[:])
            if _ < 0 :
                raise ArithmeticError
        except:
            _ = 0
        silhouette.append(_)
    # Affichage silhouette en fonction d'eps
    plt.plot(distances, silhouette, label="min_s=" + str(mins))

plt.legend()
plt.title("Score Silhouette en fonction 0.1 < eps < 0.25 \nDataset: "+dataset_name+".arff")
plt.show()
