#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.io import arff
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn import cluster, metrics
path = "./artificial/"
databrut = arff.loadarff(open(path+"D31.arff", "r"))
datanp = [[x[0],x[1]] for x in databrut [0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

model = cluster.KMeans(n_clusters=31, init='k-means++')
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

distmatrix = euclidean_distances( datanp )
fp = kmedoids.fasterpam( distmatrix , 31 )

print(metrics.mutual_info_score(model.labels_,fp.labels))