#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.io import arff
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score


path = "./artificial/"
databrut = arff.loadarff(open(path+"hypercube.arff", "r"))
datanp = [[x[0],x[1]] for x in databrut [0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

# Détermination du nombre de clusters par coefficient de silhouette
silhouette = []
davies_bouldin = []
calinski_harabasz = []
range_clust = range(2,20)
for n_clusters in range_clust : 
    print ("Appel Kmedoids pour une valeur fixee de k ")
    tps1 = time.time()
    distmatrix = euclidean_distances( datanp )
    fp = kmedoids.fasterpam( distmatrix , n_clusters )
    tps2 = time.time()
    tp_kmedoids = tps2-tps1
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels
    print( " Loss with FasterPAM : " , fp.loss )
    #plt.scatter( f0 , f1 , c = labels_kmed , s = 8 )
    print ( " nb clusters = " ,n_clusters, " , nb iter = " , iter_kmed , " ,runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
    tps1 = time.time()
    silhouette.append(silhouette_score(datanp, labels_kmed))
    tps2 = time.time()
    tp_silhouette = tps2-tps1
    tps1 = time.time()
    davies_bouldin.append(davies_bouldin_score(datanp, labels_kmed))
    tps2 = time.time()
    tp_bouldin = tps2-tps1
    tps1 = time.time()
    #calinski_harabasz.append(calinski_harabasz_score(datanp, labels_kmed)/7000)
    tps2 = time.time()
    tp_calinski = tps2-tps1
    iteration = iter_kmed
    print ("nb clusters =",n_clusters,", nb iter =",iteration, 
           ",runtime kmeans = ", round((tp_kmedoids)*1000,2),"ms",
           ",runtime silhouette = ", round((tp_silhouette)*1000,2),"ms",
           ",runtime bouldin = ", round((tp_bouldin)*1000,2),"ms") 
           #",runtime calinski = ", round((tp_calinski)*1000,2),"ms",)

print(silhouette)
plt.plot(range_clust,silhouette,'r')
plt.plot(range_clust,davies_bouldin,'g')
#plt.plot(range_clust,calinski_harabasz,'b')

plt.show() 

