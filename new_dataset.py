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
from sklearn.metrics.pairwise import euclidean_distances
import kmedoids


# Donnees dans datanp

path = './dataset-rapport/'
name = "zz2"

databrut = pd.read_csv(path+name+".txt", sep = " ", encoding = "ISO-8859-1", skipinitialspace=True)
datanp = databrut.to_numpy()

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]




# =============================================================================
# silhouette, silhouette_t = [], []
# 
# # Détermination du nombre de clusters par coefficient de silhouette
# range_clust = range(2, 20)
# for n_clusters in range_clust:
#     print("Appel Méthode Clustering ")
#     tps1 = time.time()
# # =============================================================================
#     # model = cluster.KMeans(n_clusters=n_clusters, init='k-means++')
#     # model.fit(datanp)
# # =============================================================================
#     # distmatrix = euclidean_distances( datanp )
#     # model = kmedoids.fasterpam( distmatrix , n_clusters )
# # =============================================================================
#     model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
#                                             linkage='average') # Changez le linkage entre 'single', 'average', 'complete' et 'ward'
#     model.fit(datanp)
# # =============================================================================
#     tps2 = time.time()
#     labels = model.labels_ # Enlevez "_" si vous utilisez k-Medoids
#     runtime_method = tps2 - tps1
#     tps1 = time.time()
#     silhouette.append(calinski_harabasz_score(datanp, labels))
#     tps2 = time.time()
#     tp_silhouette = tps2 - tps1
#     silhouette_t.append(tp_silhouette)
#     print("nb clusters =", n_clusters,
#           ",runtime method = ", round((runtime_method) * 1000, 2), "ms",
#           ",runtime silhouette = ", round((tp_silhouette) * 1000, 2), "ms" )
#     
# print("Best nb cluster :", silhouette.index(max(silhouette))+2, "| Result :", max(silhouette))
# 
# plt.title("Dataset "+name)
# plt.plot(range_clust, silhouette, 'r')
# plt.xlabel("n_clusters")
# plt.legend(["Silhouette (avg. "+str(round( (sum(silhouette_t)/len(range_clust)) * 1000, 2))+"ms)"])
# plt.show()
# =============================================================================










# =============================================================================
# # List of eps
# min_samples = [k for k in range(1, 2)]
# for mins in min_samples:
#     distances = [k / 0.01 for k in range(1, 100)] # Changer la distance pour préciser la valeur de epsilon
#     silhouette = []
#     print(len(datanp))
#     for eps in distances[:]:
#         # set distance_threshold (0 ensures we compute the full tree )
#         tps1 = time.time()
#         model = cluster.DBSCAN(eps=eps, min_samples=mins)
#         model = model.fit(datanp)
#         tps2 = time.time()
#         labels = model.labels_
#         print("eps = " + str(eps) + " nb labels = " + str(
#             len(labels)))
#         try:
#             _ = silhouette_score(datanp, labels[:])
#             print("aaaaaaaaa")
#             # if _ < 0 :
#             #     raise ArithmeticError
#         except:
#             _ = 0
#         silhouette.append(_)
#     print("Best eps :", distances[silhouette.index(max(silhouette))], "min_sample :", mins, "| Result :", max(silhouette))
#     # Affichage silhouette en fonction d'eps
#     plt.plot(distances, silhouette, label="min_s=" + str(mins))
# 
# plt.legend()
# plt.title("DBSCAN Score Silhouette en fonction eps\nDataset: "+name+".txt")
# plt.show()
# =============================================================================












# =============================================================================
# min_samples = [k for k in range(60, 65, 1)]
# for mins in min_samples:
#     mins_clus = [k  for k in range(2, 70, 1)]
#     silhouette = []
#     print(len(datanp))
#     for min in mins_clus:
#         # set distance_threshold (0 ensures we compute the full tree )
#         tps1 = time.time()
#         model = hdbscan.HDBSCAN(min_cluster_size=min, min_samples=mins)
#         model = model.fit(datanp)
#         tps2 = time.time()
#         labels = model.labels_
#         # print("min_cluster_size= " + str(min) + " nb labels = " + str(
#         #     len(labels)))
#         try:
#             _ = silhouette_score(datanp, labels[:])
#         except:
#             print("Exception")
#             _ = 0
#         silhouette.append(_)
#     print("Best min_cluster_size :", mins_clus[silhouette.index(max(silhouette))], "min_sample :", mins, "| Result :", max(silhouette))
#     plt.plot(mins_clus, silhouette, label="min_s=" + str(mins))
# 
# # plt.legend()
# plt.title("HDBSCAN Score Silhouette en fonction de min_cluster_size \nDataset: " + name + ".txt")
# plt.show()
# =============================================================================







# k-Means
tps1_kmeans = time.time()
# Changez les paramètres de la fonction ci-dessous
model_kmeans = cluster.KMeans(n_clusters = 5, init='k-means++')
model_kmeans.fit(datanp)
tps2_kmeans = time.time()
tps_kmeans = tps2_kmeans-tps1_kmeans
labels_kmeans = model_kmeans.labels_
score_silhouette_kmeans = silhouette_score(datanp, labels_kmeans)
score_calinski_harabasz_kmeans = calinski_harabasz_score(datanp, labels_kmeans)
score_davies_bouldin_kmeans = davies_bouldin_score(datanp, labels_kmeans)

# k-Medoids
tps1_kmedoids = time.time()
distmatrix = euclidean_distances(datanp)
# Changez les paramètres de la fonction ci-dessous
model_kmedoids = kmedoids.fasterpam(distmatrix, 5)
tps2_kmedoids = time.time()
tps_kmedoids = tps2_kmedoids-tps1_kmedoids
labels_kmedoids = model_kmedoids.labels
score_silhouette_kmedoids = silhouette_score(datanp, labels_kmedoids)
score_calinski_harabasz_kmedoids = calinski_harabasz_score(datanp, labels_kmedoids)
score_davies_bouldin_kmedoids = davies_bouldin_score(datanp, labels_kmedoids)

# Agglo
tps1_agglo = time.time()
# Changez les paramètres de la fonction ci-dessous
model_agglo = cluster.AgglomerativeClustering(n_clusters = 5,
                                              linkage = 'average')
model_agglo.fit(datanp)
tps2_agglo = time.time()
tps_agglo = tps2_agglo-tps1_agglo
labels_agglo = model_agglo.labels_
score_silhouette_agglo = silhouette_score(datanp, labels_agglo)
score_calinski_harabasz_agglo = calinski_harabasz_score(datanp, labels_agglo)
score_davies_bouldin_agglo = davies_bouldin_score(datanp, labels_agglo)

# DBSCAN
tps1_dbscan = time.time()
# Changez les paramètres de la fonction ci-dessous
model_dbscan = cluster.DBSCAN(eps = 1, min_samples = 1)
model_dbscan.fit(datanp)
tps2_dbscan = time.time()
tps_dbscan = tps2_dbscan-tps1_dbscan
labels_dbscan = model_dbscan.labels_
score_silhouette_dbscan = silhouette_score(datanp, labels_dbscan)
score_calinski_harabasz_dbscan = calinski_harabasz_score(datanp, labels_dbscan)
score_davies_bouldin_dbscan = davies_bouldin_score(datanp, labels_dbscan)

# HDBSCAN
tps1_hdbscan = time.time()
# Changez les paramètres de la fonction ci-dessous
model_hdbscan = hdbscan.HDBSCAN(min_cluster_size = 66,
                                min_samples = 62)
model_hdbscan.fit(datanp)
tps2_hdbscan = time.time()
tps_hdbscan = tps2_hdbscan-tps1_hdbscan
labels_hdbscan = model_hdbscan.labels_
score_silhouette_hdbscan = silhouette_score(datanp, labels_hdbscan)
score_calinski_harabasz_hdbscan = calinski_harabasz_score(datanp, labels_hdbscan)
score_davies_bouldin_hdbscan = davies_bouldin_score(datanp, labels_hdbscan)

print("SILHOUETTE")

print("Score k-Means = ", score_silhouette_kmeans, "| runtime =", round((tps_kmeans) * 1000, 2))
print("Score k-Medoids = ", score_silhouette_kmedoids, "| runtime =", round((tps_kmedoids) * 1000, 2))
print("Score Agglo = ", score_silhouette_agglo, "| runtime =", round((tps_agglo) * 1000, 2))
print("Score DBSCAN = ", score_silhouette_dbscan, "| runtime =", round((tps_dbscan) * 1000, 2))
print("Score HDBSCAN = ", score_silhouette_hdbscan, "| runtime =", round((tps_hdbscan) * 1000, 2))

print("CALINSKI_HARABASZ")

print("Score k-Means = ", score_calinski_harabasz_kmeans, "| runtime =", round((tps_kmeans) * 1000, 2))
print("Score k-Medoids = ", score_calinski_harabasz_kmedoids, "| runtime =", round((tps_kmedoids) * 1000, 2))
print("Score Agglo = ", score_calinski_harabasz_agglo, "| runtime =", round((tps_agglo) * 1000, 2))
print("Score DBSCAN = ", score_calinski_harabasz_dbscan, "| runtime =", round((tps_dbscan) * 1000, 2))
print("Score HDBSCAN = ", score_calinski_harabasz_hdbscan, "| runtime =", round((tps_hdbscan) * 1000, 2))

print("DAVIES_BOULDIN")

print("Score k-Means = ", score_davies_bouldin_kmeans, "| runtime =", round((tps_kmeans) * 1000, 2))
print("Score k-Medoids = ", score_davies_bouldin_kmedoids, "| runtime =", round((tps_kmedoids) * 1000, 2))
print("Score Agglo = ", score_davies_bouldin_agglo, "| runtime =", round((tps_agglo) * 1000, 2))
print("Score DBSCAN = ", score_davies_bouldin_dbscan, "| runtime =", round((tps_dbscan) * 1000, 2))
print("Score HDBSCAN = ", score_davies_bouldin_hdbscan, "| runtime =", round((tps_hdbscan) * 1000, 2))

# Affichage clustering
plt.scatter(f0, f1, c=labels_kmeans, s=8)
plt.title(" Resultat du clustering k-Means pour "+ name+".txt")
plt.show()

plt.scatter(f0, f1, c=labels_kmedoids, s=8)
plt.title(" Resultat du clustering k-Medoids pour "+ name+".txt")
plt.show()

plt.scatter(f0, f1, c=labels_agglo, s=8)
plt.title(" Resultat du clustering Agglo pour "+ name+".txt")
plt.show()

plt.scatter(f0, f1, c=labels_dbscan, s=8)
plt.title(" Resultat du clustering DBSCAN pour "+ name+".txt")
plt.show()

plt.scatter(f0, f1, c=labels_hdbscan, s=8)
plt.title(" Resultat du clustering HDBSCAN pour "+ name+".txt")
plt.show()


