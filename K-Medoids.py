import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances

# Parser un fichier de donnees au format arff
# Æ data est un tableau d’exemples avec pour chacun la liste des valeurs des features


# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster. On retire cette information

path = "./artificial/"
name = "twodiamonds"

databrut = arff.loadarff(open(path + name+".arff", "r"))
datanp = [[x[0], x[1]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste

# Ex pour f0 = [—0.499261, —1.51369, —1.60321, ...]
# Ex pour fl = [—0.0612356, 0.265446, 0.362039, ..….]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

silhouette, silhouette_t, silhouette_med, silhouette_med_t = [],[],[],[]
# Détermination du nombre de clusters par coefficient de silhouette
range_clust = range(2, 20)
distmatrix = euclidean_distances( datanp )
for n_clusters in range_clust:
    # print("Appel KMedoid pour une valeur fixee de k ")
    tps1 = time.time()
    model = kmedoids.fasterpam( distmatrix , n_clusters )
    tps2 = time.time()
    tp_kmedoid = tps2 - tps1
    labels = model.labels
    
    tps1 = time.time()
    _ = silhouette_score(datanp, labels)
    tps2 = time.time()
    silhouette.append(_)
    silhouette_t.append(tps2 - tps1)
    
    # Evaluation du la metrique embarquée dans kmedoid
    tps1 = time.time()
    _ = kmedoids.medoid_silhouette(distmatrix, model.medoids)
    tps2 = time.time()
    silhouette_med.append(_)
    silhouette_med_t.append(tps2 - tps1)

plt.title("Dataset "+name+"\nsilhouette (sklearn) vs medoid silhouette (kmedoid)")
plt.plot(range_clust, silhouette)
plt.plot(range_clust, [s[0] for s in silhouette_med])
plt.xlabel("n_clusters")
plt.legend(["Silhouette (avg. "+str(round( (sum(silhouette_t)/len(range_clust)) * 1000, 2))+"ms)",
            "Med. Silhouette (avg. "+str(round( (sum(silhouette_med_t)/len(range_clust)) * 1000, 2))+"ms)"])
plt.show()

# n_clusters = 2
# tps1 = time.time()
# distmatrix = euclidean_distances( datanp )
# model = kmedoids.fasterpam( distmatrix , n_clusters )
# labels = model.labels
# tps2 = time.time()

# # Affichage clustering
# # =============================================================================
# plt.scatter(f0, f1, c=labels, s=8)
# plt.title("Dataset "+name+"\nK-Medoid (n_clusters="+str(n_clusters)+") en "+ str( round ((tps2 - tps1)*1000, 2))+ "ms")
# plt.show()
# print (" runtime = ", round ((tps2 - tps1)*1000, 2), "ms")

