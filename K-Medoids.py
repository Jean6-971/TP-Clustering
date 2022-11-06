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
name = "twenty"

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

silhouette, davies_bouldin, calinski_harabasz = [],[],[] 
silhouette_t, davies_bouldin_t, calinski_harabasz_t = [],[],[]
# Détermination du nombre de clusters par coefficient de silhouette
range_clust = range(2, 10)
calinski_scale = 7000
for n_clusters in range_clust:
    print("Appel KMeans pour une valeur fixee de k ")
    tps1 = time.time()
    distmatrix = euclidean_distances( datanp )
    model = kmedoids.fasterpam( distmatrix , n_clusters )
    tps2 = time.time()
    labels = model.labels
    tp_kmedoid = tps2 - tps1
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    tps1 = time.time()
    silhouette.append(silhouette_score(datanp, labels))
    tps2 = time.time()
    tp_silhouette = tps2 - tps1
    silhouette_t.append(tp_silhouette)
    iteration = model.n_iter
    print("nb clusters =", n_clusters, ", nb iter =", iteration,
          ",runtime kmedoid = ", round((tp_kmedoid) * 1000, 2), "ms",
          ",runtime silhouette = ", round((tp_silhouette) * 1000, 2), "ms" )
print(silhouette)
plt.title("Dataset "+name)
plt.plot(range_clust, silhouette)
plt.xlabel("n_clusters")
plt.legend(["Silhouette (avg. "+str(round( (sum(silhouette_t)/len(range_clust)) * 1000, 2))+"ms)"])
plt.show()

n_clusters = 20
tps1 = time.time()
distmatrix = euclidean_distances( datanp )
model = kmedoids.fasterpam( distmatrix , n_clusters )
labels = model.labels
tps2 = time.time()

# Affichage clustering
# =============================================================================
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Dataset "+name+"\nK-Mean (n_clusters="+str(n_clusters)+")")
plt.show()
print (" runtime = ", round ((tps2 - tps1)*1000, 2), "ms")

