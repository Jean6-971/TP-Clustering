import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score


# Parser un fichier de donnees au format arff
#Æ data est un tableau d’exemples avec pour chacun la liste des valeurs des features


# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster. On retire cette information

path = "./artificial/"
databrut = arff.loadarff(open(path+"R15.arff", "r"))
datanp = [[x[0],x[1]] for x in databrut [0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste

# Ex pour f0 = [—0.499261, —1.51369, —1.60321, ...]
# Ex pour fl = [—0.0612356, 0.265446, 0.362039, ..….]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

plt.scatter(f0, f1, s=8)
plt.title ("Donnees initiales")
plt.show()

silhouette = []
davies_bouldin = []
calinski_harabasz = []

# Détermination du nombre de clusters par coefficient de silhouette
range_clust = range(2,50)
for n_clusters in range_clust : 
    #print ("Appel KMeans pour une valeur fixee de k ")
    #tps1 = time.time()
    model = cluster.KMeans(n_clusters=n_clusters, init='k-means++')
    model.fit(datanp)
    #tps2 = time.time()
    labels = model.labels_
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette.append(silhouette_score(datanp, labels))
    davies_bouldin.append(davies_bouldin_score(datanp, labels))
    calinski_harabasz.append(calinski_harabasz_score(datanp, labels)/7000)
print(silhouette)
plt.plot(range_clust,silhouette,'r')
plt.plot(range_clust,davies_bouldin,'g')
plt.plot(range_clust,calinski_harabasz,'b')

plt.show() 
# =============================================================================
#     iteration = model.n_iter_
#     
#     plt.scatter(f0, f1, c=labels, s=8)
#     plt.title("Donnees apres clustering Kmeans")
#     plt .show ()
#     print ("nb clusters =",n_clusters,", nb iter =",iteration, ",runtime = ", round((tps2-tps1)*1000,2),"ms")
# =============================================================================
