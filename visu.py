import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster

# Parser un fichier de donnees au format arff
#Æ data est un tableau d’exemples avec pour chacun la liste des valeurs des features


# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster. On retire cette information

path = "./artificial/"
databrut = arff.loadarff(open(path+"banana.arff", "r"))
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

print ("Appel KMeans pour une valeur fixee de k ")
tps1 = time.time()
k = 4
model = cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees apres clustering Kmeans")
plt .show ()
print ("nb clusters =",k,", nb iter =",iteration, ",runtime = ", round((tps2-tps1)*1000,2),"ms")