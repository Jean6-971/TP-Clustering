import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff


# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
def visu_dataset(name):
    path = "./artificial/"
    databrut = arff.loadarff(open(path + name + ".arff", 'r'))
    datanp = [[x[0], x[1]] for x in databrut[0]]
    # Affichage en 2D
    # Extraire chaque valeur de features pour en faire une liste
    # Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
    # Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    plt.scatter(f0, f1, s=8)
    plt.title("Visualisation du dataset "+name)
    plt.show()
