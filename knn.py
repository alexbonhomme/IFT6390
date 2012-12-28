# -*- coding: utf-8 -*-
import tools
import numpy as np

class KNN (object):

    def __init__(self, train_set, train_targets, nb_class, k, dist_func=None):
        self.k = k
        self.train_set = train_set  # Ensemble d'entrainement pour la recherche des k plus proche voisins
        self.train_targets = train_targets
        self.nClass = nb_class
        
        # Fonction de comparaison
        if dist_func != None:
            self.dist_func = dist_func
        else:
            self.dist_func = tools.minkowski_mat
    
    # Setter
    #TODO cast ? verif ?
    def setK(self, k):
        self.k = k
    
    #TODO comment
    def compute_predictions(self, test_set):
        # vecteur resultat
        c = np.zeros(self.nClass)
	
        voisins = np.zeros(self.k)
        voisins[:] = None
        dists = np.zeros(self.k)
        dists[:] = float('inf')
        
        dist = self.dist_func( test_set, self.train_set )

        for i in xrange( dist.shape[0] ):
	        j = np.argmax(dists)
	        if( dist[i] < dists[j] ):
		        voisins[j] = self.train_targets[i] # etiquette de la donnee
		        dists[j] = dist[i] # distance

        ## DEBUG
        print "Voisins:\t"+ str(voisins)

        # On somme le nombre de voisin de la meme classe
        # pour trouver quelle est la classe majoritaire
        for i in range(0, self.k):
	        if(not np.isnan(voisins[i])):
		        c[ voisins[i]-1 ] += 1 #-1 car les etiquettes commences a 1 mais le tab a zero


        ## DEBUG
        print "C:\t\t"+ str(c)

        return np.argmax( c ) + 1

