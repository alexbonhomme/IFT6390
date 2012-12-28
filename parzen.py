# -*- coding: utf-8 -*-
import numpy as np
import tools

class ParzenWindows (object):

    def __init__(self, train_set, train_targets, nb_class, Theta, kernel_func=None):
        self.theta = Theta
        self.train_set = train_set  # Ensemble d'entrainement pour la recherche des k plus proche voisins
        self.train_targets = train_targets
        self.nClass = nb_class
        
        # Fonction de comparaison
        if kernel_func != None:
            self.kernel_func = kernel_func
        else:
            self.kernel_func = tools.gaussianKernel
    
    # Setter
    def setTheta(self, Theta):
        self.theta = Theta

    #TODO comment
    def compute_predictions(self, test_set, kNN=None):
        if kNN != None:
            print kNN
            k = kNN
        else:
            k = self.nClass

        #vecteur de sortie
        c = np.zeros(self.nClass)
	    
        voisins = np.zeros(k)
        voisins[:] = None
        dists = np.zeros(k)
        dists[:] = 0.0
	
	    #TODO Normalisation ?
	    #TODO ca ressemble a du bricolage... a checker
        trainMatMax = self.train_set.max()
        normalizedTrain = self.train_set / trainMatMax
        test_set = test_set / trainMatMax

        for p in range(0, int( normalizedTrain.shape[1] )):
	        gaussValue = 1
	        res = self.kernel_func( test_set, normalizedTrain[:,p], self.theta )
	        for i in range(0, res.shape[0]):
		        gaussValue *= res[i]
	
	        #print gaussValue
	
	        j = np.argmin(dists)
	        if( gaussValue > dists[j] ):
		        voisins[j] = self.train_targets[p] # etiquette de la donnee
		        dists[j] = gaussValue # Coef K()

        # On somme le nombre de voisin de la meme classe
        # pour trouver quelle est la classe majoritaire
        for i in range(0, k):
	        if(not np.isnan(voisins[i])):
		        c[ voisins[i]-1 ] += dists[ i ] #-1 car les etiquettes commences a 1 mais le tab a zero

        print c

        return np.argmax( c ) + 1

