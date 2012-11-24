# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp


class Eigenfaces (object):

    def __init__(self, theta, k):
        self.theta = theta
        self.k = k
        
        #Init
        self.train_data_inds = None#= np.array([])
        self.eigenVectors = None
        self.eigenValues = None
        self.projectedTrainFaces = None
    
    def train(self, train_data, train_data_inds):
        self.train_data_inds = train_data_inds
        # Calcul des eigenvectors, eigenValues et projection
        self.eigenVectors,
        self.eigenValues, 
        self.projectedTrainFaces = self.__doPCAandProjection( train_data )
    
    def compute_predictions(self, test_data, type_algo=""):
        avgFace = np.mean(test_data, axis=1)
        
        #TODO need loop to do projection
        #
        projectedTestFace = self.__projectTestFace( test_data, self.eigenVectors, avgFace )

        if type_algo == "knn":
            return self.__kNearestNeighbors( projectedTestFace, self.projectedTrainFaces, self.train_data_inds, k=self.k )

        elif type_algo == "parzen":
            return self.__parzenWindow( projectedTestFace, self.projectedTrainFaces, self.train_data_inds, k=self.k, theta=self.theta )

        else:
            return self.__findNearestNeighbor( projectedTrainFaces, self.projectedTestFace, self.eigenValues )
            

## Fonctions annexes (a trier)
    def __doPCAandProjection( self, data ):

	    # Normalisation des vecteurs images
	    avg = np.mean(data, axis=1)
	    A = data - avg.reshape( avg.shape[0], 1 )

	    # Decomposition en valeurs propres
	    # Mode "rapide"
	    # U = eigenvectors
	    U, S, V = np.linalg.svd( A, full_matrices=False )
	
	    # Reduction du nombre de eigenface à K
	    #TODO Calculer le coefficient optimal
	    k = A.shape[1] / 1.5 # Nombre d'image "entrainé" -1
	    print "Dimentionnality: "+ str(k) +" Before: "+ str(A.shape[1])
	    U = U[:, :k]
	
	    # Calcul des eigenvalues
	    S = np.square(S)

	    # Projection des images dans la base des eigenvectors
	    wVectors = np.dot( np.transpose(U), A )
	
	    return U, S, wVectors
    
    def __projectTestFace( self, dataTest, eigenVectors, avgFaceVector ):
	    # Projection des images dans la base des eigenvectors
	    return np.dot( np.transpose(eigenVectors), (dataTest - avgFaceVector) )

    """
	    Recherche du plus proche voisin (k = 1)
	    Utilise la distance Euclidien ou de Mahalanobis
    """
    def __findNearestNeighbor( self, projectTrainFace, projectTestFace, eValues, distCalc='E' ):
	    nearestNeighbor = 0
	    minDeltaSq = float('inf')
	
	    # for each projection
	    for p in range(0, int( projectTrainFace.shape[1] )):
		    deltaSq = 0
		
		    # for each coef
		    for i in range(0, int( projectTrainFace.shape[0] )):
			    delta = projectTestFace[i] - projectTrainFace[i, p]
			     
			    if( distCalc == 'M' ):
				    deltaSq += delta*delta / eValues[i];  # Mahalanobis distance
			    else:	
				    deltaSq += delta*delta # Eclidean distance
		
		    if( deltaSq < minDeltaSq ):
			    minDeltaSq = deltaSq
			    nearestNeighbor = p
	
	    return nearestNeighbor

    """
	    Gaussienne
	    X: centre de la gaussienne
	    x: coordonnée de la valeurs recherché
	    theta: ecart type (ou largeur du noyau...)
    """	
    #def softKernel(X, x, theta=0.5, d=2):
    #	return (1/( ((2*np.pi)**(d/2)) * theta**d )) * np.exp( (-1/2)* ((np.abs( X - x )**2) / (theta**2)) )

    def __gaussianKernel( self,X, x, theta=0.5 ):
	    return np.exp( -((X - x)**2) / (theta**2) ) 

    def __softKernelDist( self, dist, d=2, theta=0.5 ):
	    return (1/( ((2*np.pi)**(d/2)) * (theta**d) )) * np.exp((-1/2) * ((dist) / (theta**2)))

    """
	    Fenetre de Parzen
    """
    def __parzenWindow( self, projectTestFace, projectTrainFace, dataTrainIndices, k=3, theta=0.5 ):
	
	    # Calcul de m
	    m = []
	    for i in range(0, int( dataTrainIndices.shape[0] )):
		    if( not dataTrainIndices[i] in m ):
			    m.append( dataTrainIndices[i] )
	    m = np.size(m)
	
	    c = np.zeros(m) #tab de taille m = nb classe

	    voisins = np.zeros(k)
	    voisins[:] = None
	    dists = np.zeros(k)
	    dists[:] = 0.0
	
	    trainMatMax = projectTrainFace.max()
	    projectTrainFace = projectTrainFace / trainMatMax
	    projectTestFace = projectTestFace / trainMatMax
	
	    for p in range(0, int( projectTrainFace.shape[1] )):
		    gaussValue = 1
		    res = gaussianKernel( projectTestFace, projectTrainFace[:,p], theta )
		    for i in range(0, res.shape[0]):
			    gaussValue *= res[i]
		
		    #print gaussValue
		
		    j = np.argmin(dists)
		    if( gaussValue > dists[j] ):
			    voisins[j] = dataTrainIndices[p] # etiquette de la donnee
			    dists[j] = gaussValue # Coef K()
	
	    # On somme le nombre de voisin de la meme classe
	    # pour trouver quelle est la classe majoritaire
	    for i in range(0, k):
		    if(not np.isnan(voisins[i])):
			    c[ voisins[i]-1 ] += dists[ i ] #-1 car les etiquettes commences a 1 mais le tab a zero

	    print c
	
	    return np.argmax( c ) + 1

    """
	    m = nombre de classes (visages distincts) dans les données d'apprentissage
    """
    def __kNearestNeighbors( self, projectTestFace, projectTrainFace, dataTrainIndices, k=3 ):
	
	    # Calcul de m
	    m = []
	    for i in range(0, int( dataTestIndices.shape[0] )):
		    if( not dataTestIndices[i] in m ):
			    m.append( dataTestIndices[i] )
	    m = np.size(m)
	
	    c = np.zeros(m) #tab de taille m = nb classe
		
	    voisins = np.zeros(k)
	    voisins[:] = None
	    dists = np.zeros(k)
	    dists[:] = float('inf')

	    dist = minkowski_mat( projectTestFace, projectTrainFace ) #vector distance entre x et data

	    for i in range(0, int( dist.shape[0] )):
		    j = np.argmax(dists)
		    if( dist[i] < dists[j] ):
			    voisins[j] = dataTrainIndices[i] # etiquette de la donnee
			    dists[j] = dist[i] # distance
	
	    ## DEBUG
	    print "Voisins:\t"+ str(voisins)
	
	    # On somme le nombre de voisin de la meme classe
	    # pour trouver quelle est la classe majoritaire
	    for i in range(0, k):
		    if(not np.isnan(voisins[i])):
			    c[ voisins[i]-1 ] += 1 #-1 car les etiquettes commences a 1 mais le tab a zero

	
	    ## DEBUG
	    print "C:\t\t"+ str(c)
	
	    return np.argmax( c ) + 1
