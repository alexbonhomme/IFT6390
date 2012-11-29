# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import tools

class Eigenfaces (object):

    def __init__(self, theta, k):
        self.theta = theta
        self.k = k
        
        #Init
        self.train_data_inds = None
        self.eigenVectors = None
        self.eigenValues = None
        self.projectedTrainFaces = None
        self.avgFaceVector = None
    
    def train(self, train_data, train_data_inds):
        # Etiquettes
        self.train_data_inds = train_data_inds
        
        # Calcul du nombre de class
        #TODO Devrait peut etre etre inclu dans le fichier de test... maybe
       	m = []
       	for i in range(0, int( self.train_data_inds.shape[0] )):
		    if( not self.train_data_inds[i] in m ):
			    m.append( self.train_data_inds[i] )
        self.nClass = np.size(m)
        
        # Calcul des eigenvectors, eigenValues et projection
        (self.eigenVectors, self.eigenValues, self.projectedTrainFaces) = self.__doPCAandProjection( train_data )
        
        # Visage moyen
        self.avgFaceVector = np.mean(train_data , axis=1)
    
    def compute_predictions(self, test_data, type_algo=""):
        
        # Projection du visage de test
        projectedTestFace = self.__projectTestFace( test_data, self.eigenVectors, self.avgFaceVector )

		# Switch sur le type d'algo
        if type_algo == "knn":
            return self.__kNearestNeighbors( projectedTestFace, k=self.k )

        elif type_algo == "parzen":
            return self.__parzenWindow( projectedTestFace, k=self.k, theta=self.theta )

        else:
            return self.__findNearestNeighbor( projectedTestFace )
            

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
    def __findNearestNeighbor( self, projectTestFace, distCalc='E' ):
	    nearestNeighbor = 0
	    minDeltaSq = float('inf')
	
	    # for each projection
	    for p in range(0, int( self.projectedTrainFaces.shape[1] )):
		    deltaSq = 0
		
		    # for each coef
		    for i in range(0, int( self.projectedTrainFaces.shape[0] )):
			    delta = projectTestFace[i] - self.projectedTrainFaces[i, p]
			     
			    if( distCalc == 'M' ):
				    deltaSq += delta*delta / self.eigenValues[i];  # Mahalanobis distance
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
    def __parzenWindow( self, projectTestFace, k=3, theta=0.5 ):
	"""
	    # Calcul de m
	    m = []
	    for i in range(0, int( dataTrainIndices.shape[0] )):
		    if( not dataTrainIndices[i] in m ):
			    m.append( dataTrainIndices[i] )
	    m = np.size(m)
	    c = np.zeros(m) #tab de taille m = nb classe
	"""
        c = np.zeros(self.nClass)
	    
        voisins = np.zeros(k)
        voisins[:] = None
        dists = np.zeros(k)
        dists[:] = 0.0
	
        trainMatMax = self.projectedTrainFaces.max()
        projectTrainFace = self.projectedTrainFaces / trainMatMax
        projectTestFace = projectTestFace / trainMatMax

        for p in range(0, int( projectTrainFace.shape[1] )):
	        gaussValue = 1
	        res = self.__gaussianKernel( projectTestFace, projectTrainFace[:,p], theta )
	        for i in range(0, res.shape[0]):
		        gaussValue *= res[i]
	
	        #print gaussValue
	
	        j = np.argmin(dists)
	        if( gaussValue > dists[j] ):
		        voisins[j] = self.train_data_inds[p] # etiquette de la donnee
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
    def __kNearestNeighbors( self, projectTestFace, k=3 ):

        # Calcul de m
        """
        m = []
        for i in range(0, int( dataTestIndices.shape[0] )):
	        if( not dataTestIndices[i] in m ):
		        m.append( dataTestIndices[i] )
        m = np.size(m)
        c = np.zeros(m) #tab de taille m = nb classe
        """
        c = np.zeros(self.nClass)
	
        voisins = np.zeros(k)
        voisins[:] = None
        dists = np.zeros(k)
        dists[:] = float('inf')

        dist = tools.minkowski_mat( projectTestFace, self.projectedTrainFaces ) #vector distance entre x et data

        for i in range(0, int( dist.shape[0] )):
	        j = np.argmax(dists)
	        if( dist[i] < dists[j] ):
		        voisins[j] = self.train_data_inds[i] # etiquette de la donnee
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
