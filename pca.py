# -*- coding: utf-8 -*-
import numpy as np
import logging as log

class PCA (object):

    def __init__(self, data_set):
        self.data_init = data_set
        
        self.avg = None
        self.eigenVectors = None
        self.eigenValues = None
        self.weightsVectors = None
    
    def transform(self, n_eigens=None):
        # Normalisation des vecteurs images
        self.avg = np.mean(self.data_init, axis=1)
        A = self.data_init - self.avg.reshape( self.avg.shape[0], 1 )

        # Decomposition en valeurs propres
        # Mode "rapide"
        # U = eigenvectors
        U, S, V = np.linalg.svd( A, full_matrices=False )

        # Reduction du nombre de eigenface à K
        #TODO Calculer le coefficient optimal
        #TODO uniliser le parametre n_eigen
        k = A.shape[1]
        log.debug("Dimentionnality: "+ str(int(k)) +" - Before: "+ str(A.shape[1]))
        
        self.eigenVectors = U[:, :int(k)]
        log.debug("self.eigenVectors shape:" + str(self.eigenVectors.shape) )

        # Calcul des eigenvalues
        self.eigenValues = np.square(S)

        # Projection des images dans la base des eigenvectors
        self.weightsVectors = np.dot( np.transpose(self.eigenVectors), A )
    
        return self.eigenVectors, self.eigenValues, self.weightsVectors
        
    ##### ATTENTION les accesseurs/fcontions suivants doivent êtres appelé APRES transform()
    # Accesseur
    def getAVG(self):
        return self.avg
    
    def getEigenVectors(self):
        return self.eigenVectors
    
    def getEigenValues(self):
        return self.eigenValues
    
    def getWeightsVectors(self):
        return self.weightsVectors
    
    # Projection des images de test dans le "eigen space"
    def getProjection(self, data):
        return np.dot( np.transpose( self.eigenVectors ), (data  - self.avg) )
