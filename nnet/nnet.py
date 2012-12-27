# -*- coding: utf-8 -*-
import numpy as np
import tools
import time

class NeuralNetwork (object):

    def __init__(self, n_in, n_hidden, n_out, lr, wd):
        # pas du gradient (taux d'apprentissage)
        self.lr = lr
        
        # coef de la penalité
        self.wd = wd
    
        # Nombre de neurones dans la couche d'entrée
        self.n_in = n_in
        
        # Nombre de neurones dans la couche cachée
        self.n_hid = n_hidden
        
        # Nombre de neurones dans la couche de sortie (classes)
        self.n_out = n_out
        
        # Parametres
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    # Pas mal d'arguments sont ignoré dans cette version pour pouvoir facilement faire la liaison avec les version optimisée
    def train(self, train_set, n_epoch, batch_size=1, fDif=False, EarlyStopping=False):

        features = train_set[:,:-1]
        targets = train_set[:,-1]
        
        train_size = features.shape[0]
        
        # init
        self.W1 = self.__initWeights((self.n_hid, self.n_in), self.n_in)
        self.b1 = np.zeros((self.n_hid, 1))
        self.W2 = self.__initWeights((self.n_out, self.n_hid), self.n_hid)
        self.b2 = np.zeros((self.n_out, 1))
        R = 0.
        out = []
        
        
        # Arret prématuré
        """
        if EarlyStopping:
            prev_R = np.inf
            min_W1 = np.zeros((self.n_hid, self.n_in))
            min_b1 = np.zeros((self.n_hid, 1))
            min_W2 = np.zeros((self.n_out, self.n_hid))
            min_b2 = np.zeros((self.n_out, 1))
        #"""

        if fDif:
            print "Début de l'entrainement par différence finie"
        else:
            print "Début de l'entrainement par descente de gradient type mini-batch"

        t_start = time.clock()
        for k in xrange(n_epoch):
            t1 = time.clock()

            # Non optimal <_<
            for z in xrange(train_size/batch_size):
                
                grad_sumW1 = np.zeros(self.W1.shape)
                grad_sumb1 = np.zeros(self.b1.shape)
                grad_sumW2 = np.zeros(self.W2.shape)
                grad_sumb2 = np.zeros(self.b2.shape)
                
                ct = z * batch_size
                
                # On parcours le "mini-batch"
                for i in xrange(ct, ct + batch_size):
                    # finite difference
                    if fDif:
                        grad_W1, grad_b1, grad_W2, grad_b2 = self.finiteDifference(features[i], targets[i])

                    # back propagation
                    else:                    
                        grad_W1, grad_b1, grad_W2, grad_b2 = self.bprop(features[i], targets[i])
                    
                    # somme des gradients
                    grad_sumW1 += grad_W1
                    grad_sumb1 += grad_b1
                    grad_sumW2 += grad_W2
                    grad_sumb2 += grad_b2
     
                # pénalité L2
                if self.wd > 0:
                    grad_penality = self.compute_penality()[1]
                    grad_sumW1 += grad_penality[0]
                    grad_sumW2 += grad_penality[1]
                  
                # weights update
                self.W1 -= self.lr * grad_sumW1
                self.b1 -= self.lr * grad_sumb1
                self.W2 -= self.lr * grad_sumW2
                self.b2 -= self.lr * grad_sumb2

            # regularisation
            if self.wd > 0:
                penality = self.compute_penality()[0]
            else:
                penality = 0.

            # Affichage perte
            R = np.mean(self.compute_loss(features, targets))# + penality            
            t2 = time.clock()
            print "Epoch:", k, "-- Temps de calcul: %.3fs" % float(t2-t1), " - R:", round(R, 6)
            out.append(R)
            
            # On mémorise les minimums
            """
            if EarlyStopping:
                if prev_R+10e-3 >= R:  #TODO Mettre une marge ?
                    prev_R = R
                    min_W1 = self.W1
                    min_b1 = self.b1
                    min_W2 = self.W2
                    min_b2 = self.b2
                else:
                    self.W1 = min_W1
                    self.b1 = min_b1
                    self.W2 = min_W2
                    self.b2 = min_b2
                    
                    return out
            #"""
        t_stop = time.clock()
        print "Temps total: %.4fs\n" % float(t_stop-t_start)
                
        return out
    
    def compute_predictions(self, data_set):
        return np.transpose( self.fprop(data_set.T)[0] )
        
    # Calcul la pénalité et son gradient
    def compute_penality(self):
        L2 = self.wd * (np.sum(self.W1**2) + np.sum(self.W2**2))
        grad_L2 = self.wd * np.array([2.*np.sum(self.W1), 2.*np.sum(self.W2)])
        
        return L2, grad_L2
    
    def compute_loss(self, features, targets):
        if np.isscalar(targets):
            # On "verticalise" les donnée
            features = features.reshape((features.shape[0], -1))
            
            # Propagation avant
            os = self.fprop( features )[0]
            
            # Perte
            L = -np.log( os[targets] )[0]
        else:
            # Propagation avant
            # ICI, On ne peux pas utiliser directement la sortie de fprop 
            # car elle utilise la version "vectoriel" de softmax
            os = tools.softmaxMat( self.fprop( features.T )[1] ) 
            
            # Perte
            L = np.zeros((features.shape[0],))
            for i in xrange(features.shape[0]):
                L[i] = -np.log( os[targets[i]][i] )

        return L

    # Propagation avant
    # return : os, oa, hs, ha
    def fprop(self, features):

        # Couche cachée
        ha = np.dot(self.W1, features) + self.b1
        hs = np.tanh( ha ) 
        
        # Couche de sortie
        oa = np.dot(self.W2, hs) + self.b2
        os = tools.softmax( oa )
        
        return os, oa, hs, ha
    
    # Propagation arriere
    # return : grad_W1, grad_b1, grad_W2, grad_b2, L
    def bprop(self, features, targets):
        # On "verticalise" les donnée
        features = features.reshape((features.shape[0], -1))
    
        # Propagation avant
        os, oa, hs, ha = self.fprop( features )
        
        # Calcul du gradient
        grad_oa = os - self.__onehot( targets )
  
        grad_W2 = np.dot( grad_oa, np.transpose( hs ) )
        grad_b2 = grad_oa
        
        grad_hs = np.dot( np.transpose( self.W2 ), grad_oa )
        grad_ha = grad_hs * (1. - np.square(np.tanh( ha )))

        grad_W1 = np.dot( grad_ha, np.transpose( features ) )
        grad_b1 = grad_ha
        
        return grad_W1, grad_b1, grad_W2, grad_b2
    
    # Différence finie
    def finiteDifference(self, features, target):
    
        target = int(target)
        features = features.reshape((features.shape[0], -1))
       
        E = 10e-5 # faible variation

        # On fait varier chaque param de E
        grad_W1 = np.zeros((self.n_hid, self.n_in))
        grad_b1 = np.zeros((self.n_hid, 1))
        grad_W2 = np.zeros((self.n_out, self.n_hid))
        grad_b2 = np.zeros((self.n_out, 1))

        # grad W1
        for i in xrange(self.W1.shape[0]):
            for j in xrange(self.W1.shape[1]):
                cache = self.W1[i,j]
                self.W1[i,j] += E
                f1 = self.compute_loss(features, target)
                self.W1[i,j] = cache - E
                f2 = self.compute_loss(features, target)
                grad_W1[i,j] = (f1-f2)/(2.*E)
                self.W1[i,j] = cache
        
        # grad b1
        for i in xrange(self.b1.shape[0]):
            cache = self.b1[i]
            self.b1[i] += E
            f1 = self.compute_loss(features, target)
            self.b1[i] = cache - E
            f2 = self.compute_loss(features, target)
            grad_b1[i] = (f1-f2)/(E)
            self.b1[i] = cache
        
        # grad W2
        for i in xrange(self.W2.shape[0]):
            for j in xrange(self.W2.shape[1]):
                cache = self.W2[i,j]
                self.W2[i,j] += E
                f1 = self.compute_loss(features, target)
                self.W2[i,j] = cache - E
                f2 = self.compute_loss(features, target)
                grad_W2[i,j] = (f1-f2)/(2.*E)
                self.W2[i,j] = cache
        
        # grad b2
        for i in xrange(self.b2.shape[0]):
            cache = self.b2[i]
            self.b2[i] += E
            f1 = self.compute_loss(features, target)
            self.b2[i] = cache - E
            f2 = self.compute_loss(features, target)
            grad_b2[i] = (f1-f2)/(E)
            self.b2[i] = cache
        
        _grad_W1, _grad_b1, _grad_W2, _grad_b2 = self.bprop(features, target)
        print ">> W1\n", grad_W1/_grad_W1
        print ">> b1\n", grad_b1/_grad_b1
        print ">> W2\n", grad_W2/_grad_W2
        print ">> b2\n", grad_b2/_grad_b2
        
        return grad_W1, grad_b1, grad_W2, grad_b2
    
    #################################
    # Private functions
    #################################
    def __initWeights(self, size, n):
        sqrt_inv = 1./np.sqrt(n)
        return np.random.uniform(-sqrt_inv, sqrt_inv, size)
    
    def __onehot(self, y):
        out = np.zeros((self.n_out,)).reshape((self.n_out,-1))
        out[y] = 1
        return out

