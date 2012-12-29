# -*- coding: utf-8 -*-
import numpy as np
import tools
import time
import logging as log

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
        
        # Parametres theta
        # theta regroupe W1, b1, W2, b2
        self.W1 = self.__initWeights((self.n_hid, self.n_in), self.n_in)
        self.b1 = np.zeros((self.n_hid, 1))
        self.W2 = self.__initWeights((self.n_out, self.n_hid), self.n_hid)
        self.b2 = np.zeros((self.n_out, 1))

    def train(self, train_set, n_epoch, batch_size=1, valid_set=None, test_set=None, EarlyStopping=False, filename=""):

        # Ensemble d'entrainement
        features = train_set[:,:-1]
        targets = train_set[:,-1]
        train_size = features.shape[0]
        
        R = 0.
        L_total = 0.
        out = []    # Vecteur de sortie pour affichage
        
        # Ensemble de validation
        if valid_set != None:
            valid_features = valid_set[:,:-1]
            valid_targets = valid_set[:,-1]
            
        valid_out = []  # Vecteur de sortie pour affichage
        
        # Ensemble de test
        if test_set != None:
            test_features = test_set[:,:-1]
            test_targets = test_set[:,-1]
            
        test_out = []  # Vecteur de sortie pour affichage
        
        # Arret prématuré
        if EarlyStopping:
            prev_R = np.inf
            min_W1 = np.zeros((self.n_hid, self.n_in))
            min_b1 = np.zeros((self.n_hid, 1))
            min_W2 = np.zeros((self.n_out, self.n_hid))
            min_b2 = np.zeros((self.n_out, 1))
            
        # Ouverture du fichier de sortie
        if filename != "":
            f_out = open(filename, "w")

        log.info("Début de l'entrainement par descente de gradient type mini-batch=" + str(batch_size))
        log.debug("Nombre de neurones: " + str(self.n_hid))
        log.debug("Taux d'apprentissage: " + str(self.lr))
        log.debug("Pénalité L2: " + str(self.wd))
        log.debug("Nombre d'époque: " + str(n_epoch))
        log.debug("Taille du batch: " + str(batch_size))

        t_start = time.clock()
        for k in xrange(n_epoch):
            t1 = time.clock()

            # Non optimal <_<
            for z in xrange(train_size/batch_size):
                
                # Calcul de l'index
                ct = z * batch_size

                # back propagation sur le "mini-batch"             
                grad_sumW1, grad_sumb1, grad_sumW2, grad_sumb2 = self.bprop(features[ct:ct+batch_size], targets[ct:ct+batch_size])
                
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

            """
            if self.wd > 0:
                penality = self.compute_penality()[0]
            else:
                penality = 0.
            """
 
            t2 = time.clock()
            
            # Calcul des pertes/erreurs
            R = np.sum(self.compute_loss(features, targets))
            L_total += R
            R = (R/features.shape[0])# + penality
            Err = self.compute_classif_error(train_set)
            out.append([R, Err])
            
            if valid_set != None:
                R_valid = np.mean(self.compute_loss(valid_features, valid_targets))# + penality
                Err_valid = self.compute_classif_error(valid_set)
                valid_out.append([R_valid, Err_valid])
            
            if test_set != None:
                R_test = np.mean(self.compute_loss(test_features, test_targets))# + penality
                Err_test = self.compute_classif_error(test_set)
                test_out.append([R_test, Err_test])
            
            # Affichage
            log.info("Epoch %4d" % k + " - Temps de calcul = %.3fs" % float(t2-t1) + " - Cout optimisé total = %.3f" % L_total) 
            log.info("Train:\tErreur de classification = %3.1f" % (Err*100.) + "%" + " - Cout moyen = %.4f" % R) 
            #print "Valid:\tErreur de classification = %3.1f" % (Err_valid*100.), "%" + " - Cout moyen = %.4f" % R_valid
            #print "Test: \tErreur de classification = %3.1f" % (Err_test*100.),  "%" + " - Cout moyen = %.4f" % R_test
            
            # Sortie dans un fichier
            if filename != "":
                f_out.write("%f, %f, %f, %f, %f, %f\n" % (Err, R, Err_valid, R_valid, Err_test, R_test))

            # On mémorise les minimums
            #"""
            if EarlyStopping:
                # Si la perte précédente est supperieux on continue
                if prev_R + 10e-6 > R_valid:  # Le epsilon est sensée evité les minima locaux... sensée...
                    prev_R = R_valid
                    min_W1 = self.W1
                    min_b1 = self.b1
                    min_W2 = self.W2
                    min_b2 = self.b2
                # Sinon on stop le training
                else:
                    self.W1 = min_W1
                    self.b1 = min_b1
                    self.W2 = min_W2
                    self.b2 = min_b2
                    
                    v_out = [out]
                    if valid_set != None:
                        v_out.append(valid_out)
                    
                    if test_set != None:
                        v_out.append(test_out)
                    
                    return v_out
            #"""
        # Fermeture fichier
        if filename != "":
            f_out.close()

        t_stop = time.clock()
        log.info("Temps total: %.4fs\n" % float(t_stop-t_start)) 
        
        # On construit une list contenant les différent courbes (train, valid, test)
        v_out = [out]
        if valid_set != None:
            v_out.append(valid_out)
        
        if test_set != None:
            v_out.append(test_out)
        
        return v_out
    
    # Calcul les predictions
    def compute_predictions(self, data_set):
        return np.transpose( self.fprop(data_set)[0] )
        
    # Calcul la pénalité et son gradient
    def compute_penality(self):
        L2 = self.wd * (np.sum(self.W1**2) + np.sum(self.W2**2))
        grad_L2 = self.wd * np.array([2.*np.sum(self.W1), 2.*np.sum(self.W2)])
        
        return L2, grad_L2
    
    # Calcul la perte
    def compute_loss(self, features, targets):
        # Propagation avant
        os = self.fprop( features )[0]
        
        # Perte
        L = np.zeros((features.shape[0],))
        for i in xrange(features.shape[0]):
            L[i] = -np.log( os[targets[i]][i] )

        return L
    
    def compute_classif_error(self, data_set):
        features = data_set[:,:-1]
        targets = data_set[:,-1].astype(int)
        
        # On calcule les prédiction
        pred = np.argmax(self.compute_predictions(features), axis=1)
        
        # On compare avec les cibles
        comp = np.logical_and(True, pred == targets)

        return 1. - np.sum(comp)/float(targets.shape[0])
    
    # Propagation avant
    # return : os, oa, hs, ha
    def fprop(self, features):
        
        # Couche cachée
        ha = np.dot(self.W1, features.T) + self.b1
        hs = np.tanh( ha ) 
        
        # Couche de sortie
        oa = np.dot(self.W2, hs) + self.b2
        os = tools.softmaxMat( oa )

        return os, oa, hs, ha
    
    # Propagation arriere
    # return : grad_W1, grad_b1, grad_W2, grad_b2, L
    def bprop(self, features, targets):

        # Propagation avant
        os, oa, hs, ha = self.fprop( features )
        
        # Calcul du gradient
        grad_oa = os - self.__onehotMat( targets )
  
        grad_W2 = np.dot( grad_oa, np.transpose( hs ) )
        grad_b2 = np.sum(grad_oa, axis=1).reshape((grad_oa.shape[0], -1))
        
        grad_hs = np.dot( np.transpose( self.W2 ), grad_oa )
        grad_ha = grad_hs * (1. - np.square(np.tanh( ha )))

        grad_W1 = np.dot( grad_ha, features )
        grad_b1 = np.sum(grad_ha, axis=1).reshape((grad_ha.shape[0], -1))

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
    
    def __onehotMat(self, Y):
        out = np.zeros((self.n_out, Y.shape[0]))
        for i in xrange(Y.shape[0]):
            out[Y[i]][i] = 1

        return out

