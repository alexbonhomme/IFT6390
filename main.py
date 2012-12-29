#! /usr/bin/env python2
# -*- coding: utf-8 -*-
###################################
#
# Reconnaissance par "Eigenfaces"
#
###################################
import tools
import sys
import logging as log
import numpy as np

from pca import PCA
from knn import KNN
from parzen import ParzenWindows
from nnet import NeuralNetwork

#### DEBUT CLASSE MAIN ####################################
class Main (object):

    def __init__(self, K=1, Theta=0.5, 
                 batch_size=1, n_epoch=100, n_hidden=10, lr=0.001, wd=0.,
                 trainFile="", testFile="", debug_mode=True):
        # KNN
        self.K = K
        
        # Parzen
        self.Theta = Theta
        
        # NNET
        self.batch_size = int(batch_size)
        self.n_epoch = int(n_epoch)
        self.n_hidden = int(n_hidden)
        self.lr = float(lr)
        self.wd = float(wd)
        
        # fichiers de train
        if trainFile == "":
            self.trainFile = "./Databases/train1.txt"
        else:
            self.trainFile = trainFile
        
        # ... de test
        if testFile == "":
            self.testFile = "./Databases/test4.txt"
        else:
            self.testFile = testFile
        
        # logger pour debbug
        if debug_mode:
            log.basicConfig(stream=sys.stderr, level=log.DEBUG)
        else:
            log.basicConfig(stream=sys.stderr, level=log.INFO)
    
    #TODO trouver un nom plus subtile..?
    def main(self, algo="KNN", textview=None):
        
        # Remplace "print"
        def print_output(text):
            if textview != None:
                buf = textview.get_buffer()
                buf.insert_at_cursor(text)
                textview.scroll_mark_onscreen(buf.get_insert())
            else:
                log.info(text)
    
        # Chargement des données
        dataTrain, dataTrainIndices = tools.loadImageData( self.trainFile )
    
        # tranformation pca
        pca_model = PCA( dataTrain )
        pca_model.transform() # on transforme les donné dans un le "eigen space"
        
        # Calcul du nombre de class
        #TODO Devrait peut etre etre inclu dans le fichier de test... maybe
       	nClass = tools.countClass( dataTrainIndices )
        
        ##### Recherche pas KNN
        if algo == "KNN":
            
            # On build le model pour recherche par KNN
            knn_model = KNN( pca_model.getWeightsVectors(), dataTrainIndices, nClass, self.K )
            
            # On build le model pour Parzen
            parzen_model = ParzenWindows( pca_model.getWeightsVectors(), dataTrainIndices, nClass, self.Theta )

            ## TEST ###########################
            #TODO Toute cette partie est a revoir pour sortir des graphes
            # de train, validation, test
            dataTest, dataTestIndices = tools.loadImageData( self.testFile )

            # compteurs de bons résultats   
            nbGoodResult = 0
            nbGoodResult2 = 0 
            nbGoodResult3 = 0

            for i in range(0, int( dataTest.shape[1] )):
                #TODO faire ne projection matriciel
                proj = pca_model.getProjection( dataTest[:,i] )

                # k = 1, pour réference
                # on force k
                knn_model.setK( 1 )
                result1NN = knn_model.compute_predictions( proj )
                if(result1NN == dataTestIndices[i]):
                    nbGoodResult += 1

                # k = n
                # replace k a ca position initial
                knn_model.setK( self.K )
                resultKNN = knn_model.compute_predictions( proj )
                if(resultKNN == dataTestIndices[i]):
                    nbGoodResult2 += 1

                #
                resultParzen = parzen_model.compute_predictions( proj )
                if(resultParzen == dataTestIndices[i]):
                    nbGoodResult3 += 1

                out_str = "Classic method: "+ str( result1NN ) +" | KNN method: "+ str( resultKNN ) +" | KNN+Parzen method: "+ str( resultParzen ) +" | Expected: "+ str( dataTestIndices[i] ) +"\n" # +1 car l'index de la matrice commence a 0
                print_output(out_str)

            res = (float(nbGoodResult) / float(dataTest.shape[1])) * 100.
            out_str = "\nAccuracy with classic method: %.3f" % res + "%\n"
            res = (nbGoodResult2 / float(dataTest.shape[1])) * 100.
            out_str += "Accuracy with KNN method (k="+ str( self.K ) +"): %.3f" % res + "%\n"
            res = (nbGoodResult3 / float(dataTest.shape[1])) * 100.
            out_str += "Accuracy with KNN + Parzen window method (theta="+ str( self.Theta ) +"): %.3f" % res + "%\n"
            print_output(out_str)
        
        #### Recherche pas NNET
        elif algo == "NNET":
            # parametre, donnees, etc...
            
            dataTrain = pca_model.getWeightsVectors()
            dataTrainTargets = (dataTrainIndices - 1).reshape(dataTrainIndices.shape[0], -1)
            train_set = np.concatenate((dataTrain, dataTrainTargets), axis=1)

            # On build et on entraine le model pour recherche par KNN
            nnet_model = NeuralNetwork( dataTrain.shape[0], self.n_hidden, nClass, self.lr, self.wd )
            nnet_model.train( train_set, self.n_epoch, self.batch_size )
            

#### FIN CLASSE MAIN ####################################

# Si le script est appelé directement on execute se code
if __name__ == "__main__":
    from optparse import OptionParser

    # Options du script
    parser = OptionParser()
    parser.add_option("--trainfile", 
                      dest="train_filename",
                      help="train FILE", 
                      default="./Databases/train1.txt",
                      metavar="FILE")
    
    parser.add_option("--testfile", 
                      dest="test_filename",
                      help="test FILE", 
                      default="./Databases/test4.txt",
                      metavar="FILE")
    
    parser.add_option("-k",
                      dest="k", 
                      type="int",
                      default=1,
                      help="number of neighbors")
    
    parser.add_option("-t", "--theta",
                      dest="theta",
                      type="float",
                      default=0.5,
                      help="gaussian kernel size")         

    parser.set_defaults(verbose=True)
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="print status messages to stdout")
    parser.add_option("-q", "--quiet", action="store_false", dest="verbose", help="don't print status messages to stdout")

    parser.add_option("--type",
                      dest="algo_type", 
                      type="string",
                      default="KNN",
                      help="algorythm to use")

    (opts, args) = parser.parse_args()
    
    # On ne traite que les options connu (parsées dans opts)
    trainFile = opts.train_filename
    testFile = opts.test_filename
    K = opts.k
    Theta = opts.theta
    debug_mode = opts.verbose
    algo_type = opts.algo_type.upper()

    #### Début du programme
    faceReco = Main( K=K, Theta=Theta, trainFile=trainFile, testFile=testFile, debug_mode=debug_mode )
    faceReco.main( algo=algo_type )

