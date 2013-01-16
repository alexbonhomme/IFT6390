#! /usr/bin/env python2
# -*- coding: utf-8 -*-
###################################
#
# Reconnaissance par "Eigenfaces"
#
###################################
import tools
import sys, os
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
				 trainFile="", testFile="", debug_mode=True, categorie="ORL", nbExemples=5, stock=0, curv=0):
		# KNN
		self.K = K

		# Parzen
		self.Theta = Theta

		# NNET
		self.batch_size = batch_size
		self.n_epoch = n_epoch
		self.n_hidden = n_hidden
		self.lr = lr
		self.wd = wd

		# categorie  ("LFW", "ORL", "BOTH")
		self.categorie = categorie
		self.nbExemples = nbExemples
		if self.categorie not in ["LFW","ORL"]:
			log.error("La  categorie d'images étudiées doit être LFW ou ORL")
			exit(-1)
		if self.nbExemples < 0:
			log.error("Le nombre d'exemples envisagés doit être positif")
			exit(-1)
		if self.nbExemples >= 10 and self.categorie == "ORL":
			log.error("Le nombre d'entrees de l'ensemble d'entrainement doit etre constitue de moins de 10 exemples par classes pour le domaine ORL")
			exit(-1)

		# stock & courbes
		self.stock=stock
		if self.stock not in [0,1]:
			self.stock=0
		self.curv=curv
		if self.curv not in [0,1]:
			self.curv=0

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
                buf.insert_at_cursor(text + "\n")
                textview.scroll_mark_onscreen(buf.get_insert())
            else:
                log.info(text)
        
        # liste des resultats utilises pour les courbes
        listeRes=[]

        # creation des trainFile et testFile
        print_output("Construction des fichiers d'entrainement...")
        tools.constructLfwNamesCurrent( self.nbExemples )
        #TODO ca ne sert plus a rien finalement
        ( nbClassesLFW, nbClassesORL ) = tools.trainAndTestConstruction( self.nbExemples )

        # Chargement des données
        dataTrain, dataTrainIndices, nClass = tools.loadImageData( "train", self.categorie)
        
        # tranformation pca
        print_output("Calcul des vecteurs propres...")
        pca_model = PCA( dataTrain )
        pca_model.transform() # on transforme les donné dans un le "eigen space"

        ##### Recherche pas KNN
        if algo == "KNN":
            print_output("Début de l'algorithme des K plus proches voisins...")
            
            # On build le model pour recherche par KNN
            knn_model = KNN( pca_model.getWeightsVectors(), dataTrainIndices, nClass, self.K )
            
            # On build le model pour Parzen
            parzen_model = ParzenWindows( pca_model.getWeightsVectors(), dataTrainIndices, nClass, self.Theta )

            ## TEST ###########################
            #TODO Toute cette partie est a revoir pour sortir des graphes
            # de train, validation, test
            dataTest, dataTestIndices, nClass = tools.loadImageData( "test", self.categorie )

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

            ### Resultats
            listeRes.append(res)

            #### Stockage
            if self.stock == 1 :
                fichier = open("curvAccuracyKnn"+self.categorie,"a")
                fichier.write(str(self.nbExemples)+" "+str(res)+" "+str(self.K)+"\n")
                fichier.close()
        
        #### Recherche pas NNET
        elif algo == "NNET":
			print_output("Début de l'algorithme du Perceptron multicouche...")
			
			# parametre, donnees, etc...
			dataTrain = pca_model.getWeightsVectors()
			dataTrainTargets = (dataTrainIndices - 1).reshape(dataTrainIndices.shape[0], -1)
			#! contrairement au KNN le NNET prends les vecteurs de features en ligne et non pas en colonne
			train_set = np.concatenate((dataTrain.T, dataTrainTargets), axis=1)

			# On build et on entraine le model pour recherche par KNN
			nnet_model = NeuralNetwork( dataTrain.shape[0], self.n_hidden, nClass, self.lr, self.wd )
			nnet_model.train( train_set, self.n_epoch, self.batch_size )

			## TEST ###########################
			#TODO Toute cette partie est a revoir pour sortir des graphes
			# de train, validation, test
			dataTest, dataTestIndices, nClass = tools.loadImageData( "test", self.categorie )

			# compteurs de bons résultats   
			nbGoodResult = 0

			for i in range(0, int( dataTest.shape[1] )):
				#TODO faire ne projection matriciel
				proj = pca_model.getProjection( dataTest[:,i] )
				proj = proj.reshape(1, proj.shape[0])

				#
				resultNNET = np.argmax(nnet_model.compute_predictions( proj ), axis=1)[0] + 1
				if(resultNNET == dataTestIndices[i]):
					nbGoodResult += 1
				out_str = "Result: "+ str( resultNNET ) + " | Expected: "+ str( dataTestIndices[i] ) +"\n" # +1 car l'index de la matrice commence a 0
				print_output(out_str)

			res = (float(nbGoodResult) / float(dataTest.shape[1])) * 100.
			out_str = "\nAccuracy : %.3f" % res + "%\n"
			print_output(out_str)
                        
                        #### Trace de courbes
                        if self.stock == 1 :
                            fichier = open("curvAccuracy"+self.categorie,"a")
                            fichier.write(str(self.nbExemples)+" "+str(res)+"\n")
                            fichier.close()

#### FIN CLASSE MAIN ####################################

# Si le script est appelé directement on execute ce code
if __name__ == "__main__":
    import argparse

    # Options du script
    parser = argparse.ArgumentParser(description='Facial recognition')
    
    parser.set_defaults(verbose=True)
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", help="print status messages to stdout")
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", help="don't print status messages to stdout")

    parser.add_argument("--trainfile", 
                      dest="train_filename",
                      help="train FILE", 
                      default="./Databases/train1.txt",
                      metavar="FILE")
    
    parser.add_argument("--testfile", 
                      dest="test_filename",
                      help="test FILE", 
                      default="./Databases/test4.txt",
                      metavar="FILE")

    parser.add_argument("--nExamples", 
                      dest="nbExemples",
                      help="Number of exemples per Classe for training",
                      type=int,  
                      default=7)

    parser.add_argument("--categorie", 
                      dest="categorie",
                      help="LFW or ORL", 
                      default="ORL")

    parser.add_argument("--stock", 
                      dest="stock",
                      help="1 si l'on veut stocker les données, 0 sinon", 
                      type=int,
                      default=0)

    parser.add_argument("--curv", 
                      dest="curv",
                      help="1 si l'on veut tracer les courbes, 0 sinon", 
                      type=int,
                      default=0)
    
    # sous parseur pour knn et nnet
    subparsers = parser.add_subparsers(title='Algorythms',
                                       description='Type of algorythm tou can use.',
                                       dest='algo_type')
    # KNN parser
    parser_knn = subparsers.add_parser('knn',
                                       help='use the k nearest neighbors algorythm',
                                       description='K Nearest Neighbors algorythm')
    parser_knn.add_argument("-k",
                            dest="k", 
                            type=int,
                            default=1,
                            help="number of neighbors")
    
    parser_knn.add_argument("-t", "--theta",
                            dest="theta",
                            type=float,
                            default=0.5,
                            help="gaussian kernel size")
    
    # NNET parser
    parser_nnet = subparsers.add_parser('nnet',
                                        help='use a neural network algorythm',
                                        description='Neural Network algorythm')
    parser_nnet.add_argument("--epoch",
                            dest="n_epoch", 
                            type=int,
                            default=100,
                            help="number of train epoch",
                            metavar="N")
    
    parser_nnet.add_argument("--hid",
                            dest="n_hidden", 
                            type=int,
                            default=10,
                            help="number of hidden neurons",
                            metavar="N")
    
    parser_nnet.add_argument("--batch",
                            dest="batch_size", 
                            type=int,
                            default=1,
                            help="size of the batch",
                            metavar="N")
    
    parser_nnet.add_argument("--lr",
                            dest="lr", 
                            type=float,
                            default=0.001,
                            help="learning rate",
                            metavar="NU")

    parser_nnet.add_argument("--wd", "--L2",
                            dest="wd", 
                            type=float,
                            default=0.0,
                            help="weight decay (L2 penality)",
                            metavar="ALPHA")

    # on parse la commande
    args = parser.parse_args()
    
    # On ne traite que les options connu (parsées dans opts)
    trainFile = args.train_filename
    testFile = args.test_filename
    debug_mode = args.verbose
    algo_type = args.algo_type.upper()
    categorie = args.categorie
    nbExemples = args.nbExemples
    stock = args.stock
    curv = args.curv

    #### Début du programme
    if algo_type == "KNN":
        K = args.k
        Theta = args.theta
        xVector = [ nbExemples ]
        yVector = []
        if curv == 1 :
            if categorie == "ORL" :
                tools.completion( xVector, 8)
            elif categorie == "LFW" :
                xVector = [ nbExemples ]
                tools.completion( xVector, 10)
        faceReco = Main( K=K, Theta=Theta, trainFile=trainFile, testFile=testFile, categorie=categorie, stock=stock, curv=curv, nbExemples=nbExemples, debug_mode=debug_mode)
        for n in xVector:
            faceReco.nbExemples = n
            listeRes=faceReco.main( algo=algo_type )
            yVector.append( listeRes )
        
    
    elif algo_type == "NNET":
        n_epoch = args.n_epoch
        n_hidden = args.n_hidden
        batch = args.batch_size
        lr = args.lr
        wd = args.wd
        faceReco = Main( batch_size=batch, n_epoch=n_epoch, n_hidden=n_hidden, lr=lr, wd=wd, 
                         trainFile=trainFile, testFile=testFile, debug_mode=debug_mode, categorie=categorie, stock=stock, curv=curv, nbExemples=nbExemples)
        faceReco.main( algo=algo_type )

