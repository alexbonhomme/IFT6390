#,! /usr/bin/env python2
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
import time

# Répertoire d'enregistrement des images
IMG_DIR = "graphes/"

#### DEBUT CLASSE MAIN ####################################
class Main (object):

    def __init__(self, K=1, Theta=0.5, 
                 batch_size=1, n_epoch=100, n_hidden=10, lr=0.001, wd=0.,
                 trainFile="", testFile="", debug_mode=True, categorie="ORL", nbExemples=5, stock=0, curv="0", pourcentageTrain=0.6, validation=True):
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

        # validation
        self.validation = validation
        
        # categorie  ("LFW", "ORL", "BOTH"), nbExemples
        self.categorie = categorie
        self.nbExemples = nbExemples
        if self.categorie not in ["LFW", "ORL"]:
            log.error("La  categorie d'images étudiees doit être LFW ou ORL")
        if self.nbExemples < 4:
             log.error("Le nombre d'exemples envisages doit >= 4")
        #if self.nbExemples>=400 and self.categorie=="LFW":
            #log.error("Le nombre d'entrees de l'ensemble d'entrainement doit etre constitue de moins de 400 exemples par classes pour le domaine LFW")
        if self.nbExemples > 10 and self.categorie == "ORL":
            log.error("Le nombre d'entrees pour l'etude doit etre constitue de moins de 10 exemples par classes pour le domaine ORL")
        self.pourcentageTrain = pourcentageTrain
        if self.pourcentageTrain >= 1.0 or self.pourcentageTrain <= 0 :
            log.error("Le pourcentage doit etre dans ]0;1[")

        # stock & courbes
        self.stock = stock
        if self.stock not in [0,1]:
            self.stock = 0
        self.curv = curv
        if self.curv == "n" :
            self.curv = 1
        if self.curv not in ["k", "n", "0"]:
            self.curv = "0"

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
        
        
        # liste des types de set
        if self.validation == 1:
            listeTypesSet = ["train", "validation", "test"]
        else:
            listeTypesSet = ["train", "test"]

        # liste des resultats utilises pour les courbes
        listeRes=[]

        # creation des trainFile et testFile
        log.debug("Construction des fichiers d'entrainement")
        tools.constructLfwNamesCurrent( self.nbExemples )   

        #TODO ca ne sert plus a rien finalement
        ( nbClassesLFW, nbClassesORL ) = tools.trainAndTestConstruction( self.pourcentageTrain, self.nbExemples )

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
            for trainTest in listeTypesSet:
                if trainTest == "train":
                    dataTest, dataTestIndices = dataTrain, dataTrainIndices
                else :
                    ### si l'on n'effectue pas de validation on concatene les entrees de test et de validation initiales pour obtenir le test
                    #if "validation" not in listeTypesSet:
                        #dataTestInitial, dataTestInitialIndices, nClass = tools.loadImageData( "test", self.categorie )
                        #dataValidation, dataValidationIndices, nClass = tools.loadImageData( "validation", self.categorie )
                        #dataTest = np.zeros(dataTestInitial.size + dataValidation.size)
                        #dataTestIndices = np.zeros( dataTest.size )
                        #dataTest[ : dataTestInitial.size], dataTestIndices[ : dataTestInitial.size] = dataTestInitial, dataTestInitialIndices
                        #dataTest[dataTestInitial.size : ], dataTestIndices[dataTestInitial.size : ] = dataValidation, dataValidationIndices
                        
                        
                    #else:
                        dataTest, dataTestIndices, nClass = tools.loadImageData( trainTest, self.categorie )
                print_output("Projection des données de test...")
                dataTest_proj = pca_model.getProjection( dataTest )
                

            	# compteurs de bons résultats   
                nbGoodResult = 0
                nbGoodResult2 = 0 
                nbGoodResult3 = 0

                t_start = time.clock()
                for i in range(0, int( dataTest.shape[1] )):

					# k = 1, pour réference
					# on force k
                    knn_model.setK( 1 )
                    result1NN = knn_model.compute_predictions( dataTest_proj[:,i] )
                    if(result1NN == dataTestIndices[i]):
                        nbGoodResult += 1

		            # k = n
		            # replace k a ca position initial
                    knn_model.setK( self.K )
                    resultKNN = knn_model.compute_predictions( dataTest_proj[:,i] )
                    if(resultKNN == dataTestIndices[i]):
                        nbGoodResult2 += 1

                
                    resultParzen = parzen_model.compute_predictions( dataTest_proj[:,i] )
                    if(resultParzen == dataTestIndices[i]):
                        nbGoodResult3 += 1
     
                    out_str = "Classic method: "+ str( result1NN ) +" | KNN method: "+ str( resultKNN ) +" | KNN+Parzen method: "+ str( resultParzen ) +" | Expected: "+ str( dataTestIndices[i] ) +"\n" # +1 car l'index de la matrice commence a 0
                    print_output(out_str)

                resClassic = (float(nbGoodResult) / float(dataTest.shape[1])) * 100.
                out_str = "\nAccuracy with classic method: %.3f" % resClassic + "%\n"
                resKNN = (nbGoodResult2 / float(dataTest.shape[1])) * 100.
                out_str += "Accuracy with KNN method (k="+ str( self.K ) +"): %.3f" % resKNN + "%\n"
                res = (nbGoodResult3 / float(dataTest.shape[1])) * 100.
                out_str += "Accuracy with KNN + Parzen window method (theta="+ str( self.Theta ) +"): %.3f" % res + "%\n"
                print_output(out_str)
                
                t_stop = time.clock()
                log.info("Temps total: %.4fs\n" % float(t_stop-t_start)) 

				#### recupere les valeurs finale de l'erreur
                listeRes.append( 100 - resClassic )
                listeRes.append( 100 - resKNN )
                listeRes.append( 100 - res )

            
        
        #### Recherche pas NNET
        elif algo == "NNET":
			print_output("Début de l'algorithme du Perceptron multicouche...")
			
			# parametre, donnees, etc...
			dataTrain = pca_model.getWeightsVectors()
			dataTrainTargets = (dataTrainIndices - 1).reshape(dataTrainIndices.shape[0], -1)
			#! contrairement au KNN le NNET prends les vecteurs de features en ligne et non pas en colonne
			train_set = np.concatenate((dataTrain.T, dataTrainTargets), axis=1)

                        # recuperation des données de validation
			dataValidation, dataValidationIndices, nClass = tools.loadImageData( "validation", self.categorie )
			print_output("Projection des données de validation...")
			dataValidation_proj = pca_model.getProjection( dataValidation )
			dataValidationTargets = (dataValidationIndices - 1).reshape(dataValidationIndices.shape[0], -1)
			validation_set = np.concatenate((dataValidation_proj.T, dataValidationTargets), axis=1)

			# recuperation des données de test
			dataTest, dataTestIndices, nClass = tools.loadImageData( "test", self.categorie )
			print_output("Projection des données de test...")
			dataTest_proj = pca_model.getProjection( dataTest )
			dataTestTargets = (dataTestIndices - 1).reshape(dataTestIndices.shape[0], -1)
			test_set = np.concatenate((dataTest_proj.T, dataTestTargets), axis=1)

			# On build et on entraine le model pour recherche par KNN
			nnet_model = NeuralNetwork( dataTrain.shape[0], self.n_hidden, nClass, self.lr, self.wd )
                        if self.validation == 1:
                            train_out, valid_out, test_out = nnet_model.train( train_set, self.n_epoch, self.batch_size, valid_set=validation_set, test_set=test_set)
                        else :
                            train_out, test_out = nnet_model.train( train_set, self.n_epoch, self.batch_size, test_set=test_set)

			# affichage des courbes d'entrainement
			x = []
			y = []
			y_err = []
			color = []
			legend = []
			legend_err = []
			filename = IMG_DIR + "Risque__Epoch_"+ str(self.n_epoch) +"_Hidden_"+ str(self.n_hidden) +"_Lr_"+ str(self.lr) +"_L2_"+ str(self.wd) +"_"
			filename_err = IMG_DIR + "Erreur_classification__Epoch_"+ str(self.n_epoch) +"_Hidden_"+ str(self.n_hidden) +"_Lr_"+ str(self.lr) +"_L2_"+ str(self.wd) +"_"

			train_out = np.array(train_out)
			x.append(np.array(xrange(train_out.shape[0])))
		
			# parametres courbes train
			color.append('g-')
			legend.append("R Train")
			filename += "_Train"
			y.append(train_out[:,0])
			y_err.append(train_out[:,1])
			legend_err.append("Err Train")
			filename_err += "_Train"

                        # parametre courbes validation
                        if self.validation == 1:
                            valid_out = np.array(valid_out)
                            x.append(np.array(xrange(valid_out.shape[0])))
                            y.append(valid_out[:,0])
                            y_err.append(valid_out[:,1])
                            color.append('b-')
                            legend.append("R Validation")
                            legend_err.append("Err Validation")
                            filename += "_Validation"
                            filename_err += "_Validation"

			# parametre courbes test
			test_out = np.array(test_out)
			x.append(np.array(xrange(test_out.shape[0])))
			y.append(test_out[:,0])
			y_err.append(test_out[:,1])
			color.append('r-')
			legend.append("R Test")
			legend_err.append("Err Test")
			filename += "_Test"
			filename_err += "_Test"

			
			# affichage
			title = u"\nEpoque: " + str(self.n_epoch) + " - Taille du batch: " + str(self.batch_size) + u" - Neurones cachés: " + str(self.n_hidden) + "\nL2: " + str(self.wd) + " - Taux d'apprentissage: " + str(self.lr) + " - Catégorie: " + str(self.categorie)
			tools.drawCurves(x, y, color, legend, bDisplay=True, filename=filename, title=title, xlabel="Epoque", ylabel=u"Risque régularisé")
			tools.drawCurves(x, y_err, color, legend_err, bDisplay=True, filename=filename_err, title=title, xlabel="Epoque", ylabel="Erreur classification")

                         #### construction fichier pour courbes ameliorees
                        if self.stock == 1 :
                            fichier = open("curvErrorNNet"+''.join( ''.join( title.split(' ') ).split('\n') ),"w")
                            fichier.write("#epoch errorTrain errorValidation errorTest\n")
                            
                            if len(x) == 3:
                            	for j in range(len( x[0] )):
                            	    fichier.write(str( x[0][j] )+" "+str( y[0][j] )+" "+str( y[1][j] )+" "+str( y[2][j] )+"\n")

                            fichier.close()

                        
			"""
			/!\ Cette partie n'est plus utile car effectué dans le nnet durant le train
			
			## TEST ###########################
			#TODO Toute cette partie est a revoir pour sortir des graphes
			# de train, validation, test
			
			# compteurs de bons résultats   
			nbGoodResult = 0

			for i in range(0, int( dataTest.shape[1] )):

				#
				resultNNET = np.argmax(nnet_model.compute_predictions( dataTest_proj[:,i] ), axis=1)[0]
				if(resultNNET == dataTestTargets[i]):
					nbGoodResult += 1
				out_str = "Result: "+ str( resultNNET ) + " | Expected: "+ str( dataTestTargets[i] ) +"\n" # +1 car l'index de la matrice commence a 0
				print_output(out_str)

			res = (float(nbGoodResult) / float(dataTest.shape[1])) * 100.
			out_str = "\nAccuracy : %.3f" % res + "%\n"
			print_output(out_str)
            """            
   
        return listeRes

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
                      help="Number of exemples per Classe for the study",
                      type=int,  
                      default=10)

    parser.add_argument("--pTrain", 
                      dest="pourcentageTrain",
                      help="]0;1[, percent of training exemples",
                      type=float,  
                      default=0.6)

    parser.add_argument("--valid", 
                      dest="validation",
                      help="consideration of a validation set",
                      type=bool,  
                      default=False)

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
                      help="k si l'on veut tracer les courbes en fonction de k, n si l'on veut tracer les courbes en fonction du nombre d'exemples Train", 
                      type=str,
                      default="0")
    
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
    pourcentageTrain = args.pourcentageTrain
    validation = bool(args.validation)


    #### Début du programme
    if algo_type == "KNN":
        K = args.k
        Theta = args.theta
        
        #### initialisation des abscisses et ordonnees #
        if curv == "n":
            xVector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if curv == "k":
            xVector = [int( 0.1*nbExemples ), int( 0.2*nbExemples ), int( 0.3*nbExemples ), int( 0.4*nbExemples ), int( 0.5*nbExemples ), int( 0.6*nbExemples ), int( 0.7*nbExemples ), int( 0.8*nbExemples )]
        yVectorClassicTrain = []
        yVectorClassicValidation = []
        yVectorClassicTest = []
        yVectorKNNTrain = []
        yVectorKNNValidation = []
        yVectorKNNTest = []
        yVectorParzenTrain = []
        yVectorParzenValidation = []
        yVectorParzenTest = []

        #### construction main #
        faceReco = Main( K=K, Theta=Theta, trainFile=trainFile, testFile=testFile, validation=validation, categorie=categorie, stock=stock, curv=curv, pourcentageTrain=pourcentageTrain, nbExemples=nbExemples, debug_mode=debug_mode)
        if curv != "0" :
            indice = 0
            
            #### constru,,ction ordonnees et recuperation res
            for p in xVector:
                if curv == "n" :
                    faceReco.pourcentageTrain = p
                if curv == "k" :
                    faceReco.K = p
                listeRes = faceReco.main( algo=algo_type )
                yVectorClassicTrain.append( listeRes[0] )
                yVectorKNNTrain.append( listeRes[1] )
                yVectorParzenTrain.append( listeRes[2] )

                 ### Si validation
                if validation == 1 :
                    yVectorClassicValidation.append( listeRes[3] )
                    yVectorKNNValidation.append( listeRes[4] )
                    yVectorParzenValidation.append( listeRes[5] )
                    yVectorClassicTest.append( listeRes[6] )
                    yVectorKNNTest.append( listeRes[7] )
                    yVectorParzenTest.append( listeRes[8] )
                else : 
                    yVectorClassicTest.append( listeRes[3] )
                    yVectorKNNTest.append( listeRes[4] )
                    yVectorParzenTest.append( listeRes[5] )
                if curv == "n" :
                    xVector[indice] = np.max(( int(p * nbExemples), 1 ))
                    indice += 1

            #### construction des conteneurs de nos 6 listes d'abscisses et ordonnees
            x=[]
            y=[] 
            if curv == "n":
                nbVectors = 6
                colorVect = ["g--", "b--", "r--", "g", "b", "r"]
                legendVect = ["k=1 on train ", "k="+str(K)+" on train", "Parzen theta="+str(Theta)+" on train", "k=1 on test ", "k="+str(K)+" on test", "Parzen theta="+str(Theta)+" on test"]
            if curv == "k":
                nbVectors = 4
                colorVect = ["g--", "r--", "g", "r"]
                legendVect = ["Train", "+ Parzen theta="+str(Theta)+" Train", "Test", "+ Parzen theta="+str(Theta)+" Test"]
               
            if curv == "n" : 
                y.append(yVectorClassicTrain)
            y.append(yVectorKNNTrain)
            y.append(yVectorParzenTrain)
            if curv == "n" : 
                y.append(yVectorClassicTest)
            y.append(yVectorKNNTest)
            y.append(yVectorParzenTest)


            ### si validation
            if validation == 1:
               
                if curv == "n" :
                    nbVectors += 3
                if curv == "k" :
                    nbVectors += 2
                if curv == "n" :
                    y.append(yVectorClassicTrain)
                y.append(yVectorKNNValidation)
                y.append(yVectorParzenValidation)

                if curv == "n" :
                    colorVect.append("g-.")
                colorVect.append("b-.")
                colorVect.append("r-.")

                if curv == "n" :
                    legendVect.append("k=1 on validation ")
                    legendVect.append("k="+str(K)+" on validation")
                if curv == "k" :
                    legendVect.append("Validation")
                legendVect.append("Parzen theta="+str(Theta)+" on validation")

            for i in range( nbVectors ):
                x.append(xVector)  

            if curv == "n" :
                filename = IMG_DIR + "ErrorKnn"+str(K)+"Ex"+str(nbExemples)+categorie
                title = "Error Rate on Train/Test with "+categorie
                xlabel = "Examples p. class"
            if curv == "k" :
                filename = IMG_DIR + "ErrorKnnEx"+str(nbExemples)+categorie
                title = "Error Rate with "+str(nbExemples)+" on "+categorie
                xlabel = "K"
            tools.drawCurves( x, y, colorVect, legendVect, title=title, xlabel=xlabel, ylabel="Error rate", filename=filename)

            #### construction fichier pour courbes ameliorees
            if stock == 1 :
                fichier = open("curvErrorKnn"+str(K)+"Ex"+str(nbExemples)+categorie,"w")
                fichier.write("#nbExTrain errorClassicTrain errorClassicTest errorKNNTrain errorKNNTest errorParzenTrain errorParzenTest\n")
                for i in range(len(xVector)) :
                    fichier.write(str(xVector[i])+" "+str(yVectorClassicTrain[i])+" "+str(yVectorClassicTest[i])+" "+str(yVectorKNNTrain[i])+" "+str(yVectorKNNTest[i])+" "+str(yVectorParzenTrain[i])+" "+str(yVectorParzenTest[i])+"\n")
                fichier.close()

        else : 
            listeRes = faceReco.main( algo=algo_type )

        
    
    elif algo_type == "NNET":
        n_epoch = args.n_epoch
        n_hidden = args.n_hidden
        batch = args.batch_size
        lr = args.lr
        wd = args.wd
        faceReco = Main( batch_size=batch, n_epoch=n_epoch, n_hidden=n_hidden, lr=lr, wd=wd, 
                         trainFile=trainFile, testFile=testFile, debug_mode=debug_mode, categorie=categorie, stock=stock, curv=curv, validation=validation, nbExemples=nbExemples)
        faceReco.main( algo=algo_type )

