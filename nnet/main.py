#! /usr/bin/env python2
# -*- coding: utf-8 -*-
############################################
#
#	Alexandre BONHOMME
#	BONA20128906
#
#	Devoir 3 - IFT6390
#
#	Réseau de neurone type Perceptron multicouche
#
############################################
#
#
#
############################################
import numpy as np
import random, time, sys
import pylab
import gzip, pickle
import operator

# Fichier de fonctions perso
import tools

# Répertoire d'enregistrement des images
IMG_DIR = "graphes/"

# Choix du des données a utiliser par defaut
DATA_TYPE = "MNIST"
#DATA_TYPE = "2MOONS"
#DATA_TYPE = "IRIS"

# (Hyper)Parametres
lr = .001           # Taux d'apprentissage
wd = .0001          # Penalité L2
n_hidden = 10       # Nombre de neurones cachés
es = False          # Arrêt prématuré
n_epoch = 100       # Nombre d'époque
batch_size = 100    # Taille du (mini)batch

bFiniteDif = False  # Si active, active automatiquement le "slow mode"

# Gestion des arguments du programme
bDisplay = True     # on affiche les image par defaut
bFast = True        # mode optimisé est actif par defaut
if( np.size(sys.argv) > 1 ):
	for i in range(1, np.size(sys.argv), 1):
		
		# Mode Fast (and Furious)
		if sys.argv[i] == "-fast":
		    bFast = True
		elif sys.argv[i] == "-slow":
		    bFast = False
		elif sys.argv[i] == "-finiteDif":
		    bFiniteDif = True
		    bFast = False

		# Affichage des courbes
		elif sys.argv[i] == "-display" and np.size(sys.argv) > i+1: 
		    bDisplay = bool(int(sys.argv[i+1]))
		    i += 1
		
		# Choix des données a utilise
		elif sys.argv[i] == "-data" and np.size(sys.argv) > i+1: 
		    if sys.argv[i+1] == "MNIST":
		        DATA_TYPE = "MNIST"
		    elif sys.argv[i+1] == "2MOONS":
		        DATA_TYPE = "2MOONS"
		    i += 1
		
		# (hyper)parametre
		elif sys.argv[i] == "-lr" and np.size(sys.argv) > i+1:
		    lr = float(sys.argv[i+1])
		    i += 1
		elif sys.argv[i] == "-wd" and np.size(sys.argv) > i+1:
		    wd = float(sys.argv[i+1])
		    i += 1
		elif sys.argv[i] == "-epoch" and np.size(sys.argv) > i+1:
		    n_epoch = int(sys.argv[i+1])
		    i += 1
		elif sys.argv[i] == "-batch" and np.size(sys.argv) > i+1:
		    batch_size = int(sys.argv[i+1])
		    i += 1
		elif sys.argv[i] == "-hidden" and np.size(sys.argv) > i+1:
		    n_hidden = int(sys.argv[i+1])
		    i += 1
		elif sys.argv[i] == "-es" and np.size(sys.argv) > i+1:
		    es = bool(int(sys.argv[i+1]))
		    i += 1

if bFast:
    import nnet_fast as nnet
else:
    import nnet

print "Utilisation des données:", DATA_TYPE

# decommenter pour avoir des resultats non-deterministes 
random.seed(3395)
np.random.seed(3396)

# Taille de la grille = grid_size x grid_size
grid_size = 75

# chargement des donnees
if DATA_TYPE == "2MOONS":
    data = np.loadtxt('2moons.txt')
    #data = np.loadtxt('cercle_plearn.txt')
    
    # Nombre de classes
    n_classes = 2

    # Nombre de points d'entrainement
    n_train = 100
    
    # Les colonnes (traits/caracteristiques) sur lesqueles on va entrainer notre modele
    # Pour que gridplot fonctionne, len(train_cols) devrait etre 2
    train_cols = [0,1]
    # L'indice de la colonne contenant les etiquettes
    target_ind = [data.shape[1] - 1]

    print "On va entrainer sur ", n_train, " exemples d'entrainement"

    # Déterminer au hasard des indices pour les exemples d'entrainement et de test
    inds = range(data.shape[0])
    random.shuffle(inds)

    train_inds = inds[:n_train]
    test_inds = inds[n_train:]

    # Separer les donnees dans les deux ensembles
    train_set = data[train_inds,:]
    train_set = train_set[:,train_cols + target_ind]
    test_set = data[test_inds,:]
    test_set = test_set[:,train_cols + target_ind]

    meantrain = np.mean(train_set[:,:-1],axis=0)
    stdtrain = np.std(train_set[:,:-1],axis=0)

    #### COMMENTEZ POUR VOIR L"INFLUENCE DE LA NORMALISATION
    train_set[:,:-1] = (train_set[:,:-1] - meantrain)/(stdtrain+0.001)
    test_set[:,:-1] = (test_set[:,:-1] - meantrain)/(stdtrain+0.001)
    ####
    
    #dirty(just for tests)
    valid_set = test_set

    # separarer l'ensemble de test dans les entrees et les etiquettes
    test_inputs = test_set[:,:-1]
    test_labels = test_set[:,-1]

elif DATA_TYPE == "IRIS":
    data = np.loadtxt('iris_plearn.txt')
    
    # Nombre de classes
    n_classes = 3

    # Nombre de points d'entrainement
    n_train = 100
    
    # Les colonnes (traits/caracteristiques) sur lesqueles on va entrainer notre modele
    # Pour que gridplot fonctionne, len(train_cols) devrait etre 2
    train_cols = [0,1]
    # L'indice de la colonne contenant les etiquettes
    target_ind = [data.shape[1] - 1]

    print "On va entrainer sur ", n_train, " exemples d'entrainement"

    # Déterminer au hasard des indices pour les exemples d'entrainement et de test
    inds = range(data.shape[0])
    random.shuffle(inds)

    train_inds = inds[:n_train]
    test_inds = inds[n_train:]

    # Separer les donnees dans les deux ensembles
    train_set = data[train_inds,:]
    train_set = train_set[:,train_cols + target_ind]
    test_set = data[test_inds,:]
    test_set = test_set[:,train_cols + target_ind]

    meantrain = np.mean(train_set[:,:-1],axis=0)
    stdtrain = np.std(train_set[:,:-1],axis=0)

    #### COMMENTEZ POUR VOIR L"INFLUENCE DE LA NORMALISATION
    train_set[:,:-1] = (train_set[:,:-1] - meantrain)/(stdtrain+0.001)
    test_set[:,:-1] = (test_set[:,:-1] - meantrain)/(stdtrain+0.001)
    ####
    
    #dirty(just for tests)
    valid_set = test_set

    # separarer l'ensemble de test dans les entrees et les etiquettes
    test_inputs = test_set[:,:-1]
    test_labels = test_set[:,-1]

elif DATA_TYPE == "MNIST":
    f = gzip.open('mnist.pkl.gz')
    data = pickle.load(f)
    
    # Nombre de classes
    n_classes = 10

    print "On va entrainer sur ", data[0][0].shape[0], " exemples d'entrainement"

    train_inputs = data[0][0]
    train_labels = data[0][1].reshape((data[0][1].shape[0], 1))
    train_set = np.hstack((train_inputs, train_labels))

    valid_inputs = data[1][0]
    valid_labels = data[1][1].reshape((data[1][1].shape[0], 1))
    valid_set = np.hstack((valid_inputs, valid_labels))

    test_inputs = data[2][0]
    test_labels = data[2][1].reshape((data[1][1].shape[0], 1))
    test_set = np.hstack((test_inputs, test_labels))

else:
    print "No data !"
    exit()

############################
#
# Init
#
############################
assert train_set.shape[0]%batch_size == 0 # la taille de la batch doit etre un diviseur du nombre de donnees
model = nnet.NeuralNetwork(train_set.shape[1]-1, n_hidden, n_classes, lr, wd)

############################
#
# Train
#
############################
print "\n\tNombre de classe:", n_classes
print "\tNombre de neurones cachés:", n_hidden
print "\n\tNombre d'époque:", n_epoch, "\t\tTaille du batch:", batch_size
print "\tTaux d'apprentissage:", lr, "\tPénalité L2:", wd, "\t Arrêt prématuré:", es, "\n"

if bFast:
    out1, valid_out1, test_out1 = model.train(train_set, n_epoch, batch_size, valid_set=valid_set, test_set=test_set, EarlyStopping=es, filename="data.csv")
else:
    out1 = model.train(train_set, n_epoch, batch_size, fDif=False, EarlyStopping=es)
    
    # On génère une graphique pour la veification par différence finie
    if bFiniteDif:
        x = []
        y = []
        color = []
        legend = []
        out1_dif = model.train(train_set, n_epoch, batch_size, fDif=True, EarlyStopping=es)
        out1_dif = np.array(out1)
        out1_ = np.array(out1)
        x.append(np.array(xrange(out1_.shape[0])))
        x.append(np.array(xrange(out1_dif.shape[0])))
        y.append(out1_)
        y.append(out1_dif)
        color.append('r-')
        color.append('b--')
        legend.append(u"R Descente de gradient")
        legend.append(u"R Différence finie")
        
        filename = IMG_DIR + DATA_TYPE + "/Risque__Epoch_"+ str(n_epoch) +"_Hidden_"+ str(n_hidden) +"_Lr_"+ str(lr) +"_L2_"+ str(wd) +"___DIF"
        title = u"\nEpoque: " + str(n_epoch) + " - Taille du batch: " + str(batch_size) + " - Neurones cachés: " + str(n_hidden) + "\nL2: " + str(wd) + " - Taux d'apprentissage: " + str(lr)
        tools.drawCurves(x, y, color, legend, bDisplay=bDisplay, filename=filename, title=title, xlabel="Epoque", ylabel=u"Risque régularisé")

############################
#
# Affichage des résultats
#
############################
x = []
y = []
y_err = []
color = []
legend = []
legend_err = []
filename = IMG_DIR + DATA_TYPE + "/Risque__Epoch_"+ str(n_epoch) +"_Hidden_"+ str(n_hidden) +"_Lr_"+ str(lr) +"_L2_"+ str(wd) +"_"
filename_err = IMG_DIR + DATA_TYPE + "/Erreur_classification__Epoch_"+ str(n_epoch) +"_Hidden_"+ str(n_hidden) +"_Lr_"+ str(lr) +"_L2_"+ str(wd) +"_"
if out1:
    out1 = np.array(out1)
    x.append(np.array(xrange(out1.shape[0])))
    
    color.append('r-')
    legend.append("R Train")
    filename += "_Train"
    
    if bFast:
        y.append(out1[:,0])
        y_err.append(out1[:,1])
        legend_err.append("Err Train")
        filename_err += "_Train"
    else:
        y.append(out1)
if bFast:   
    if valid_out1:
        valid_out1 = np.array(valid_out1)
        x.append(np.array(xrange(valid_out1.shape[0])))
        y.append(valid_out1[:,0])
        y_err.append(valid_out1[:,1])
        color.append('g-')
        legend.append("R Validation")
        legend_err.append("Err Validation")
        filename += "_Valid"
        filename_err += "_Valid"

    if test_out1:
        test_out1 = np.array(test_out1)
        x.append(np.array(xrange(test_out1.shape[0])))
        y.append(test_out1[:,0])
        y_err.append(test_out1[:,1])
        color.append('b-')
        legend.append("R Test")
        legend_err.append("Err Test")
        filename += "_Test"
        filename_err += "_Valid"

# Affichage courbes R et Erreur de classification
title = u"\nEpoque: " + str(n_epoch) + " - Taille du batch: " + str(batch_size) + " - Neurones cachés: " + str(n_hidden) + "\nL2: " + str(wd) + " - Taux d'apprentissage: " + str(lr)
tools.drawCurves(x, y, color, legend, bDisplay=bDisplay, filename=filename, title=title, xlabel="Epoque", ylabel=u"Risque régularisé")
if bFast:
    tools.drawCurves(x, y_err, color, legend_err, bDisplay=bDisplay, filename=filename_err, title=title, xlabel="Epoque", ylabel="Erreur classification")

##########################################
#
#   Affichage des zones de décisions (2D)
#
##########################################
if DATA_TYPE != "MNIST":
    filename = IMG_DIR + DATA_TYPE + "/Frontieres__Epoch_"+ str(n_epoch) +"_Hidden_"+ str(n_hidden) +"_Lr_"+ str(lr) +"_L2_"+ str(wd)

    # Surface de decision
    t1 = time.clock()
    tools.gridplot(model, train_set, test_set, n_points=grid_size, bDisplay=bDisplay, filename=filename, title=title)
    t2 = time.clock()
    print 'Ca nous a pris ', t2-t1, ' secondes pour calculer les predictions sur ', grid_size * grid_size, ' points de la grille'
