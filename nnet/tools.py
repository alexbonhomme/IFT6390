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
#   Fonctions utiles
#
############################################
import numpy as np
import pylab

# Non linéarité de type softmax
def softmax(x):
    e = np.exp(x)
    return e / np.sum( e )

def softmaxMat(X):
    out = np.zeros(X.shape)
    for i in xrange(X.shape[1]):
        e = np.exp(X[:,i])
        out[:,i] = e / np.sum( e )
    
    return out

######################################################################################
#
#   Affichage plusieurs courbes sur la même figure
#
#   x: liste des coordonnées en x
#   y: liste des coordonnées en y
#   cType: liste des représentations associées aux courbes
#   legend: liste des légendes associées aux courbes
#   xlim, ylim: bornes en x, y
#   xlabel, ylabel: labels en x, y
#   title: titre de la figure
#   bGrid: active/desactive l'affichage de la grille
#   bDisplay: active/desactive l'affichage des courbes
#   filename: si renseigné, enregistre la figure dans un fichier nommé <filename.png>
#
######################################################################################
def drawCurves(x, y, cType, legend="", xlim="", ylim="", xlabel="", ylabel="", title="", bGrid=True, bDisplay=True, filename=""):
    if len(x) != len(y) != len(cType):
        print "Attention: Les deux listes doivent avoir la même taille."
        return -1
        
    for i in xrange(len(x)):
        pylab.plot(x[i], y[i], cType[i])
        pylab.annotate('Min: '+ str(round(np.min(y[i]),3)),
                        xy=(np.argmin(y[i]), np.min(y[i])),
                        #xytext=(np.argmin(y[i])-(y[i].shape[0]/1000.), np.min(y[i])+(y[i].shape[0]/1000.)),
                        va='top',
                        ha='center')
                        #arrowprops=dict(arrowstyle='->'))
	pylab.grid(bGrid)
	
	# titre / labels / legende
	if title != "":
	    pylab.title(title)
	if xlabel != "":
	    pylab.xlabel(xlabel)
	if ylabel != "":
	    pylab.ylabel(ylabel)
	if legend != "":
	    pylab.legend(legend)
	
	# bornes
	if xlim != "":
	    pylab.xlim(xlim)
	if ylim != "":
	    pylab.ylim(ylim)
    
    # sauvegarde de la courbe
    if filename != "":
	    filename += '.png'
	    print 'On sauvegarde la figure dans', filename
	    pylab.savefig(filename,format='png')

    # affichage
    if bDisplay:
        pylab.show()

    # close
    pylab.clf()

######################################################################################
#####################################
# Fonctions importées de la demo 8
# + quelques modifications mineurs
#####################################
# fonction plot
def gridplot(classifieur, train, test, n_points=50, bDisplay=True, title="", filename=""):

    train_test = np.vstack((train,test))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))

    xgrid = np.linspace(min_x1,max_x1,num=n_points)
    ygrid = np.linspace(min_x2,max_x2,num=n_points)

	# calcule le produit cartesien entre deux listes
    # et met les resultats dans un array
    thegrid = np.array(combine(xgrid,ygrid))

    les_sorties = classifieur.compute_predictions(thegrid)
    classesPred = np.argmax(les_sorties,axis=1)

    # La grille
    # Pour que la grille soit plus jolie
    #props = dict( alpha=0.3, edgecolors='none' )
    pylab.scatter(thegrid[:,0],thegrid[:,1],c = classesPred, s=50)
    # Les points d'entrainment
    pylab.scatter(train[:,0], train[:,1], c = train[:,-1], marker = 'v', s=50)
    # Les points de test
    pylab.scatter(test[:,0], test[:,1], c = test[:,-1], marker = 's', s=50)

    ## Un petit hack, parce que la fonctionalite manque a pylab...
    h1 = pylab.plot([min_x1], [min_x2], marker='o', c = 'w',ms=5) 
    h2 = pylab.plot([min_x1], [min_x2], marker='v', c = 'w',ms=5) 
    h3 = pylab.plot([min_x1], [min_x2], marker='s', c = 'w',ms=5) 
    handles = [h1,h2,h3]
    ## fin du hack

    labels = ['grille','train','test']
    #pylab.legend(handles,labels)

    pylab.axis('equal')
    
    if title != "":
	    pylab.title(title)
    
    if filename != "":
        filename += '.png'
        print 'On va sauvegarder la figure dans ', filename
        pylab.savefig(filename, format='png')
        
    # affichage
    if bDisplay:
        pylab.show()

    # close
    pylab.clf()

## http://code.activestate.com/recipes/302478/
def combine(*seqin):
    '''returns a list of all combinations of argument sequences.
for example: combine((1,2),(3,4)) returns
[[1, 3], [1, 4], [2, 3], [2, 4]]'''
    def rloop(seqin,listout,comb):
        '''recursive looping function'''
        if seqin:                       # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]     # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:],listout,newcomb)
        else:                           # processing last sequence
            listout.append(comb)        # comb finished, add to list
    listout=[]                      # listout initialization
    rloop(seqin,listout,[])         # start recursive process
    return listout
