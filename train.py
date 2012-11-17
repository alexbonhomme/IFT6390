#! /usr/bin/env python2
# -*- coding: utf-8 -*-
###################################
#
# Reconnaissance par "Eigenfaces"
# Partie "apprentissage"
#
###################################

import numpy as np
import scipy as sp
import matplotlib.pylab as pylab
import sys

"""
	Importe les images liste "filename" et les convertient en vecteurs
"""
def loadImageData( filename ):

	import Image as im

	# Lecture du fichier
	imageList = []
	f = open(filename, 'r')

	for line in f:
		e = line.split() # On split selon les espaces
		imageList.append([e[0], e[1]])

	f.close()
	imageList = np.array( imageList )

	# Recuperation des valeurs depuis les images
	data = []
	for image in imageList[:,1]:
		img = im.open("orl_faces/"+ str( image ))
		data.append( list(img.getdata()) )

	return np.transpose( data ), imageList[:, 0].astype(int)

"""
	Sauvegarde les donné d'entrainement dans le fichier "train.xml"
"""
def storeData():
	
	from xml.dom.minidom import parseString
	
	#dom = parse("train.xml")
	dom = parseString("<dataTrain></dataTrain>")
	
	x = dom.createElement("matrix")  # creates <foo />
	mat = np.array([[1,2,3], [4,5,6]])
	
	txt = dom.createTextNode(str(mat))  # creates "hello, world!"
	
	x.appendChild(txt)  # results in <foo>hello, world!</foo>
	dom.childNodes[0].appendChild(x)  # appends at end of 1st child's children
	
	# Ecriture dans le fichier xml
	f = open("train.xml", 'w')
	f.write( dom.toxml() )
	f.close
	
def loadData():
	from xml.dom.minidom import parse
	
	dom = parse("train.xml")
	e = dom.getElementsByTagName("matrix")
	x = np.array(e[0].childNodes[0].nodeValue)
	print x.shape

def doPCAandProjection( data ):

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

def projectTestFace( dataTest, eigenVectors, avgFaceVector ):
	# Projection des images dans la base des eigenvectors
	return np.dot( np.transpose(eigenVectors), (dataTest - avgFaceVector) )

"""
	Recherche du plus proche voisin (k = 1)
	Utilise la distance Euclidien ou de Mahalanobis
"""
def findNearestNeighbor( projectTrainFace, projectTestFace, eValues, distCalc='E' ):
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
	Calcul la distance de Minkowski entre un vecteur et une matrice
"""
def minkowski_mat(x, Y, p=2.0, axis=0):
	x = x.reshape( x.shape[0], 1 )
	return np.sum( np.abs( x - Y )**p, axis=axis )**(1.0/p)
"""
# slow algo
def euclide_mat(x, Y):
	out = []
	for p in range(0, int( Y.shape[1] )):
		deltaSq = 0
		for i in range(0, int( Y.shape[0] )):
			delta = x[i] - Y[i, p]
			deltaSq += delta*delta  # Eclidean distance
		out.append( np.sqrt(deltaSq) )
	
	return np.array( out )
"""
"""
	Gaussienne
	X: centre de la gaussienne
	x: coordonnée de la valeurs recherché
	theta: ecart type (ou largeur du noyau...)
"""	
#def softKernel(X, x, theta=0.5, d=2):
#	return (1/( ((2*np.pi)**(d/2)) * theta**d )) * np.exp( (-1/2)* ((np.abs( X - x )**2) / (theta**2)) )

def gaussianKernel(X, x, theta=0.5):
	return np.exp( -((X - x)**2) / (theta**2) ) 

def softKernelDist(dist, d=2, theta=0.5):
	return (1/( ((2*np.pi)**(d/2)) * (theta**d) )) * np.exp((-1/2) * ((dist) / (theta**2)))

"""
	Fenetre de Parzen
"""
def parzenWindow( projectTestFace, projectTrainFace, dataTrainIndices, k=3, theta=0.5 ):
	
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
def kNearestNeighbors( projectTestFace, projectTrainFace, dataTrainIndices, k=3 ):
	
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

"""
def doImageFile(mode, size, data, name='out', ext='JPEG'):
	img = im.new(mode, size)
	img.putdata(data)
	img.save(name, ext)
#"""

## MAIN ###########################

# Parametres du script
K = 1
Theta = 0.5
trainFile = "train.txt"
testFile = "test.txt"

if( np.size(sys.argv) > 2 ):
	for i in range(1, np.size(sys.argv), 2):
		if(sys.argv[i] == "-k"):
			K = int( sys.argv[i+1] )
		
		elif(sys.argv[i] == "-theta"):
			Theta = float( sys.argv[i+1] )

		elif(sys.argv[i] == "-train"):
			trainFile = sys.argv[i+1]

		elif(sys.argv[i] == "-test"):
			testFile = sys.argv[i+1]


# Chargement des données
dataTrain, dataTrainIndices = loadImageData( trainFile )

# Calcul des eigenvectors, eigenValues et projection
eigenVectors, eigenValues, projectedTrainFaces = doPCAandProjection( dataTrain )


## TEST ###########################
dataTest, dataTestIndices = loadImageData( testFile )
avg = avg = np.mean(dataTest, axis=1)

nbGoodResult = 0
nbGoodResult2 = 0 # compteurs de bons résultats
nbGoodResult3 = 0

#x = np.array(range(0, 100))
#y = gaussianKernel( 50, x, theta=Theta )
#print y

#pylab.plot(x, y)
#pylab.show()
storeData()
loadData()

for i in range(0, int( dataTest.shape[1] )):
	projectedTestFace = projectTestFace( dataTest[:,i], eigenVectors, avg )

	iDataTrain = findNearestNeighbor( projectedTrainFaces, projectedTestFace, eigenValues )
	if(dataTrainIndices[iDataTrain] == dataTestIndices[i]):
		nbGoodResult += 1
		
	resultKNN = kNearestNeighbors( projectedTestFace, projectedTrainFaces, dataTrainIndices, k=K )
	if(resultKNN == dataTestIndices[i]):
		nbGoodResult2 += 1
		
	resultParzen = parzenWindow( projectedTestFace, projectedTrainFaces, dataTrainIndices, k=K, theta=Theta )
	if(resultParzen == dataTestIndices[i]):
		nbGoodResult3 += 1
	
	print "Classic method: "+ str( dataTrainIndices[iDataTrain] ) +" | KNN method: "+ str( resultKNN ) +" | KNN+Parzen method: "+ str( resultParzen ) +" | Expected: "+ str( dataTestIndices[i] ) +"\n" # +1 car l'index de la matrice commence a 0

print "Accuracy with classic method: "+ str( (nbGoodResult / float(dataTest.shape[1])) * 100 ) +"%"
print "Accuracy with KNN method (k="+ str(K) +"): "+ str( (nbGoodResult2 / float(dataTest.shape[1])) * 100 ) +"%"
print "Accuracy with KNN + Parzn window method (k="+ str(K) +" theta="+ str(Theta) +"): "+ str( (nbGoodResult3 / float(dataTest.shape[1])) * 100 ) +"%"


