# -*- coding: utf-8 -*-
import numpy as np
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
		img = im.open("Databases/orl_faces/"+ str( image ))
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

"""
    Calcul la distance de Minkowski entre un vecteur et une matrice
"""
def minkowski_mat(x, Y, p=2.0, axis=0):
    x = x.reshape( x.shape[0], 1 )
    return np.sum( np.abs( x - Y )**p, axis=axis )**(1.0/p)

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
    Noyau gaussien
"""
def gaussianKernel(X, x, theta):
    return np.exp( -((X - x)**2) / (theta**2) )

#TODO check this
def softKernelDist( dist, d=2, theta=0.5 ):
    return (1/( ((2*np.pi)**(d/2)) * (theta**d) )) * np.exp((-1/2) * ((dist) / (theta**2)))
  
"""
    Compte le nombre de classe a partir du vecteur de labels/targets
"""
def countClass( targets ):
    m = []
    for i in range(0, int( targets.shape[0] )):
        if( not targets[i] in m ):
            m.append( targets[i] )
    return np.size(m)

"""
    Non linéarité de type softmax
"""
def softmax(x):
    e = np.exp(x)
    return e / np.sum( e )

def softmaxMat(X):
    out = np.zeros(X.shape)
    for i in xrange(X.shape[1]):
        e = np.exp(X[:,i])
        out[:,i] = e / np.sum( e )
    
    return out

