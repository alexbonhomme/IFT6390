#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as pylab

def minkowski_vec(x1, x2, p=2.0):
	dist = np.sum( np.abs( x1 - x2 )**p )**(1.0/p)
	return dist # cette ligne permet de retourner le résultat

#print minkowski_vec(np.array([1, 1]), np.array([2, 2]))

def minkowski_mat (x, Y, p=2.0):
	dist = np.sum( np.abs( x - Y )**p, axis=1 )**(1.0/p)
	return dist


#print minkowski_mat(np.array([1, 1]), np.array([[2, 2], [2, 2]]))

## 1-PPV
def ppv(x, data, p=2):
	caract = data[:,:-1]
	dist = minkowski_mat(x, caract, p)

	return data[ np.argmin( dist ), -1] # argmin trouve l'indice de la valeur la plus faible du tableau

## TEST
iris = np.loadtxt("iris.txt")
p = 2

#for i in range(0, int( iris.shape[0] )):
#	print "Expeted: "+ str( iris[i,-1] ) +" ¦ Found: "+ str( ppv(iris[i,:-1], iris, p) )
	
## k-PPV
def kppv(x, data, p=2, k=1):
	m = 3
	c = np.zeros(m) #tab de taille m = nb classe
	voisins = np.zeros(k)
	voisins[:] = None
	dists = np.zeros(k)
	dists[:] = float('inf')
	
	p=2
	dist = minkowski_mat(x, data[:,:-1], p) #vector distance entre x et data
	
	for i in range(0, int( dist.shape[0] )):
		j = np.argmax(dists)
		if( dist[i] < dists[j] ):
			voisins[j] = data[i,-1] # etiquette de la donnee
			dists[j] = dist[i] # distance
	
	for i in range(0, k):
		if(voisins[i] != None):
			c[ voisins[i]-1 ] += 1 #-1 car les etiquettes commences a 1 mais le tab a zero
	
	return np.argmax( c ) + 1 

## TEST
error = 0
k = 1
for i in range(0, int( iris.shape[0] )):
	expeted = iris[i,-1]
	found = kppv(iris[i,:-1], iris, p, k)
	
	print "Expeted: "+ str( expeted ) +" ¦ Found: "+ str( found )
	if(int(expeted) != found):
		error += 1

total = float(iris.shape[0])
acc = ((total - error) / total)*100
print "Nb error: "+ str(error) +" Total: "+ str(total) +" ¦ Accuracy: "+ str( acc ) +"% with k="+ str( k )

