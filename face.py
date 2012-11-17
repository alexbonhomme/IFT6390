#! /usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import Image as im
import matplotlib.pylab as pylab

### Fonctions ###

## Reconstruction de l'image
def buildImage(mode, size, data, name='out'):
	imgOut = im.new(mode, size)
	imgOut.putdata(data)
	imgOut.save(name, 'JPEG')

### Main ###

## Face vector space building

print "Construction du dataset depuis les images..."
"""
for j in range(1, 41):
	for i in range(1, 10):
		img = im.open("orl_faces/s"+ str(j) +"/"+ str(i) +".pgm")
		if(j==1 and i==1):
			faceVectorSpace = np.transpose(list(img.getdata()))
		else:
			faceVectorSpace = np.c_[faceVectorSpace, np.array(list(img.getdata()))]
"""

for j in range(1, 10):
	img = im.open("orl_faces/s1/"+ str(j) +".pgm")
	if(j==1):# and i==1):
		faceVectorSpace = np.transpose(list(img.getdata()))
	else:
		faceVectorSpace = np.c_[faceVectorSpace, np.array(list(img.getdata()))]

## Average face vector calcul

print "Calcul du visage moyen..."

averageFaceVector = []
for i in range(0, img.size[0] * img.size[1]):
	averageFaceVector.append(np.mean(faceVectorSpace[i,:]))

averageFaceVector = np.transpose(averageFaceVector)
#buildImage(img.mode, img.size, averageFaceVector, 'averageFaceVector')

## Normalize face vectors

print "Normalisation des vecteurs image..."

#for i in range(0, (9*40)-1):
for i in range(0, int( faceVectorSpace.shape[1] )):
	faceVectorSpace[:,i] = faceVectorSpace[:,i] - averageFaceVector


## Calcul of eigen vectors
print "Calcul des vecteur propres par SVD..."
U,s,V = np.linalg.svd(faceVectorSpace, full_matrices=False)

## Calcul of weights vectors
print "Calcul des coefficients..."
weightsVectors = np.dot( np.transpose(U), faceVectorSpace )

#print "##DEBUG " + str(weightsVectors.shape)

#J = np.dot( U, weightsVectors ) ## Images projetés
#print J.shape

#buildImage(img.mode, img.size, J[:,0] + averageFaceVector, 'ImageReconstruite')

## Partial reconstruction
"""
out = np.transpose(np.zeros(10304))

print "Reconstruction de l'image à partir des vecteurs propres..."
for i in range(0, 200):
	out = out + ( U[:, i] * weightsVectors[i, 0] )
	
buildImage(img.mode, img.size, averageFaceVector + out, 'ImagePartiel')
#"""


## Partial reconstruction analys
#for i in range(0, 5):
"""
error = []
img = im.open("orl_faces/s1/1.pgm")
for x in range(1, 36):
	out = np.transpose(np.zeros(10304))
	for i in range(0, x*10):
		out = out + ( U[:, i] * weightsVectors[i, 0] )
	
	# calcul de l'erreur
	error.append( [x*10, np.linalg.norm(np.transpose(list(img.getdata())) - (out + averageFaceVector) )] )

error = np.array(error)
print error[:,1]

pylab.plot(error[:,0], error[:,1])
pylab.show()
"""


## Unknow image recognition
#"""
print "# Starting recognition #"
print "Image -> vector"
img2 = im.open("orl_faces/s1/5.pgm")
unknowFaceVector = np.transpose(list(img2.getdata()))

print "Normalisation..."
unknowFaceVector = unknowFaceVector - averageFaceVector

print "Calcul des coefficients..."
weightsVectorsOfUnknowFace = np.dot( np.transpose(U), unknowFaceVector )

"""
out = np.transpose(np.zeros(10304))

print "Reconstruction de l'image à partir des vecteurs propres..."
for i in range(0, weightsVectorsOfUnknowFace.shape[0]):
	out = out + ( U[:, i] * weightsVectorsOfUnknowFace[i] )
	
buildImage(img.mode, img.size, averageFaceVector + out, 'ImagePartiel')
#"""

results = []
"""
for i in range(0, 360):
	results.append([i, np.linalg.norm( weightsVectors[:,i] - weightsVectorsOfUnknowFace )])
"""

unknowFace = np.transpose(np.zeros(10304))
for i in range(0, int( weightsVectorsOfUnknowFace.shape[0] )):
	unknowFace = unknowFace + ( U[:, i] * weightsVectorsOfUnknowFace[i] )
unknowFace = unknowFace + averageFaceVector

for x in range(0, int( weightsVectors.shape[1] )):
	datasetFace = np.transpose(np.zeros(10304))
	for i in range(0, int( weightsVectors.shape[0] )):
		datasetFace = datasetFace + ( U[:, i] * weightsVectors[i, x] )

	# calcul de l'erreur
	results.append( [x, np.linalg.norm( unknowFace - (datasetFace + averageFaceVector) )] )

results = np.array(results)
pylab.plot(results[:,0], results[:,1], 'o')
pylab.show()
#"""
