#! /usr/bin/env python2
# -*- coding: utf-8 -*-
###################################
#
# Reconnaissance par "Eigenfaces"
#
###################################
from eigenfaces import Eigenfaces
import tools
import sys

## MAIN ###########################

# Parametres du script
K = 1
Theta = 0.5
trainFile = "./Databases/train.txt"
testFile = "./Databases/test.txt"

if( len(sys.argv) > 2 ):
	for i in range(1, len(sys.argv), 2):
		if(sys.argv[i] == "-k"):
			K = int( sys.argv[i+1] )
		
		elif(sys.argv[i] == "-theta"):
			Theta = float( sys.argv[i+1] )

		elif(sys.argv[i] == "-train"):
			trainFile = sys.argv[i+1]

		elif(sys.argv[i] == "-test"):
			testFile = sys.argv[i+1]


# Chargement des données
dataTrain, dataTrainIndices = tools.loadImageData( trainFile )

# eigen model
eigen_model = Eigenfaces(Theta, K)
eigen_model.train(dataTrain, dataTrainIndices)

## TEST ###########################
dataTest, dataTestIndices = tools.loadImageData( testFile )

nbGoodResult = 0
nbGoodResult2 = 0 # compteurs de bons résultats
nbGoodResult3 = 0

#x = np.array(range(0, 100))
#y = gaussianKernel( 50, x, theta=Theta )
#print y

#pylab.plot(x, y)
#pylab.show()
#storeData()
#loadData()

iDataTrain = eigen_model.compute_predictions( dataTest )
resultKNN = eigen_model.compute_predictions( dataTest, "knn" )
resultParzen = eigen_model.compute_predictions( dataTest, "parzen" )

for i in range(0, int( dataTest.shape[1] )):

	
	if(dataTrainIndices[iDataTrain[i]] == dataTestIndices[i]):
		nbGoodResult += 1
		
	
	if(resultKNN[i] == dataTestIndices[i]):
		nbGoodResult2 += 1
		
	
	if(resultParzen[i] == dataTestIndices[i]):
		nbGoodResult3 += 1
	
	print "Classic method: "+ str( dataTrainIndices[iDataTrain] ) +" | KNN method: "+ str( resultKNN ) +" | KNN+Parzen method: "+ str( resultParzen ) +" | Expected: "+ str( dataTestIndices[i] ) +"\n" # +1 car l'index de la matrice commence a 0

print "Accuracy with classic method: "+ str( (nbGoodResult / float(dataTest.shape[1])) * 100 ) +"%"
print "Accuracy with KNN method (k="+ str(K) +"): "+ str( (nbGoodResult2 / float(dataTest.shape[1])) * 100 ) +"%"
print "Accuracy with KNN + Parzn window method (k="+ str(K) +" theta="+ str(Theta) +"): "+ str( (nbGoodResult3 / float(dataTest.shape[1])) * 100 ) +"%"


