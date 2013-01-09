# -*- coding: utf-8 -*-
import numpy as np
import sys
import os

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
        Modifie les couleurs de l'image en nuances de gris
"""
def changeToGreyTot():

	fichier=open("lfwNames.txt")
	listeFiles=fichier.read().split('\n')
	listeFiles=listeFiles[2:-1]
	
	for i in range(len(listeFiles)):
		changeToGrey(listeFiles[i])
		print(i)
		
"""

"""
def changeToGrey(filename):

	import Image
	from pygame import image
	import PIL
	# Recuperation resolution
	image=image.load(filename)
	resolution=(image.get_width(),image.get_height())
	
	# Modification pixels
	img = Image.open(filename)
	pix=img.load()
	for i in range(resolution[0]):
		for j in range(resolution[1]):
			gris=int(round(0.299*pix[i,j][0]+0.587*pix[i,j][1]+0.114*pix[i,j][2]))
			pix[i,j]=(gris,gris,gris)
	img.save(filename)

"""
        Recupere la liste des images de type precise ('lfw' ou 'orl')
"""
def listPictures(indiceSeparation,liste,type="Databases/LFW/lfw"):
		import os, mimetypes
		classesTrain=0
		classesTest=0
		exemplesTrain=0
		exemplesTest=0
		with_extension = 1
 
		path = os.path.join( os.getcwd(), type)
		finalTrain= []
		finalTest= []

		for root, dirs, files in os.walk(path):
			# get length of path
			if( root is path ) :
				count = len(root)
 
			if(files):
				compteur=0
				ajoutTrain=1
				ajoutTest=1
				for f in files:
					if int(with_extension) == 0:
						f = f.split('.')[0]
					if f!="README" and indiceSeparation==0:
						print("COUCOU")
						finalTrain.append(type+root[count:]+"/"+f)	
					elif f!="README" and compteur<indiceSeparation and (root[count+1:] in liste or len(liste)==0):
						if ajoutTrain==1:
							classesTrain+=1
						exemplesTrain+=1
						finalTrain.append(type+root[count:]+"/"+f)	
						ajoutTrain=0
						compteur+=1
					elif f!="README" and compteur>=indiceSeparation and (root[count+1:] in liste or len(liste)==0):
						print("CACA")
						if ajoutTest==1:
							classesTest+=1
						exemplesTest+=1
						finalTest.append(type+root[count:]+"/"+f)
						ajoutTest=0
						compteur+=1
					
             
		return (finalTrain,finalTest,classesTrain,classesTest,exemplesTrain,exemplesTest)  

"""
        Sauvegarde les noms des images dans lfwNames.txt et orlNames.txt
"""
def picturesDictionaryConstruction():
	liste=listPictures(0,[])[0]
	liste.sort()
	fichier=file('./lfwNames.txt','w')
	fichier.write("jpg\n")
	fichier.write(str(len(liste))+'\n')
	for i in range(len(liste)):
		fichier.write(liste[i]+'\n')
	fichier.close()
	liste=listPictures(0,[],"Databases/orl_faces")[0]
	liste.sort()
	fichier=file('./orlNames.txt','w')
	fichier.write("pgm\n")
	fichier.write(str(len(liste))+'\n')
	for i in range(len(liste)):
		fichier.write(liste[i]+'\n')
	fichier.close()

"""
        Construit les fichiers train.txt et test.txt contenant les noms des images utilisées
"""
def trainAndTestConstruction(nbTrain):
	fichier=open("Databases/LFW/lfw-names_current.txt",'r')
	lignes=fichier.read().split('\n')
	lignes=lignes[:-1]  #On supprime le dernier element ''
	fichierTrain=file('train.txt','w')
	fichierTest=file('test.txt','w')
	nom=[]
	nbMax=[]
	for i in range(len(lignes)):
		(mot,nb)=(lignes[i].split('\t')[0],int(lignes[i].split('\t')[1]))
		nbMax.append(nb)
		nom.append(mot)
	(listeTrain,listeTest,classesTrainLFW,classesTestLFW,exemplesTrainLFW,exemplesTestLFW)=listPictures(nbTrain,nom)
	(listeTrainORL,listeTestORL,classesTrainORL,classesTestORL,exemplesTrainORL,exemplesTestORL)=listPictures(nbTrain,[],"Databases/orl_faces")
	listeTrain.sort()
	listeTest.sort()
	listeTrainORL.sort()
	listeTestORL.sort()
	fichierTrain.write(str(classesTrainLFW)+' '+str(exemplesTrainLFW)+'\n'+str(classesTrainORL)+' '+str(exemplesTrainORL)+'\n')
	fichierTest.write(str(classesTestLFW)+' '+str(exemplesTestLFW)+'\n'+str(classesTestORL)+' '+str(exemplesTestORL)+'\n')
	for i in range(len(nom)):
		fichierTrain.write(nom[i]+' '+str(np.min((nbTrain,nbMax[i])))+'\n')
		nbTest=nbMax[i]-nbTrain
		if nbTest>0:
			fichierTest.write(nom[i]+' '+str(nbTest)+'\n')
	for i in range(len(listeTrain)):
		fichierTrain.write(listeTrain[i]+'\n')
	for i in range(len(listeTest)):
		fichierTest.write(listeTest[i]+'\n')	
	for i in range(len(listeTrainORL)):
		fichierTrain.write(listeTrainORL[i]+'\n')
	for i in range(len(listeTestORL)):
		fichierTest.write(listeTestORL[i]+'\n')
	fichierTrain.close()
	fichierTest.close()		

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

