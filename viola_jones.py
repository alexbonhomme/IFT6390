#! /usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2.cv as cv
import numpy
import os.path
import glob
import Image as im

class Viola_Jones():

	# number_faces définit le nombre de visages que l'on souhaite détecter dans l'image 1 ou plusieurs
	def __init__(self):
		
		# modèle qui fonctionne bien pour un seul visage
		self.hc = cv.Load("haarcascade/haarcascade_frontalface_default.xml")

		# initilisation du cadre des visages		
		self.width_minimum = float("inf")
		self.height_minimum = float("inf")
	
	#obtention des cadres de découpe pour une liste
	def detection_size(self, name_img):
		# on charge une image (en grayscale)
		img = cv.LoadImage(name_img, 0)
  
		# detection des visage
		faces = cv.HaarDetectObjects(img, self.hc, cv.CreateMemStorage())
		
		# on récupère la zone de visage la plus petite pour l'utiliser comme norme
		i = 0
		for ((x,y,w,h), n) in faces:			
                    if(w < self.width_minimum) : self.width_minimum = w
		    if(h < self.height_minimum) : self.height_minimum = h


	#découpage des photos avec les tailles obtenues
	def get_faces(self, name_img):
		# on charge une image (en grayscale)
		img = cv.LoadImage(name_img, 0)
  
		# detection des visage
		faces = cv.HaarDetectObjects(img, self.hc, cv.CreateMemStorage())

		# on découpe l'image selon les visages et on en retourne des vecteurs images
		data = []

		for ((x,y,w,h), n) in faces:

		    #on récupère l'image du visage
		    face = cv.GetSubRect(img, (x, y, self.width_minimum, self.height_minimum))

		    #on vectorise l'image du visage
		    for j in range(0, face.height):
        		for i in range(0, face.width):
            	     		data.append(face[j, i])

		return data 


	#detection des visages sur une liste d'image se trouvant dans toute l'arborescence du dossier
	def detections_faces_list(self, array_images):
		data = []
	
		#détection des visages
		
		#dimension du cadre pour les visages
		for img in array_images : self.detection_size(img)

		#vectorisation des visages
		for img in array_images : data.append(list(self.get_faces(img)))

		return numpy.transpose(numpy.array(data))

