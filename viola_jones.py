#! /usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2.cv as cv
import numpy as np
import os.path
import glob
import Image as im
import logging as log
import sys

class Viola_Jones (object):

    # number_faces définit le nombre de visages que l'on souhaite détecter dans l'image 1 ou plusieurs
    def __init__(self):
	
	    # modèle qui fonctionne bien pour un seul visage
		log.debug("Initialisation du modèle de detection Viola&Jones")
		self.hc = cv.Load("haarcascade/haarcascade_frontalface_alt.xml")
		self.faces = []

		# initilisation du cadre des visages		
		self.w_resize = 96
		self.h_resize = 120
		self.ratio_resize = self.w_resize/float(self.h_resize)

    #detection des visages sur une liste d'image se trouvant dans toute l'arborescence du dossier
    def detections_faces_list(self, array_images):
		log.info("Début de la détection des visages (" + str(len(array_images)) + " images)")
		data = []

		#
		# Détection des visages
		# et calcul de taille de la zone d'intéret
		#
		i = 1
		for img in array_images:
			# detection des visage
			face_detect = cv.HaarDetectObjects(cv.LoadImage(img, 0), self.hc, cv.CreateMemStorage())
			self.faces.append( face_detect )

			# Affichage du deroulement
			sys.stdout.write("\r"+str(i)+"/"+str(len(array_images)))
			sys.stdout.flush()
			i += 1
			
			# aucun visage detecté
			if len(face_detect) == 0:
				log.warning("Aucun visage n'a été détecté sur l'image : "+ img)
		
		sys.stdout.write("\n")

		#
		# Extraction du visage 
		# et vectorisation
		#
		data = []
		for i_img in xrange(len(array_images)):
			# on découpe l'image selon les visages et on en retourne des vecteurs images
			for ((x, y, w, h), n) in self.faces[i_img]:
				vectFace = []
			
				# verification du ratio de l'image
				# si le ratio n'est pas le meme que celui defini
				# dans le constructeur on crop certaines zone de l'image
				if w != self.w_resize or h != self.h_resize:
					ratio = w/h
					if ratio != self.ratio_resize:
						if ratio > self.ratio_resize:
							h_resize = int( w/self.ratio_resize )
							y = max(0, y - (h_resize - h))
							h = h_resize
						else:
							w_resize = int( h*self.ratio_resize )
							x = max(0, x - (w - w_resize))
							w = w_resize

				#on récupère l'image du visage
				face = cv.GetSubRect(cv.LoadImage(array_images[i_img], 0), (x, y, w, h))

				# si l'image n'est pas a la bonne dimenssion
				# on resize (pas de pb car le ratio a été ajusté avant)
				if w != self.w_resize or h != self.h_resize:
					#resize
					face_resize = cv.CreateMat(self.w_resize, self.h_resize, face.type)
					cv.Resize(face, face_resize)
					vect = np.asarray(face_resize).reshape(self.w_resize*self.h_resize,)
				else:
					vect = np.asarray(face).reshape(self.w_resize*self.h_resize,)
		        
		        #log.debug(vect.shape)
		        data.append(vect)

		log.info("Fin de la détection : " + str(len(data)) + " visages ont été détectés.")

		return np.array(data).T

