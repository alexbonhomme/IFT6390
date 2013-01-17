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
		self.targets = []

		# initilisation du cadre des visages		
		self.w_resize = 92
		self.h_resize = 115
		self.ratio_resize = self.w_resize/float(self.h_resize)

    #detection des visages sur une liste d'image se trouvant dans toute l'arborescence du dossier
    def detections_faces_list(self, array_images, array_classes):
		log.info("Début de la détection des visages (" + str(len(array_images)) + " images)")

		#
		# Détection des visages
		# et calcul de taille de la zone d'intéret
		#
		for i_img in xrange(len(array_images)):
			# detection des visage
			face_detect = cv.HaarDetectObjects(cv.LoadImage(array_images[i_img], 0), 
											   self.hc, cv.CreateMemStorage(), 
											   1.2, 3, 0, 
											   min_size=(80, 80))
			self.faces.append( face_detect )
			self.targets.append( array_classes[i_img] )
			

			# Affichage du deroulement
			sys.stdout.write("\r"+str(i_img+1)+"/"+str(len(array_images)))
			sys.stdout.flush()
			
			# aucun visage detecté
			if len(face_detect) == 0:
				log.warning("Aucun visage n'a été détecté sur l'image : "+ array_images[i_img])
		
		sys.stdout.write("\n")

		#
		# Extraction du visage 
		# et vectorisation
		#
		faces_list = []
		targets_list = []
		compute_nb_class = []
		for i_img in xrange(len(array_images)):
			if self.faces[i_img] != 0:
				# calcul du nombre de classes
				if( not self.targets[i_img] in compute_nb_class ):
					compute_nb_class.append( self.targets[i_img] )
		
				# on découpe l'image selon les visages et on en retourne des vecteurs images
				for ((x, y, w, h), n) in self.faces[i_img]:
					im = cv.LoadImage(array_images[i_img], 0)
				
					# verification du ratio de l'image
					# si le ratio n'est pas le meme que celui defini
					# dans le constructeur on crop certaines zone de l'image
					if w != self.w_resize or h != self.h_resize:
						ratio = w/h
						if ratio != self.ratio_resize:
							if ratio > self.ratio_resize:
								h_resize = int( w/self.ratio_resize )
								pad = (h_resize - h)/2
								if (y-pad) + h_resize < im.height:
									y = max(0, y - pad)
								else:
									y = im.height - h_resize
								h = h_resize
							else:
								w_resize = int( h*self.ratio_resize )
								pad = (w - w_resize)/2
								if (x - pad) + w_resize < im.width:
									x = max(0, x - pad)
								else:
									x = im.width - w_resize
								w = w_resize

					#on récupère l'image du visage
					face = cv.GetSubRect(im, (x, y, w, h))

					# si l'image n'est pas a la bonne dimenssion
					# on resize (pas de pb car le ratio a été ajusté avant)
					if w != self.w_resize or h != self.h_resize:
						#resize
						face_resize = cv.CreateMat(self.h_resize, self.w_resize,  face.type)
						cv.Resize(face, face_resize)
						vect = np.asarray(face_resize).reshape(self.w_resize*self.h_resize,)
					else:
						vect = np.asarray(face).reshape(self.w_resize*self.h_resize,)
					
					faces_list.append(vect)
					targets_list.append(self.targets[i_img])

				    
		log.info("Fin de la détection : " + str(len(faces_list)) + " visages ont été détectés.")

		tmp = np.array(targets_list).reshape(len(targets_list), 1)
		log.debug(tmp.shape)
		return np.transpose(faces_list), tmp, len(compute_nb_class)

