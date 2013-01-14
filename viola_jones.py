#! /usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2.cv as cv
#import cv2 as cv

# Example d'utilisation d'opencv pour extraire le visage
# Il ne reste plus qu'a faire une classe et a retourne une array avec numpy
# NOTE: Visiblement avec la version 2 d'opencv numpy est "integré"
# et donc opencv comprends les array etc...

# On charge le classifieur a utiliser (je n'ai mis que les classifieurs de visage dans ce dossier.
# Mais il en existe d'autres pour les yeux etc.. si ca vous interesse)))
hc = cv.Load("haarcascade/haarcascade_frontalface_default.xml") # fonctionne tres bien pour un seul visage
#hc = cv.Load("haarcascade/haarcascade_frontalface_alt_tree.xml") # beaucoup mieux pour plusieurs visages

# on charge une image (en grayscale)
img = cv.LoadImage("Databases/lena.jpg", 0)
#img = cv.LoadImage("Databases/mariage.jpg", 0)
   
# detection des visage
faces = cv.HaarDetectObjects(img, hc, cv.CreateMemStorage()) 

i = 0
for ((x,y,w,h), n) in faces:
    cv.SaveImage("faces_"+ str(i) +".pgm", cv.GetSubRect(img, (x, y, w, h))) # enregistrement des visages détecté, c'est cela qu'il faut modifier
    cv.Rectangle(img, (x, y), (x+w, y+h), 255)
    i += 1

# Saugarde de l'image de base avec les zone detecté (pour visu)
cv.SaveImage("faces_detected.pgm", img)
