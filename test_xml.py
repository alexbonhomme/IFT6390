#! /usr/bin/env python2
# -*- coding: utf-8 -*-

class Adresse:

    ville = None

    

    def __init__(self):

        pass
        
class Personne:

    nom = None

    prenom = None

    adresse = Adresse()

    

    def __init__(self):

        pass


class TransformXmlToPersonnes:



    __currentNode__ = None

    __personneList__ = None

    

    def __init__(self):

        self.readXml()



    def readXml(self):

        from xml.dom.minidom import parse

        self.doc = parse('E:/python/samplexml/personnes.xml')



    def getRootElement(self):

        if self.__currentNode__ == None:

            self.__currentNode__ = self.doc.documentElement

        return self.__currentNode__



    def getPersonnes(self):

        if self.__personneList__ != None:

            return 

        self.__personneList__ = []

        for personnes in self.getRootElement().getElementsByTagName("personne"):

            if personnes.nodeType == personnes.ELEMENT_NODE:

                p = Personne()

                try:

                    p.nom = self.getText(personnes.getElementsByTagName("nom")[0])

                    p.prenom = self.getText(personnes.getElementsByTagName("prenom")[0])

                    p.adresse = self.getAdresse(personnes.getElementsByTagName("adresse")[0])

                except:

                    print 'Un des TAGS suivant est manquents : nom, prenom, adresse'

                self.__personneList__.append(p)

        return self.__personneList__



    def getAdresse(self, node):

        adress = Adresse()

        try:

            adress.ville = self.getText(node.getElementsByTagName("ville")[0])

        except:

            adress.ville = None

        return adress

                

    def getText(self, node):

        return node.childNodes[0].nodeValue

    

if __name__ == "__main__":

    x=TransformXmlToPersonnes()

    print x.getPersonnes()[1].nom
