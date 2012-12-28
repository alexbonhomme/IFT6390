#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import pygtk
pygtk.require('2.0')
import gtk

class Gui:

    def __init__(self):
    
        self.GUI_VERSION = "0.1"
    
        # Parametre pour execution du programme principal
        self.algoType = "kppv" # Par défaut
        self.K = 1 # voisinage pour kppv    
    
        # On construit la fenetre principale
        self.buildDisplay()

    # Contruction de l'interface
    def buildDisplay(self):
        # Fenetre principale
        self.mainWindow = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.mainWindow.set_title("Reconnaissance facial GUI - v" + self.GUI_VERSION)

        # Signal de fermeture (croix, fermer, etc..)
        self.mainWindow.connect("delete_event", self.evnmt_delete)

        # Cet événement se produit lorsqu'on invoque gtk_widget_destroy() sur
        # la fenêtre ou lorsque le gestionnaire du signal "delete" renvoie FALSE.
        self.mainWindow.connect("destroy", self.destroy)

        # box principale
        main_box = gtk.VBox(True, 0)
        main_box.set_border_width(10)
        main_box.show()

        # menu de selection des algo 
        lab_menu1 = gtk.Label("Algorithme a utiliser :")
        lab_menu1.set_alignment(0, 0)
        lab_menu1.show()
        
        menu1 = gtk.OptionMenu()
        menu_content = gtk.Menu()
        
        # entrées du menu
        item = gtk.MenuItem("K-PPV")
        item.connect("activate", self.updateAlgoType, "kppv")
        item.show()
        menu_content.append(item)
        
        item = gtk.MenuItem("Réseau de neurones")
        item.connect("activate", self.updateAlgoType, "nnet")
        item.show()
        menu_content.append(item)
        menu1.set_menu(menu_content)
        menu1.show()
        
        box1 = gtk.HBox(True, 0)
        box1.pack_start(lab_menu1, True, True, 5)
        box1.pack_start(menu1, True, True, 5)
        box1.show()
        frame_0 = gtk.Frame("")
        frame_0.add(box1)
        frame_0.show()
        main_box.pack_start(frame_0, False, False, 10)
        
        # Cette box contient troutes les box de parametre (kppv, nnet, etc..)
        # mais certaines sont masquées
        box_param = gtk.VBox(True, 0)
        box_param.show()
        frame_1 = gtk.Frame("Paramètres")
        frame_1.add(box_param)
        frame_1.show()
        main_box.pack_start(frame_1, True, True, 10)

        # Box nnet
        lab_1 = gtk.Label("Ici NNET")
        lab_1.show()
        self.box_nnet = gtk.HBox(True, 0)
        self.box_nnet.pack_start(lab_1, True, True, 5)
        box_param.pack_start(self.box_nnet, True, True, 5)
        
        # Box kppv
        lab_2 = gtk.Label("Nombre de voisins a consulter :")
        lab_2.set_justify(gtk.JUSTIFY_LEFT)
        lab_2.show()
        
        ajt_k = gtk.Adjustment(1.0, 1.0, 999.0, 1.0, 5.0, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "K_value")
        box_incr.show()
        
        self.box_kppv = gtk.HBox(True, 0)
        self.box_kppv.pack_start(lab_2, True, True, 5)
        self.box_kppv.pack_start(box_incr, True, True, 5)
        
        box_param.pack_start(self.box_kppv, True, True, 5)
        self.box_kppv.show()
        
        # Bottom frame
        bt_run = gtk.Button("Exécuter l'algorithme", gtk.STOCK_EXECUTE)
        bt_run.connect("clicked", self.run, None)
        bt_run.show();
        box3 = gtk.HBox(True, 0)
        box3.pack_start(bt_run, True, True, 0)
        box3.show()
        main_box.pack_start(box3, True, True, 0)

        # Placement dans la fenetre principale
        self.mainWindow.add(main_box)
        self.mainWindow.show()

    # Main loop
    def main(self):
        gtk.main()
        return 0

    # Events
    def evnmt_delete(self, widget, event, data=None):
        return False

    def destroy(self, widget, data=None):
        print "Fin du programme."
        gtk.main_quit()
    
    # Callback functions
    def run(self, widget, data):
        if self.algoType == "kppv":
            print "Run K-PPV with", self.K, "neigbours."
        elif self.algoType == "nnet":
            print "Run NNET"
    
    def updateAlgoType(self, widget, algoType):
        self.algoType = algoType
        
        if self.algoType == "kppv":
            self.box_nnet.hide()
            self.box_kppv.show()
        elif self.algoType == "nnet":
            self.box_kppv.hide()
            self.box_nnet.show()
    
    def updateParam(self, widget, data):
        if data == "K_value":
            self.K = widget.get_value_as_int()
            

# Si le script est directement executé (pas importé) on lance l'interface graphique
if __name__ == "__main__":
    myGui = Gui()
    myGui.main()
