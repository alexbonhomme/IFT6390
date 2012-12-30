#! /usr/bin/env python2
# -*- coding: utf-8 -*-
import main as MainConsole
from threading import Thread

import pygtk
pygtk.require('2.0')
import gtk

import sys
import logging as log

class Gui:

    def __init__(self, debug_mode=False):
    
        self.GUI_VERSION = "0.1"
    
        # Parametre pour execution du programme principal
        self.algoType = "kppv" # Par défaut
        
        # voisinage pour kppv
        self.K = 1  
        self.Theta = 0.5  
        
        # paremetres nnet (par default)
        self.batch_size = 1
        self.n_epoch = 100
        self.n_hidden = 10
        self.lr = .001
        self.wd = .0
    
        # On construit la fenetre principale
        self.buildDisplay()
        
        #debug
        # logger pour debbug
        self.debug_mode = debug_mode
        if self.debug_mode:
            log.basicConfig(stream=sys.stderr, level=log.DEBUG)
        else:
            log.basicConfig(stream=sys.stderr, level=log.INFO)

    # Contruction de l'interface
    def buildDisplay(self):
        # Fenetre principale
        self.mainWindow = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.mainWindow.set_title("Reconnaissance facial GUI - v" + self.GUI_VERSION)

        # Signal de fermeture (croix, fermer, etc..)
        self.mainWindow.connect("destroy", lambda wid: gtk.main_quit())
        self.mainWindow.connect("delete_event", lambda a1,a2: gtk.main_quit())

        # box principale
        main_box = gtk.VBox(True, 0)
        main_box.set_border_width(10)
        main_box.show()

        # menu de selection des algo 
        lab_menu1 = gtk.Label("Algorithme a utiliser :")
        #lab_menu1.set_alignment(0, 0)
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
        
        box_select = gtk.VBox(True, 0)
        box_select.show()
        frame_0 = gtk.Frame("")
        frame_0.add(box_select)
        frame_0.show()
        main_box.pack_start(frame_0, True, True, 10)
        
        box1 = gtk.HBox(True, 0)
        box1.pack_start(lab_menu1, True, True, 5)
        box1.pack_start(menu1, False, False, 5)
        box1.show()
        box_select.pack_start(box1, False, False, 10)
        
        # Cette box contient troutes les box de parametre (kppv, nnet, etc..)
        # mais certaines sont masquées
        box_param = gtk.VBox(True, 0)
        box_param.show()
        frame_1 = gtk.Frame("Paramètres")
        frame_1.add(box_param)
        frame_1.show()
        main_box.pack_start(frame_1, True, True, 10)

        # Box nnet
        self.box_nnet = gtk.VBox(True, 0)
        box_h1 = gtk.HBox(True, 0)
        lab_1 = gtk.Label("Nombre d'époques :")
        lab_1.set_justify(gtk.JUSTIFY_LEFT)
        lab_1.show()
        ajt_k = gtk.Adjustment(self.n_epoch, 1.0, 100000.0, 1.0, 5.0, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "Epoch_value")
        box_incr.show()
        box_h1.pack_start(lab_1, True, True, 5)
        box_h1.pack_start(box_incr, True, True, 5)
        
        lab_1 = gtk.Label("Nombre de neurones :")
        lab_1.set_justify(gtk.JUSTIFY_LEFT)
        lab_1.show()
        ajt_k = gtk.Adjustment(self.n_hidden, 1.0, 10000.0, 1.0, 5.0, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "Hid_value")
        box_incr.show()
        box_h1.pack_start(lab_1, True, True, 5)
        box_h1.pack_start(box_incr, True, True, 5)
        
        lab_1 = gtk.Label("Taille du batch :")
        lab_1.set_justify(gtk.JUSTIFY_LEFT)
        lab_1.show()
        ajt_k = gtk.Adjustment(self.batch_size, 1.0, 100000., 1.0, 5.0, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "Batch_value")
        box_incr.show()
        box_h1.pack_start(lab_1, True, True, 5)
        box_h1.pack_start(box_incr, True, True, 5)
        box_h1.show()

        self.box_nnet.pack_start(box_h1, True, True, 5)
        
        box_h1 = gtk.HBox(True, 0)
        lab_1 = gtk.Label("Taux d'apprentissage :")
        lab_1.set_justify(gtk.JUSTIFY_LEFT)
        lab_1.show()
        ajt_k = gtk.Adjustment(self.lr, 0.0, 1.0, 0.001, 0.01, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_digits(5)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "Lr_value")
        box_incr.show()
        box_h1.pack_start(lab_1, True, True, 5)
        box_h1.pack_start(box_incr, True, True, 5)
        
        lab_1 = gtk.Label("Pénalité L2 :")
        lab_1.set_justify(gtk.JUSTIFY_LEFT)
        lab_1.show()
        ajt_k = gtk.Adjustment(self.wd, 0.0, 1.0, 0.001, 0.01, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_digits(5)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "Wd_value")
        box_incr.show()
        box_h1.pack_start(lab_1, True, True, 5)
        box_h1.pack_start(box_incr, True, True, 5)
        box_h1.show()
        
        self.box_nnet.pack_start(box_h1, True, True, 5)
        box_param.pack_start(self.box_nnet, True, True, 5)
        
        # Box kppv
        self.box_kppv = gtk.HBox(True, 0)
        lab_1 = gtk.Label("Nombre de voisins a consulter :")
        lab_1.set_justify(gtk.JUSTIFY_LEFT)
        lab_1.show()
        
        ajt_k = gtk.Adjustment(self.K, 1.0, 999.0, 1.0, 5.0, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "K_value")
        box_incr.show()
        
        self.box_kppv.pack_start(lab_1, True, True, 5)
        self.box_kppv.pack_start(box_incr, True, True, 5)
        
        lab_1 = gtk.Label("Ecart type (Parzen) :")
        lab_1.set_justify(gtk.JUSTIFY_LEFT)
        lab_1.show()
        
        ajt_k = gtk.Adjustment(self.Theta, 0.0, 1.0, 0.01, 0.10, 0.0)
        box_incr = gtk.SpinButton(ajt_k, 0, 0)
        box_incr.set_digits(3)
        box_incr.set_numeric(True)
        box_incr.set_wrap(False)
        box_incr.set_snap_to_ticks(True)
        box_incr.connect("value-changed", self.updateParam, "Theta_value")
        box_incr.show()
        
        self.box_kppv.pack_start(lab_1, True, True, 5)
        self.box_kppv.pack_start(box_incr, True, True, 5)
        
        box_param.pack_start(self.box_kppv, True, True, 5)
        self.box_kppv.show()
        
        # output
        box2 = gtk.HBox(True, 0)
        box2.show()
        frame_2 = gtk.Frame("Output")
        frame_2.add(box2)
        frame_2.show()
        main_box.pack_start(frame_2, True, True, 10)
        

        scroll_win = gtk.ScrolledWindow()
        scroll_win.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        scroll_win.set_size_request(1000, 100)
        self.textview = gtk.TextView()
        self.textview.set_editable(False)
        self.textview.show()
        scroll_win.add(self.textview)
        scroll_win.show()
        box2.pack_start(scroll_win, False, False, 5)
        
        # Bottom frame
        self.bt_run = gtk.Button("Exécuter l'algorithme", gtk.STOCK_EXECUTE)
        self.bt_run.connect("clicked", self.run, None)
        self.bt_run.set_flags(gtk.CAN_DEFAULT)
        
        self.bt_run.show()
        box3 = gtk.HBox(True, 0)
        box3.pack_start(self.bt_run, True, True, 0)
        box3.show()
        main_box.pack_start(box3, True, True, 0)

        # Placement dans la fenetre principale
        self.mainWindow.add(main_box)
        self.bt_run.grab_default()  #Focus sur le bt par default
        self.mainWindow.show()

    # Main loop
    def main(self):
        gtk.main()
        return 0
    
    # Callback functions
    def run(self, widget, data):
        # on desactive le bt durant le script
        self.bt_run.set_sensitive(False)
    
        # Execution de l'algo avec kppv
        if self.algoType == "kppv":
                    
            log.info("> Run K-PPV with" + str(self.K) + "neigbours...\n")
            
            faceReco = MainConsole.Main( K=self.K, Theta=self.Theta, debug_mode=self.debug_mode )
            # On thread l'app pour le ne pas figer le gui
            t = Thread(target=faceReco.main, args=("KNN", self.textview))
            t.start()# On demarre le thread
            t.join() # On attends la fin du thread
            
            log.info("> Ending")
            
        # Execution de l'algo avec reseau de neurones    
        elif self.algoType == "nnet":
        
            log.info("> Run NNET with \n")
            
            faceReco = MainConsole.Main( n_epoch=self.n_epoch, n_hidden=self.n_hidden, batch_size=self.batch_size,
                                         lr=self.lr, wd=self.wd, debug_mode=self.debug_mode )
            # On thread l'app pour le ne pas figer le gui
            t = Thread(target=faceReco.main, args=("NNET", self.textview))
            t.start()# On demarre le thread
            t.join() # On attends la fin du thread
            
            log.info("> Ending")
        
        # reactivaton du bt
        self.bt_run.set_sensitive(True)
    
    def updateAlgoType(self, widget, algoType):
        self.algoType = algoType
        
        if self.algoType == "kppv":
            self.box_nnet.hide()
            self.box_kppv.show()
        elif self.algoType == "nnet":
            self.box_kppv.hide()
            self.box_nnet.show()
    
    def updateParam(self, widget, data):
        # KNN
        if data == "K_value":
            self.K = widget.get_value_as_int()
        elif data == "Theta_value":
            self.Theta = widget.get_value()
        
        # NNET
        elif data == "Epoch_value":
            self.n_epoch = widget.get_value_as_int()
        elif data == "Hid_value":
            self.n_hidden = widget.get_value_as_int()
        elif data == "Bath_value":
            self.batch_size = widget.get_value_as_int()
        elif data == "Lr_value":
            self.lr = widget.get_value()
        elif data == "Wd_value":
            self.wd = widget.get_value()
        

# Si le script est directement executé (pas importé) on lance l'interface graphique
if __name__ == "__main__":
    from optparse import OptionParser

    # Options du script
    parser = OptionParser()
    parser.set_defaults(verbose=True)
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="print status messages to stdout")
    parser.add_option("-q", "--quiet", action="store_false", dest="verbose", help="don't print status messages to stdout")  
    (opts, args) = parser.parse_args()

    myGui = Gui( debug_mode=opts.verbose )
    myGui.main()
