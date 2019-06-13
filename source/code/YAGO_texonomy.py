# -*- coding: utf-8 -*-
"""
Created on Monday June 10 11:19:47 2019

@author: Shahi Dost

YAGO_texonomy.py

This script is used to upload the original file of YAGO class taxonomy hieratchy file. [] 


"""

import rdflib
#from speed_issue import rdffile3

def yago_taxo_fun():
    """
    This function uploads the YAGO taxonomy .ttl file for mapping to main code
    input:
        
    output:
        g4 - YAGO parsed turtle file 
    """
#    rdffile3="F:/PhD/VKS Flickr30k/Nov-2008/V4/VTKEL and Flickr30k annotations/script/yago_test_empy.ttl"
    rdffile3="C:/Users/aa/Desktop/YOLO/object-detection-opencv-master/yago_taxonomy-v1.1.ttl"
    g3=rdflib.ConjunctiveGraph()
    g4=g3.parse(rdffile3, format="turtle")
    return g4

image_id_input='F:/PhD/VKS Flickr30k/Nov-2008/V4/VTKEL and Flickr30k annotations/script/input images/23008340.jpg'
