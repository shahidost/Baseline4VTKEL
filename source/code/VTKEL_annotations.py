# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:09:19 2019

@author: Shahi Dost
"""
#External libraries and functions data input/out
import cv2
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from YAGO_texonomy import yago_taxo_fun
from textblob import TextBlob
from get_sentence_data import *
#from Evaluations_Visual_Texual_Konwledge_v4_2 import g1

vtkel1 = Namespace('http://vksflickr30k.fbk.edu/resource/')
ks1 = Namespace('http://dkm.fbk.eu/ontologies/knowledgestore#')
dct1 = Namespace('http://purl.org/dc/terms/')

g1 = Graph()
g1.bind("vtkel",vtkel1)
g1.bind("ks",ks1)  
g1.bind("dct",dct1)              
                
def VTKEL_annotations(image_id_annotation):
    """
    This function stored the meta-data information by using RDF graph triple to stored in .ttl file for VTKEL baseline.
    Input:
    
    Output:
    """
    out_fun=get_sentence_data('F:/PhD/VKS Flickr30k/Nov-2008/V4/Flickr30k_caption/'+image_id_annotation+'.txt')
    for i in range(5):
        RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(i)+'/'])
        RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        RDF1_o=URIRef(ks1['Resource'])
        g1.add( (RDF1_s, RDF1_p, RDF1_o) )
        
        RDF1_o=URIRef(ks1['Text'])
        g1.add( (RDF1_s, RDF1_p, RDF1_o) )
        
        RDF1_p=URIRef(ks1['storedAs'])
        RDF1_o=URIRef(vtkel1[image_id_annotation+'C'+str(i)+'.txt/'])
        g1.add( (RDF1_s, RDF1_p, RDF1_o) )  
        
        RDF1_p=URIRef(dct1['identifier'])
        RDF1_o=Literal(image_id_annotation+'C'+str(i))
        g1.add( (RDF1_s, RDF1_p, RDF1_o) ) 
        
        RDF1_p=URIRef(dct1['isPartOf'])
        RDF1_o=URIRef(vtkel1[image_id_annotation])
        g1.add( (RDF1_s, RDF1_p, RDF1_o) ) 
        
        RDF1_p=URIRef(dct1['isString'])
        RDF1_o=Literal(out_fun[i]['sentence'])
        g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    return g1
