# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:09:19 2019

@author: Shahi Dost
"""
#External libraries and functions data input/out
import cv2
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from YAGO_texonomy import yago_taxo_fun, image_id_input
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
                
image_id=image_id_input
#==> Read image and save Id of respective image
image = cv2.imread(image_id)
image_id=image_id[85:]
image_id_annotation=image_id[:-4]
out_fun=get_sentence_data('F:/PhD/VKS Flickr30k/Nov-2008/V4/Flickr30k_caption/'+image_id_annotation+'.txt')
def VTKEL_annotations():
    """
    This function stored the meta-data information by using RDF graph triple to stored in .ttl file for VTKEL baseline.
    Input:
    
    Output:
    """
    RDF1_s=URIRef(vtkel1[image_id_annotation+'C0/'])
    RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    RDF1_o=URIRef(ks1['Resource'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_o=URIRef(ks1['Text'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_p=URIRef(ks1['storedAs'])
    RDF1_o=URIRef(vtkel1[image_id_annotation+'C0.txt/'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )    
    
    RDF1_p=URIRef(dct1['identifier'])
    RDF1_o=Literal(image_id_annotation+'C0')
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isPartOf'])
    RDF1_o=URIRef(vtkel1[image_id_annotation])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isString'])
    RDF1_o=Literal(out_fun[0]['sentence'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )

    ##captions C1
    RDF1_s=URIRef(vtkel1[image_id_annotation+'C1/'])
    RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    RDF1_o=URIRef(ks1['Resource'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_o=URIRef(ks1['Text'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_p=URIRef(ks1['storedAs'])
    RDF1_o=URIRef(vtkel1[image_id_annotation+'C1.txt/'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )    
    
    RDF1_p=URIRef(dct1['identifier'])
    RDF1_o=Literal(image_id_annotation+'C1')
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isPartOf'])
    RDF1_o=URIRef(vtkel1[image_id_annotation])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isString'])
    RDF1_o=Literal(out_fun[1]['sentence'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )

    ##captions C2
    RDF1_s=URIRef(vtkel1[image_id_annotation+'C2/'])
    RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    RDF1_o=URIRef(ks1['Resource'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_o=URIRef(ks1['Text'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_p=URIRef(ks1['storedAs'])
    RDF1_o=URIRef(vtkel1[image_id_annotation+'C2.txt/'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )    
    
    RDF1_p=URIRef(dct1['identifier'])
    RDF1_o=Literal(image_id_annotation+'C2')
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isPartOf'])
    RDF1_o=URIRef(vtkel1[image_id_annotation])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isString'])
    RDF1_o=Literal(out_fun[2]['sentence'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )    

    ##captions C3
    RDF1_s=URIRef(vtkel1[image_id_annotation+'C3/'])
    RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    RDF1_o=URIRef(ks1['Resource'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_o=URIRef(ks1['Text'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_p=URIRef(ks1['storedAs'])
    RDF1_o=URIRef(vtkel1[image_id_annotation+'C3.txt/'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )    
    
    RDF1_p=URIRef(dct1['identifier'])
    RDF1_o=Literal(image_id_annotation+'C3')
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isPartOf'])
    RDF1_o=URIRef(vtkel1[image_id_annotation])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isString'])
    RDF1_o=Literal(out_fun[3]['sentence'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )

    ##captions C4
    RDF1_s=URIRef(vtkel1[image_id_annotation+'C4/'])
    RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    RDF1_o=URIRef(ks1['Resource'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_o=URIRef(ks1['Text'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    
    RDF1_p=URIRef(ks1['storedAs'])
    RDF1_o=URIRef(vtkel1[image_id_annotation+'C4.txt/'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )    
    
    RDF1_p=URIRef(dct1['identifier'])
    RDF1_o=Literal(image_id_annotation+'C4')
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isPartOf'])
    RDF1_o=URIRef(vtkel1[image_id_annotation])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )        
    
    RDF1_p=URIRef(dct1['isString'])
    RDF1_o=Literal(out_fun[4]['sentence'])
    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
    return g1
