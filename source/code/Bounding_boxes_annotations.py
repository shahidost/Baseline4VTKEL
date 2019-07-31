# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:03:26 2019

@author: Shahi Dost
"""
#External libraries and functions data input/out
import cv2
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from YAGO_texonomy import yago_taxo_fun
from textblob import TextBlob
from get_sentence_data import *
#from Evaluations_Visual_Texual_Konwledge_v4_2 import *

#==> Prefixes to use for RDF graph (triples)
vtkel1 = Namespace('http://vksflickr30k.fbk.edu/resource/')
ks1 = Namespace('http://dkm.fbk.eu/ontologies/knowledgestore#')
owl1 = Namespace('http://www.w3.org/2002/07/owl#')
gaf1 = Namespace('http://groundedannotationframework.org/gaf#') 
nif1 = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
prov1 = Namespace('https://www.w3.org/TR/prov-o/#prov-o-at-a-glance/')    
yago1 = Namespace('http://dbpedia.org/class/yago/')
rdf1 = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
                 
g1 = Graph()
g1.bind("vtkel",vtkel1)
g1.bind("ks",ks1)
g1.bind("owl",owl1)
g1.bind("gaf",gaf1)
g1.bind("nif",nif1)
g1.bind("prov",prov1)
g1.bind("yago",yago1)
g1.bind("rdf",rdf1)
#==> End of Prefixes to use for RDF graph (triples)

###==>> Bounding boxes annotations
def Bounding_boxes_annotations(bounding_boxes,YOLO_class_in_YAGO,YOLO_class_names,image_id_annotation):
    """
    This function takes bounding boxes information from YOLO system(connected with YAGO ontology) and stored in a RDF triple graph to store in .ttl file.
    Input:
        bounding_boxes – all bounding boxes detected by YOLO system
        YOLO_class_in_YAGO – bounding boxes in YAGO classes
        YOLO_class_names – YOLO classes name
    Output:
        g1:return bounding box(es) annotations in RDF graphs in .ttl form
    """
    for i in range(len(YOLO_class_names)):
            RDF3_s=URIRef(vtkel1[image_id_annotation+'I'+'/#'+YOLO_class_names[i]])
            RDF3_p=URIRef(gaf1['denotedBy'])
            xywh=str(bounding_boxes[i])+','+str(bounding_boxes[i+1])+','+str(bounding_boxes[i+2])+','+str(bounding_boxes[i+3])
            xywh=str(image_id_annotation)+'I'+'/#xywh='+xywh
            RDF3_o=URIRef(vtkel1[xywh])
            g1.add( (RDF3_s, RDF3_p, RDF3_o) )

            RDF4_s=RDF3_s
            RDF4_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            RDF4_o=URIRef(yago1[YOLO_class_in_YAGO[i]])
            g1.add( (RDF4_s, RDF4_p, RDF4_o) )

            RDF5_s=RDF3_o
            RDF5_p=URIRef(rdf1['type'])
            RDF5_o=URIRef(ks1['VisualEntityMention'])
            g1.add( (RDF5_s, RDF5_p, RDF5_o) )

            RDF6_s=RDF3_o
            RDF6_p=URIRef(ks1['xmin'])
            RDF6_o=Literal(bounding_boxes[i])
            g1.add( (RDF6_s, RDF6_p, RDF6_o) )

            RDF7_s=RDF3_o
            RDF7_p=URIRef(ks1['ymin'])
            RDF7_o=Literal(bounding_boxes[i+1])
            g1.add( (RDF7_s, RDF7_p, RDF7_o) )

            RDF8_s=RDF3_o
            RDF8_p=URIRef(ks1['xmax'])
            RDF8_o=Literal(bounding_boxes[i+2])
            g1.add( (RDF8_s, RDF8_p, RDF8_o) )

            RDF9_s=RDF3_o
            RDF9_p=URIRef(ks1['ymax'])
            RDF9_o=Literal(bounding_boxes[i+3])
            g1.add( (RDF9_s, RDF9_p, RDF9_o) )

            RDF10_s=RDF3_s
            RDF10_p=URIRef(prov1['wasAttributedTo'])
            RDF10_o=URIRef(vtkel1['YOLOAnnotator'])
            g1.add( (RDF10_s, RDF10_p, RDF10_o) )
            i=i+4
    return g1