
#External libraries and functions data input/out
import cv2
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from YAGO_texonomy import yago_taxo_fun
from textblob import TextBlob
from get_sentence_data import *
#from Evaluations_Visual_Texual_Konwledge_v4_2 import g1

vtkel = Namespace('http://vksflickr30k.fbk.eu/resource/')
ks = Namespace('http://dkm.fbk.eu/ontologies/knowledgestore#')
dct = Namespace('http://purl.org/dc/terms/')
rdf = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')

g = Graph()
g.bind("vtkel",vtkel)
g.bind("ks",ks)  
g.bind("dct",dct)              
g.bind("rdf",rdf)
                
def VTKEL_annotations(image_id,image_captions):
    """
    This function takes input image id and stored the meta-data information of document (i.e. an image with 5 captions) in RDF triples.
    Input:
        image_id: image id
        image_captions: dictionary consist of all textual annotations
    
    Output:
        g: meta-data information of an image with 5 captions in RDF triples.
    """
    for i in range(5):
        uri_caption_id=URIRef(vtkel[image_id+'C'+str(i)+'/'])
        g.add( (uri_caption_id, URIRef(rdf['type']), URIRef(ks['Resource'])) )
        g.add( (uri_caption_id, URIRef(rdf['type']), URIRef(ks['Text'])) )
        uri_textual_data=URIRef(vtkel[image_id+'C'+str(i)+'.txt/'])
        g.add( (uri_caption_id, URIRef(ks['storedAs']), uri_textual_data) )  
        g.add( (uri_caption_id, URIRef(dct['identifier']), Literal(image_id+'C'+str(i))) ) 
        g.add( (uri_caption_id, URIRef(dct['isPartOf']), URIRef(vtkel[image_id])) ) 
        g.add( (uri_caption_id, URIRef(dct['isString']), Literal(image_captions[i]['sentence'])) )
    return g
