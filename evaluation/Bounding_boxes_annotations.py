
#External libraries and functions data input/out
import cv2
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from YAGO_texonomy import yago_taxo_fun
from textblob import TextBlob
from get_sentence_data import *
#from Evaluations_Visual_Texual_Konwledge_v4_2 import *

#==> Prefixes to use for RDF graph (triples)
vtkel = Namespace('http://vksflickr30k.fbk.eu/resource/')
ks = Namespace('http://dkm.fbk.eu/ontologies/knowledgestore#')
owl = Namespace('http://www.w3.org/2002/07/owl#')
gaf = Namespace('http://groundedannotationframework.org/gaf#') 
nif = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
prov = Namespace('https://www.w3.org/TR/prov-o/#prov-o-at-a-glance/')    
yago = Namespace('http://dbpedia.org/class/yago/')
rdf = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
                 
g = Graph()
g.bind("vtkel",vtkel)
g.bind("ks",ks)
g.bind("owl",owl)
g.bind("gaf",gaf)
g.bind("nif",nif)
g.bind("prov",prov)
g.bind("yago",yago)
g.bind("rdf",rdf)
#==> End of Prefixes to use for RDF graph (triples)

###==>> Bounding boxes annotations
def Bounding_boxes_annotations(yolo_bb,yolo_class_ids_to_YAGO,yolo_class_ids,image_id):
    """
    This function takes YOLO bounding box(es), class ids, YAGO entity types, image id and stored the visual entities information in RDF triples graph form.
    Input:
        yolo_bb – bounding boxes values detected by YOLO object detector
        yolo_class_ids_to_YAGO – YOLO class ids converted to YAGO type
        yolo_class_ids – yolo class ids e.g. person, racket etc..
    Output:
        g: return RDF graph of visual entities, types and bounding box values
    """
    for i in range(len(yolo_bb)):
            uri_yolo_class_id=URIRef(vtkel[image_id+'I'+'/#'+yolo_class_ids[i]])
            uri_xywh=URIRef(vtkel[str(image_id)+'I'+'/#xywh='+str(yolo_bb[i][0])+','+str(yolo_bb[i][1])+','+str(yolo_bb[i][2])+','+str(yolo_bb[i][3])])
            g.add( (uri_yolo_class_id, URIRef(gaf['denotedBy']), uri_xywh) )
            g.add( (uri_yolo_class_id, URIRef(rdf['type']), URIRef(yago[yolo_class_ids_to_YAGO[i]])) )
            g.add( (uri_xywh, URIRef(rdf['type']), URIRef(ks['VisualEntityMention'])) )
            g.add( (uri_xywh, URIRef(ks['xmin']), Literal(yolo_bb[i][0])) )
            g.add( (uri_xywh, URIRef(ks['ymin']), Literal(yolo_bb[i][1])) )
            g.add( (uri_xywh, URIRef(ks['xmax']), Literal(yolo_bb[i][2])) )
            g.add( (uri_xywh, URIRef(ks['ymax']), Literal(yolo_bb[i][3])) )
            g.add( (uri_xywh, URIRef(prov['wasAttributedTo']), URIRef(vtkel['YOLOAnnotator'])) )
    return g