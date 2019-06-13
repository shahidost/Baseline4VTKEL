# -*- coding: utf-8 -*-
"""
Created on Monday June 10 11:19:47 2019

@author: Shahi Dost

main_source_code.py

This scripts is []...... 


"""

#External libraries and functions data input/out
import requests
import urllib.request
from rdflib import ConjunctiveGraph
import os
import rdflib
import cv2
import argparse
import numpy as np
from textblob import TextBlob
from rdflib import Graph, URIRef, BNode, Literal, Namespace
import datetime
from YAGO_texonomy import yago_taxo_fun, image_id_input
from get_sentence_data import *
from YOLO_classes_to_YAGO import *
from duplicate_visual_mentions import *
from removed_duplicate_mentions import *
from Coreference_rdfTripes_from_PIKES import *
from Bounding_boxes_annotations import *
from VTKEL_annotations import *
import time
from collections import Counter

#==> Prefixes to use for RDF graph (triples)
g1 = Graph()
dc1 = Namespace('http://purl.org/dc/elements/1.1#')
dct1 = Namespace('http://purl.org/dc/terms/')
gaf1 = Namespace('http://groundedannotationframework.org/gaf#')
ks1 = Namespace('http://dkm.fbk.eu/ontologies/knowledgestore#')
nfo1 = Namespace('http://oscaf.sourceforge.net/')
nif1 = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
owl1 = Namespace('http://www.w3.org/2002/07/owl#')
prov1 = Namespace('https://www.w3.org/TR/prov-o/#prov-o-at-a-glance/')
rdf1 = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
rdfs1 = Namespace('http://www.w3.org/2000/01/rdf-schema#')
vtkel1 = Namespace('http://vksflickr30k.fbk.edu/resource/')
xml1 = Namespace('http://www.w3.org/XML/1998/namespace')
xsd1 = Namespace('http://www.w3.org/2001/XMLSchema#')
yago1 = Namespace('http://dbpedia.org/class/yago/')

g1.bind("dc",dc1)
g1.bind("dct",dct1)
g1.bind("gaf",gaf1)
g1.bind("ks",ks1)
g1.bind("nfo",nfo1)
g1.bind("nif",nif1)
g1.bind("owl",owl1)
g1.bind("prov",prov1)
g1.bind("rdf",rdf1)
g1.bind("rdfs",rdfs1)
g1.bind("vtkel",vtkel1)
g1.bind("xml",xml1)
g1.bind("xsd",xsd1)
g1.bind("yago",yago1)

#==> End of Prefixes to use for RDF graph (triples)

#==> Read input image file
image_id=image_id_input
#==> End of Read input image file

#==> Read image and save Id of respective image
image = cv2.imread(image_id)
image_id=image_id[85:]

#==> End of Read image and save Id of respective image

def read_glove_vecs(glove_file):
    """
    This function to upload the glove file for word embedding
    input:
        glove_file - GLOVE file to be upload from your system
    
    output:
        words - array vectors for words
        word_to_vec_map - word to vector map
    """
    with open(glove_file, 'r',encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map
words, word_to_vec_map = read_glove_vecs('C:/Users/aa/Desktop/YOLO/word2vec/data/glove.6B.50d.txt')

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)
    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    ### START CODE HERE ###
    # Compute the dot product between u and v (≈1 line)

    cosine_similarity=0
    if len(v)>0:
        dot = np.dot(u.T, v)
        # Compute the L2 norm of u (≈1 line)
        norm_u = np.sqrt(np.sum(np.power(u, 2)))
    
        # Compute the L2 norm of v (≈1 line)
        norm_v = np.sqrt(np.sum(np.power(v, 2)))
        # Compute the cosine similarity defined by formula (1) (≈1 line)
        cosine_similarity = np.divide(dot, norm_u * norm_v)
        ### END CODE HERE ###

    return cosine_similarity

#==> read the caption files with respect to Image Id
image_id_annotation=image_id[:-4]
out_fun=get_sentence_data('insert image caption file .txt from Flickr30k entities datasets->captions files')
out_fun=out_fun
#==> to test the results
print('----------------------------------------------------\nImage captions processing....\n')
print('C0:',out_fun[0]['sentence'])
print('C1:',out_fun[1]['sentence'])
print('C2:',out_fun[2]['sentence'])
print('C3:',out_fun[3]['sentence'])
print('C4:',out_fun[4]['sentence'])


###====> PIKES System tool Phase
print('\n-------------------------------------------------\nPIKES processing....')
### connect to 'PIKES server' for knowledge graph in RDF Trig format
PUBLIC_PIKES_SERVER = 'https://knowledgestore2.fbk.eu/pikes-demo/api/'
LOCAL_PIKES_SERVER = 'http://localhost:8011/'

def pikes_text2rdf(x):
    """
    Takes a input natural language sentence and passed through ‘PIKES server’ for knowledge graph extraction 
    input:
      x – input natural language text
    
    output:
      .ttl file – a turtle RDF format output file,  which stored the knowledge graph of natural language in Triples form

    """
    return requests.get(PUBLIC_PIKES_SERVER+"text2rdf?",{'text':x})

def get_entities_from_text(text):
    """
    This function process the RDF graph processed by PIKES and extract for textual entity mentions recognized and linked by PIKES from natural language.  First this function takes input natural language text (i.e. image captions from Flickr30k dataset) and passed through PIKES (i.e. calling function pikes_text2rdf()). After extracting RDF graph, the textual entities are recognized and linked from knowledgebase using Sparql query.  
    input:
      x –  Image caption of from Flickr30k entities dataset
    
    output:
     sparql query result  – which stored all the entities recognized and linked by PIKES

    """
    pikes_answer = pikes_text2rdf(text.lower())
    
    g = ConjunctiveGraph()
    g.parse(data = pikes_answer.content.decode('utf-8'),format="trig")
    sparql_query = """SELECT ?s ?o
           WHERE {
           GRAPH ?gg {?s a ks:Entity}
           GRAPH ?g {?s rdf:type ?o}
           }"""

    return g.query(sparql_query)

def PIKES_entities():
    print('\n-------------------------------------------------\nPIKES entities processing....')
    """
    This function extract and stored all the entities from PIKES in caption base (five captions for one image).  
    input:
      no input
    
    output:
    pikes_mention_c0 – entity mentions of first caption
    pikes_mention_c1 – entity mentions of second caption
    pikes_mention_c2 – entity mentions of third caption
    pikes_mention_c3 – entity mentions of fourth caption
    pikes_mention_c4 – entity mentions of fifth
    YAGO_class_c0 – YAGO classes of first caption
    YAGO_class_c1 – YAGO classes of second caption
    YAGO_class_c2 – YAGO classes of third caption
    YAGO_class_c3 – YAGO classes of fourth caption
    YAGO_class_c4 – YAGO classes of fifth caption
    """    
    pikes_mention_c0=[]
    pikes_mention_c1=[]
    pikes_mention_c2=[]
    pikes_mention_c3=[]
    pikes_mention_c4=[]
    YAGO_class_c0=[]
    YAGO_class_c1=[]
    YAGO_class_c2=[]
    YAGO_class_c3=[]
    YAGO_class_c4=[]
    for i in range(5):
        caption=out_fun[i]['sentence']
        caption_entities = get_entities_from_text(caption)
        for row in caption_entities:
            if 'http://dbpedia.org/class/yago/' in row[1] and i==0 and 'http://www.newsreader-project.eu/time/P1D' not in row[0]:
                pikes_mention_c0.append(row[0][21:])
                YAGO_class_c0.append(row[1])
            elif 'http://dbpedia.org/class/yago/' in row[1] and i==1 and 'http://www.newsreader-project.eu/time/P1D' not in row[0]:
                pikes_mention_c1.append(row[0][21:])
                YAGO_class_c1.append(row[1])
            elif 'http://dbpedia.org/class/yago/' in row[1] and i==2 and 'http://www.newsreader-project.eu/time/P1D' not in row[0]:
                pikes_mention_c2.append(row[0][21:])
                YAGO_class_c2.append(row[1])
            elif 'http://dbpedia.org/class/yago/' in row[1] and i==3 and 'http://www.newsreader-project.eu/time/P1D' not in row[0]:
                pikes_mention_c3.append(row[0][21:])
                YAGO_class_c3.append(row[1])
            elif 'http://dbpedia.org/class/yago/' in row[1] and i==4 and 'http://www.newsreader-project.eu/time/P1D' not in row[0]:
                pikes_mention_c4.append(row[0][21:])
                YAGO_class_c4.append(row[1])
        for row in caption_entities:
            if 'http://groundedannotationframework.org/gaf#denotedBy' in row[1] and i==0:
                print('...')
    return pikes_mention_c0,pikes_mention_c1,pikes_mention_c2,pikes_mention_c3,pikes_mention_c4,YAGO_class_c0,YAGO_class_c1,YAGO_class_c2,YAGO_class_c3,YAGO_class_c4
pikes_mention_c0,pikes_mention_c1,pikes_mention_c2,pikes_mention_c3,pikes_mention_c4,YAGO_class_c0,YAGO_class_c1,YAGO_class_c2,YAGO_class_c3,YAGO_class_c4=PIKES_entities()

print('\n-------------------------------------------------\nAlignment processing....')
def allignment_Flickr30k_PIKES(caption_c0,pikes_mention_c0,YAGO_class_c0):
    """
    This function find alignment between entities mentions of Flickr30k entities dataset caption(s) and PIKES extracted textual entities mentions. 
    input:
      caption_c0 – Image caption (natural language text)
     pikes_mention_c0 – PIKES entity mentions
    YAGO_class_c0 – Respective YAGO classes

    output:
    mention_align_pikes – array of aling textual mentions between Flickr30k and PIKES
    mention_align_flickr – Flickr30k entities mention 
    mention_align_YAGO – YAGO align classes
    """ 
    caption_c0=caption_c0.lower()
    caption_c0=TextBlob(caption_c0)
    caption_c0_words_list=caption_c0.words
    curr_similarity=0
    min_similarity=-100
    mention_align_pikes=[]
    mention_align_flickr=[]
    mention_align_YAGO=[]
    second_word=''
    flag=True
    for j in range(len(pikes_mention_c0)):
        min_similarity=-100
        for i in range(len(caption_c0_words_list)):
            flag=True
            for ch in pikes_mention_c0[j]:
                if ch.isdigit()==True:
                    flag=False
            if flag==False:
                pikes_mention_c0[j]=pikes_mention_c0[j][:]
            if len(caption_c0_words_list[i])>1:
                if pikes_mention_c0[j][-2:]!='_2':
                    if pikes_mention_c0[j][-2:]=='_3':
                        first_word = word_to_vec_map[pikes_mention_c0[j][:-2]]
                    else:
                        first_word = word_to_vec_map[pikes_mention_c0[j]]
                    try:
                        second_word = word_to_vec_map[caption_c0_words_list[i]]
                    except:
                        print('Error:',caption_c0_words_list[i])
                    curr_similarity=cosine_similarity(first_word, second_word)
                    if curr_similarity>=min_similarity:
                        min_similarity=curr_similarity
                        pikes_word=pikes_mention_c0[j]
                        YAGO_word=YAGO_class_c0[j]
                        flickr_word=caption_c0_words_list[i]
                elif pikes_mention_c0[j][-2:]=='_2':
                    first_word = word_to_vec_map[pikes_mention_c0[j][:-2]]
                    try:
                        second_word = word_to_vec_map[caption_c0_words_list[i]]
                    except:
                        print('Error:',caption_c0_words_list[i])
                    curr_similarity=cosine_similarity(first_word, second_word)
                    if curr_similarity>=min_similarity:
                        min_similarity=curr_similarity
                        pikes_word=pikes_mention_c0[j]
                        YAGO_word=YAGO_class_c0[j]
                        flickr_word=caption_c0_words_list[i]
        mention_align_pikes.append(pikes_word)
        mention_align_flickr.append(flickr_word)
        mention_align_YAGO.append(YAGO_word)
    return mention_align_pikes,mention_align_flickr,mention_align_YAGO
mention_align_pikes_c0,mention_align_flickr_c0,mention_align_YAGO_c0=allignment_Flickr30k_PIKES(out_fun[0]['sentence'],pikes_mention_c0,YAGO_class_c0)
mention_align_pikes_c1,mention_align_flickr_c1,mention_align_YAGO_c1=allignment_Flickr30k_PIKES(out_fun[1]['sentence'],pikes_mention_c1,YAGO_class_c1)
mention_align_pikes_c2,mention_align_flickr_c2,mention_align_YAGO_c2=allignment_Flickr30k_PIKES(out_fun[2]['sentence'],pikes_mention_c2,YAGO_class_c2)
mention_align_pikes_c3,mention_align_flickr_c3,mention_align_YAGO_c3=allignment_Flickr30k_PIKES(out_fun[3]['sentence'],pikes_mention_c3,YAGO_class_c3)
mention_align_pikes_c4,mention_align_flickr_c4,mention_align_YAGO_c4=allignment_Flickr30k_PIKES(out_fun[4]['sentence'],pikes_mention_c4,YAGO_class_c4)

print('\n-------------------------------------------------\nYAGO mapping....')
def Mapping_of_YAGO(g3,word1,word2):
    """
    This function takes two YAGO classes (Woman110787470, Person100007846) and find if they are in sub-class or same class hierarchy by mapping on YAGO taxonomy file.  
    input:
      g3 – Sparql query to store the YAGO taxonomy
    word1 – First class
    word2 – Second class
    output:
    success_flag – binary flag for hierarchy condition (true of same or sub-class success otherwise false)
    mapping_hierarchy – the subclass hierarchy between two class extracted from YAGO taxonomy and stored in an array.
    """

    word1=str('http://dbpedia.org/class/yago/'+word1)
    word2=str('http://dbpedia.org/class/yago/'+word2)    
    mapping_hierarchy=[]
    end=time.time()
    success_flag=False
    taxonomy_loop_counter=0
    temp_word_o=word1
    while success_flag!=True and taxonomy_loop_counter<3:
        taxonomy_loop_counter+=1
        if word1==word2:
            success_flag=True
            mapping_hierarchy.append(word1)
            break
        elif success_flag==False:
            temp_word_s=word1
            for s,p,o in g3:
                if (temp_word_s == s[:]) and (word2==o[:]) and (success_flag==False):
                    success_flag=True
                    mapping_hierarchy.append(temp_word_s)
                    break
                elif (temp_word_o == s[:]) and (success_flag==False):
                    mapping_hierarchy.append(s[:])
                    temp_word_s=s[:]
                    temp_word_o=o[:]
                    if temp_word_o==word2 and success_flag==False:
                        success_flag=True
                        mapping_hierarchy.append(temp_word_o)
                        break
            if success_flag==False and taxonomy_loop_counter==3:
                break
    return success_flag,mapping_hierarchy
start = time.time()
total_visual_mentions=[]
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

"""
Pre-train YOLO version 3 model for detecting visual objects.
"""    
# read class names from text file
classes = None
with open('insert file yolov3.txt from YOLO folder', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet('insert file yolov3_.weights from YOLO', 'insert yolov3.cfg from YOLO')

# create input blob
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
net.setInput(blob)

def get_output_layers(net):
    """
    Pre-train YOLO Neural Network layer for object detections, 
    input: 
    net - A neural network layer
    
    Output:
    output_layers - Features for output layer
    """    

    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
class_names=[]
bounding_boxes=[]
YOLO_class_names=[]
print('\n-------------------------------------------------\nYOLO objects detection processing....')
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Pre-train YOLO version 3 model for detecting visual objects.
    """    
    """
        This function takes x, y, w, h, image.jpg and image id received from YOLO model and draw the bounding boxes at respective image.  
    input:
        img – jpg image from Flickr30k entities dataset
        class_id – class id from YOLO pre-trained model
        confidence – YOLO object detected condifence score (if score is >= 0.5 IUO object detected)
        x – left top corner of bounding box (in x-axis)
        y – left to corner of bounding box (in y-axis)
        x_plus_w – width of bounding box
        y_plus_h – height of bounding box
    Output:
        
    """

    if x<0:
        x=0
    if y<0:
        y=0
    if x_plus_w<0:
        x_plus_w=0
    if y_plus_h<0:
        y_plus_h=0
    label = str(classes[class_id])
    class_names.append(label)
    YOLO_class_names.append(label)
    class_names.append(class_id)
    class_names.append(confidence)
    class_names.append(x)
    class_names.append(y)
    class_names.append(x_plus_w)
    class_names.append(y_plus_h)
    bounding_boxes.append(x)
    bounding_boxes.append(y)
    bounding_boxes.append(x_plus_w)
    bounding_boxes.append(y_plus_h)
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
outs = net.forward(get_output_layers(net))

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.3
#Non-maximum suppression (NMS)
nms_threshold = 0.4

# for each detetion from each output layer 
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])          
# apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
bb_count=0
for i in indices:
    bb_count+=1
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

YOLO_class_in_YAGO=YOLO_classes_to_YAGO(YOLO_class_names)   


def Allignment_YOLO_PIKES(caption_no,mention_id,total_number_mentions,qres1,YOLO_class_names,YOLO_class_in_YAGO,YAGO_class_c0,mention_align_pikes,mention_align_flickr,total_visual_mentions):
    """
    This function takes all the visual mentions detected by YOLO system and textual mentions detected by PIKES tool to make alignment between visual and textual mentions with respect to their entity mention in YAGO taxonomy. The function mapped YAGO taxonomy by processing two classes (one visual objects base class and second textual object base class) and mapped through YAGO taxonomy for same-class or sub-class hierarchy detection. If they are same or sub-class aligned otherwise try for next case.    
    input:
        caption_no – caption number (0 for first, 1 for first till fifth caption)
        mention_id – mention id detected by PIKES tool 
        total_number_mentions – Textual entity list detected by PIKES tool
        qres1 – Sparql query for PIKES entity 
        YOLO_class_names – Visual objects list detected by YOLO system
        YOLO_class_in_YAGO – YOLO visual objects linked to YAGO classes
        YAGO_class_c0 – YAGO class caption wise
        mention_align_pikes – alignment of textual mention between PIKES and Flickr30k base
        mention_align_flickr – Flickr30k entities dataset entity mentions
        total_visual_mentions – total visual mention of one image 

     Output:
        mention_id – ids of visual-textual aligned entity mentions
        total_visual_mentions – all aligned textual mentions
    """
    
    mapping_result=False
    for i in range(len(YOLO_class_in_YAGO)):
        word1=YOLO_class_in_YAGO[i]
        visual_mention=YOLO_class_names[i]
        same_visual_mention_count=0

        for j in range(len(YAGO_class_c0)):
            word2=YAGO_class_c0[j][30:]
            mapping_hierarchy=[]
            mapping_result,mapping_hierarchy=Mapping_of_YAGO(qres1,word1,word2)
            if mapping_result==True:
                same_visual_mention_count+=1
                mention_id=i
                if word1!=word2:
                    RDF1_s=URIRef(yago1[word1])
                    RDF1_p=URIRef(rdfs1['subClassOf'])
                    RDF1_o=URIRef(yago1[word2])
                    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                if same_visual_mention_count==1:
                    RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                    RDF1_p=URIRef(ks1['shownIn'])
                    RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                    g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                elif same_visual_mention_count==2:
                    for k in range(len(YOLO_class_names)):
                        if YOLO_class_names[k][-2:]=='_2':
                            visual_mention=YOLO_class_names[k]
                            RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                            RDF1_p=URIRef(ks1['shownIn'])
                            RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                            g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                        elif word1==word2:
                            RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                            RDF1_p=URIRef(ks1['shownIn'])
                            RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                            g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                elif same_visual_mention_count==3:
                    for k in range(len(YOLO_class_names)):
                        if YOLO_class_names[k][-2:]=='_2':
                            visual_mention=YOLO_class_names[k]
                            RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                            RDF1_p=URIRef(ks1['shownIn'])
                            RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                            g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                        elif word1==word2:
                            RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                            RDF1_p=URIRef(ks1['shownIn'])
                            RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                            g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                elif same_visual_mention_count==4:
                    for k in range(len(YOLO_class_names)):
                        if YOLO_class_names[k][-2:]=='_2':
                            visual_mention=YOLO_class_names[k]
                            RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                            RDF1_p=URIRef(ks1['shownIn'])
                            RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                            g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                        elif word1==word2:
                            RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                            RDF1_p=URIRef(ks1['shownIn'])
                            RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                            g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                                                 
                    
                total_visual_mentions.append(mention_align_pikes[j])
                RDF2_s=RDF1_o
                RDF2_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                RDF2_o=URIRef(YAGO_class_c0[j])
                g1.add( (RDF2_s, RDF2_p, RDF2_o) )

                start_index=0
                start_index=out_fun[caption_no]['sentence'].find(mention_align_flickr[j],start_index)
                end_index=len(mention_align_flickr[j])+start_index
                RDF4_s=RDF2_s
                RDF4_p=URIRef(gaf1['denotedBy'])
                RDF4_o=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)+'/#char='+str(start_index)+','+str(end_index)])
                g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                
                RDF5_s=RDF4_o
                RDF5_p=URIRef(rdf1['type'])
                RDF5_o=URIRef(ks1['TextualEntityMention'])
                g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                
                RDF6_s=RDF4_o
                RDF6_p=URIRef(ks1['mentionOf'])
                RDF6_o=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)+'/'])
                g1.add( (RDF6_s, RDF6_p, RDF6_o) )

                RDF7_s=RDF4_o
                RDF7_p=URIRef(nif1['anchorOf'])
                RDF7_o=Literal(mention_align_pikes[j])

                RDF8_s=RDF4_o
                RDF8_p=URIRef(nif1['beginIndex'])
                RDF8_o=Literal(start_index)

                RDF9_s=RDF4_o
                RDF9_p=URIRef(nif1['endIndex'])
                RDF9_o=Literal(end_index)

                RDF10_s=RDF4_o
                RDF10_p=URIRef(prov1['wasAttributedTo'])
                RDF10_o=URIRef(vtkel1['PikesAnnotator'])
                g1.add( (RDF10_s, RDF10_p, RDF10_o) )                
                

            elif mapping_result==False:
                mapping_result,mapping_hierarchy=Mapping_of_YAGO(qres1,word2,word1)
                if mapping_result==True:
                    mention_id=i
                    same_visual_mention_count+=1
                    if word1!=word2:
                        RDF1_s=URIRef(yago1[word2])
                        RDF1_p=URIRef(rdfs1['subClassOf'])
                        RDF1_o=URIRef(yago1[word1])
                        g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                    if same_visual_mention_count==1:
                        RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                        RDF1_p=URIRef(ks1['shownIn'])
                        RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                        g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                    elif same_visual_mention_count==2:
                        for k in range(len(YOLO_class_names)):
                            if YOLO_class_names[k][-2:]=='_2':
                                visual_mention=YOLO_class_names[k]
                                RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                                RDF1_p=URIRef(ks1['shownIn'])
                                RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                            elif word1==word2:
                                RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                                RDF1_p=URIRef(ks1['shownIn'])
                                RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                    elif same_visual_mention_count==3:
                        for k in range(len(YOLO_class_names)):
                            if YOLO_class_names[k][-2:]=='_2':
                                visual_mention=YOLO_class_names[k]
                                RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                                RDF1_p=URIRef(ks1['shownIn'])
                                RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                            elif word1==word2:
                                RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                                RDF1_p=URIRef(ks1['shownIn'])
                                RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                    elif same_visual_mention_count==4:
                        for k in range(len(YOLO_class_names)):
                            if YOLO_class_names[k][-2:]=='_2':
                                visual_mention=YOLO_class_names[k]
                                RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                                RDF1_p=URIRef(ks1['shownIn'])
                                RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                            elif word1==word2:
                                RDF1_s=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)]+'/#'+mention_align_pikes[j])
                                RDF1_p=URIRef(ks1['shownIn'])
                                RDF1_o=URIRef(vtkel1[image_id_annotation+'I'+'/#'+visual_mention])
                                g1.add( (RDF1_s, RDF1_p, RDF1_o) )

                    total_visual_mentions.append(mention_align_pikes[j])
                    RDF2_s=RDF1_o
                    RDF2_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    RDF2_o=URIRef(YAGO_class_c0[j])
                    g1.add( (RDF2_s, RDF2_p, RDF2_o) )
                    start_index=0
                    start_index=out_fun[caption_no]['sentence'].find(mention_align_flickr[j],start_index)
                    end_index=len(mention_align_flickr[j])+start_index
                    RDF4_s=RDF2_s
                    RDF4_p=URIRef(gaf1['denotedBy'])
                    RDF4_o=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)+'/#char='+str(start_index)+','+str(end_index)])
                    g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                    
                    RDF5_s=RDF4_o
                    RDF5_p=URIRef(rdf1['type'])
                    RDF5_o=URIRef(ks1['TextualEntityMention'])
                    g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                    
                    RDF6_s=RDF4_o
                    RDF6_p=URIRef(ks1['mentionOf'])
                    RDF6_o=URIRef(vtkel1[image_id_annotation+'C'+str(caption_no)+'/'])
                    g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                    RDF7_s=RDF4_o
                    RDF7_p=URIRef(nif1['anchorOf'])
                    RDF7_o=Literal(mention_align_pikes[j])
#                    g1.add( (RDF7_s, RDF7_p, RDF7_o) )
    
                    RDF8_s=RDF4_o
                    RDF8_p=URIRef(nif1['beginIndex'])
                    RDF8_o=Literal(start_index)
#                    g1.add( (RDF8_s, RDF8_p, RDF8_o) )
    
                    RDF9_s=RDF4_o
                    RDF9_p=URIRef(nif1['endIndex'])
                    RDF9_o=Literal(end_index)
#                    g1.add( (RDF9_s, RDF9_p, RDF9_o) )
    
                    RDF10_s=RDF4_o
                    RDF10_p=URIRef(prov1['wasAttributedTo'])
                    RDF10_o=URIRef(vtkel1['PikesAnnotator'])
                    g1.add( (RDF10_s, RDF10_p, RDF10_o) )                    
    return mention_id,total_visual_mentions

#==> Stored all textual entity mentions from five captions
total_mentions=[]
total_mentions_YAGO=[]
for i in range(len(mention_align_pikes_c0)):
    total_mentions.append(mention_align_pikes_c0[i])
    total_mentions_YAGO.append(mention_align_YAGO_c0[i])
for i in range(len(mention_align_pikes_c1)):
    total_mentions.append(mention_align_pikes_c1[i])
    total_mentions_YAGO.append(mention_align_YAGO_c1[i])
for i in range(len(mention_align_pikes_c2)):
    total_mentions.append(mention_align_pikes_c2[i])
    total_mentions_YAGO.append(mention_align_YAGO_c2[i])
for i in range(len(mention_align_pikes_c3)):
    total_mentions.append(mention_align_pikes_c3[i])
    total_mentions_YAGO.append(mention_align_YAGO_c3[i])
for i in range(len(mention_align_pikes_c4)):
    total_mentions.append(mention_align_pikes_c4[i])
    total_mentions_YAGO.append(mention_align_YAGO_c4[i])
b = set()
total_number_mentions = []
total_mentions_YAGO_class = []
index_yago=0
for x in total_mentions:
    if x not in b:
        total_number_mentions.append(x)
        total_mentions_YAGO_class.append(total_mentions_YAGO[index_yago])
        b.add(x)
        index_yago+=1
    else:
        index_yago+=1
print('\n-------------------------------------------------\nCoreference processing....')
rdf_mentions_pikes,rdf_mentions_yago=removed_duplicate_mentions(total_number_mentions,total_mentions_YAGO_class)

g4,total_number_mentions,total_mentions_YAGO_class=Coreference_rdfTripes_from_PIKES(YOLO_class_names,YOLO_class_in_YAGO,total_number_mentions,total_mentions_YAGO_class,mention_align_pikes_c0,mention_align_pikes_c1,mention_align_pikes_c2,mention_align_pikes_c3,mention_align_pikes_c4)
g1=g1+g4
#=> YOLO class merging
b = set()
unique_YOLO_class_in_YAGO = []
for x in YOLO_class_in_YAGO:
    if x not in b:
        unique_YOLO_class_in_YAGO.append(x)
        b.add(x)

end=time.time()
flag_for_yago=False
if flag_for_yago==False:
    flag_for_yago=True
    g4=yago_taxo_fun()
    qres2=g4
mention_id=0
end=time.time()
#print('time after YAGO file loading->',end-start)
#####checking time
#end=time.time()
#print('time before YAGO mapping->',end-start)

                        
YOLO_class_names_unique=duplicate_visual_mentions(YOLO_class_names)
g3=Bounding_boxes_annotations(bounding_boxes,YOLO_class_in_YAGO,YOLO_class_names_unique)
g1=g1+g3
mention_id,total_visual_mentions=Allignment_YOLO_PIKES(0,mention_id,total_number_mentions,qres2,YOLO_class_names_unique,unique_YOLO_class_in_YAGO,YAGO_class_c0,mention_align_pikes_c0,mention_align_flickr_c0,total_visual_mentions)
mention_id,total_visual_mentions=Allignment_YOLO_PIKES(1,mention_id,total_number_mentions,qres2,YOLO_class_names_unique,unique_YOLO_class_in_YAGO,YAGO_class_c1,mention_align_pikes_c1,mention_align_flickr_c1,total_visual_mentions)
mention_id,total_visual_mentions=Allignment_YOLO_PIKES(2,mention_id,total_number_mentions,qres2,YOLO_class_names_unique,unique_YOLO_class_in_YAGO,YAGO_class_c2,mention_align_pikes_c2,mention_align_flickr_c2,total_visual_mentions)
mention_id,total_visual_mentions=Allignment_YOLO_PIKES(3,mention_id,total_number_mentions,qres2,YOLO_class_names_unique,unique_YOLO_class_in_YAGO,YAGO_class_c3,mention_align_pikes_c3,mention_align_flickr_c3,total_visual_mentions)
mention_id,total_visual_mentions=Allignment_YOLO_PIKES(4,mention_id,total_number_mentions,qres2,YOLO_class_names_unique,unique_YOLO_class_in_YAGO,YAGO_class_c4,mention_align_pikes_c4,mention_align_flickr_c4,total_visual_mentions)
RDF1_s=URIRef(vtkel1)

#==> Stored meta-data information in RDF graph for VTKEL annotations instantiations
RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDF1_o=URIRef("http://purl.org/dc/dcmitype/Software")
g1.add( (RDF1_s, RDF1_p, RDF1_o) )

RDF110_p=URIRef("http://purl.org/dc/terms/creator")
RDF110_o=Literal("Shahi Dost & Luciano Serafini")
g1.add( (RDF1_s, RDF110_p, RDF110_o) )

t= datetime.datetime.now()
RDF120_p=URIRef("http://purl.org/dc/terms/created")
RDF120_o=Literal( str(t.year)+'-'+str(t.month)+'-'+str(t.day)+':'+'-'+str(t.hour)+':'+str(t.minute)+':'+str(t.second))
g1.add( (RDF1_s, RDF120_p, RDF120_o) )

RDF130_p=URIRef("http://purl.org/dc/terms/language")
RDF130_o=URIRef("http://lexvo.org/id/iso639-3/eng")
g1.add( (RDF1_s, RDF130_p, RDF130_o) )

RDFf1_p=URIRef("http://purl.org/dc/terms/title")
RDFf1_o=Literal("Visual-Textual-Knowledge-Entity-Linking (VTKEL) Baseline system")
g1.add( (RDF1_s, RDFf1_p, RDFf1_o) )

RDF1_s=URIRef(vtkel1["#"+image_id_annotation])
RDF1_p=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDF1_o=URIRef("http://purl.org/dc/dcmitype/Collection")
g1.add( (RDF1_s, RDF1_p, RDF1_o) )

g2=VTKEL_annotations()
g1=g1+g2
#==> stored VTKEL baseline annotated file in turtle formate
g1.serialize(destination='insert output directory path where you want to stored annotated RDF graph of VTKEL baseline'+image_id_annotation+'VTKEL_annotations.ttl', format='turtle')
print('end')
#==> stored the resultant image file .jpg form
cv2.imwrite('insert output directory path where you want to stored .jpg output image file of VTKEL baseline'+image_id_annotation+'_yolo.jpg', image)
cv2.destroyAllWindows()