# -*- coding: utf-8 -*-
"""
Created on Friday January 01 11:19:47 2020

@author: Shahi Dost

    This is used for the evaluation of VT-LinKEr quality. To run the script please follow the guidelines listed in the main repo of GitHub.

"""

#External libraries and functions data input/out
import requests
import os
import cv2
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace, ConjunctiveGraph
import datetime
from collections import Counter
from textblob import TextBlob

#Integrated functions
from get_sentence_data import *
from Textual_Entities_rdfTripes import *
from Bounding_boxes_annotations import *
from VTKEL_annotations import *
from AgeGender import *


#==> Prefixes to use for RDF graph (triples)
g1 = Graph()
dc = Namespace('http://purl.org/dc/elements/1.1#')
dct = Namespace('http://purl.org/dc/terms/')
gaf = Namespace('http://groundedannotationframework.org/gaf#')
ks = Namespace('http://dkm.fbk.eu/ontologies/knowledgestore#')
nfo = Namespace('http://oscaf.sourceforge.net/')
nif = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
owl = Namespace('http://www.w3.org/2002/07/owl#')
prov = Namespace('https://www.w3.org/TR/prov-o/#prov-o-at-a-glance/')
rdf = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')
vtkel = Namespace('http://vksflickr30k.fbk.eu/resource/')
xml = Namespace('http://www.w3.org/XML/1998/namespace')
xsd = Namespace('http://www.w3.org/2001/XMLSchema#')
yago = Namespace('http://dbpedia.org/class/yago/')

g1.bind("dc",dc)
g1.bind("dct",dct)
g1.bind("gaf",gaf)
g1.bind("ks",ks)
g1.bind("nfo",nfo)
g1.bind("nif",nif)
g1.bind("owl",owl)
g1.bind("prov",prov)
g1.bind("rdf",rdf)
g1.bind("rdfs",rdfs)
g1.bind("vtkel",vtkel)
g1.bind("xml",xml)
g1.bind("xsd",xsd)
g1.bind("yago",yago)

#COCO dataset class type in YAGO Ontology (80 classes)
COCO_TO_YAGO = {
    'airplane':'Airplane102691156',
    'person':'Person100007846',
    'apple':'Apple107739125',
    'backpack':'Backpack102769748',
    'banana':'Banana112352287',
    'bat':'Bat102806379',
    'glove':'Glove103441112',
    'bear':'Bear102131653',
    'bed':'Bed102818832',
    'bench':'Bench102828884',
    'bicycle':'Bicycle102834778',
    'bird':'Bird101503061',
    'boat':'Boat102858304',
    'book':'Book106410904',
    'bottle':'Bottle102876657',
    'bowl':'Bowl102881193',
    'broccoli':'Broccoli111876803',
    'bus':'Bus1029241116',
    'cake':'Cake102937469',
    'car':'Car102958343',
    'carrot':'Carrot112937678',
    'cat':'Cat11021211620',
    'phone':'Telephone104401088',
    'chair':'Chair103001627',
    'clock':'Clock103046257',
    'couch':'Sofa104256520',
    'cow':'Cow102403454',
    'cup':'Cup103147509',
    'table':'DiningTable103201208',
    'dog':'Dog102084071',
    'donut':'Doughnut107639069',
    'elephant':'Elephant102503517',
    'hydrant':'Hydrant103550916',
    'fork':'Fork103383948',
    'frisbee':'Frisbee4387401',
    'giraffe':'Giraffe102439033',
    'drier':'Dryer103251766',
    'handbag':'Bag102774152',
    'horse':'Horse102374451',
    'hotdog':'Frank107676602',
    'keyboard':'Keboard103614007',
    'kite':'Kite113382471',
    'knife':'Knife103623556',
    'laptop':'Laptop103642806',
    'microwave':'Microwave103761084',
    'motorcycle':'Motorcycle103790512',
    'mouse':'Mouse102330245',
    'orange':'Orange107747607',
    'oven':'Oven103862676',
    'parking':'ParkingMeter103891332',
    'pizza':'Pizza107873807',
    'plant':'Plant100017222',
    'refrigerator':'Refrigerator104070727',
    'remote':'RemoteControl104074963',
    'sandwich':'Sandwich107695965',
    'scissors':'Scissors104148054',
    'sheep':'Sheep102411705',
    'sink':'Sink104223580',
    'skateboard':'Skateboard104225987',
    'skis':'Ski104228054',
    'snowboard':'Snowboard1042517911',
    'spoon':'Spoon104284002',
    'ball':'Ball102778669',
    'sign':'Signboard104217882',
    'suitcase':'Bag102773838',
    'surfboard':'Surfboard104363559',
    'teddy':'Teddy104399382',
    'racket':'Racket104039381',
    'tie':'Necktie103815615',
    'toaster':'Toaster104442312',
    'toilet':'Toilet1104446521',
    'toothbrush':'Toothbrush104453156',
    'light':'TrafficLight106874185',
    'train':'Train104468005',
    'truck':'Truck104490091',
    'tv':'TelevisionReceiver104405907',
    'umbrella':'Umbrella104507155',
    'vase':'Vase104522168',
    'glass':'Glass103438257',
    'zebra':'Zebra102391049'}

### connect to 'PIKES server' for knowledge graph in RDF Trig format
PUBLIC_PIKES_SERVER = 'https://knowledgestore2.fbk.eu/pikes-demo/api/'
LOCAL_PIKES_SERVER = 'http://localhost:8011/'

def pikes_text2rdf(img_caption):
    """
    This function takes natural language sentence (i.e. image caption of VTKEL dataset) and passed through ‘PIKES server’ for knowledge graph extraction. 
    input:
      img_caption – input natural language text
    
    output:
      .ttl file – a turtle RDF format output file,  which stored the knowledge graph of natural language in Triples form

    """
    return requests.get(PUBLIC_PIKES_SERVER+"text2rdf?",{'text':img_caption})

def get_textual_entities_metadata(text):
    """
    This function takes natural language sentence and passed through ‘PIKES server’ for knowledge graph extraction with metadata extraction
    (i.e. textual entity mention, YAGO type, anchor, begin index and end index). 
    input:
      text – natural language text
    
    output:
      g.query(sparql_query) – output data from RDF grpah using sparql query

    """    
    pikes_answer = pikes_text2rdf(text.lower())
    
    g = ConjunctiveGraph()
    g.parse(data = pikes_answer.content.decode('utf-8'),format="trig")
    sparql_query = """SELECT ?TED ?TEM ?TET ?anchor ?beginindex ?endindex
           WHERE {
           GRAPH ?g1 {?TED <http://groundedannotationframework.org/gaf#denotedBy> ?TEM}
           GRAPH ?g2 {?TED a ?TET}
           GRAPH ?g3 {?TEM nif:anchorOf ?anchor}
           GRAPH ?g4 {?TEM nif:beginIndex ?beginindex}
           GRAPH ?g5 {?TEM nif:endIndex ?endindex}
           }"""

    return g.query(sparql_query)

def Textual_entities_detetction_linking(image_id):
    print('\n-------------------------------------------------\nPIKES entities processing....')
    """
    This function detects and links textual entities from image captions (i.e. five captions per image) using the PIKES tool and saves it in .ttl file.  
    input:
        image_id - Flickr30k dataset image id
    output:
        textual_entities - Textual entities detected by PIKES
        textual_entities_YAGO_type - YAGO types linked by PIKES
    """    
    
    temp_textual_entity=[]
    temp_textual_entity_type=[]

    #TED and TET
    textual_entities=[]
    textual_entities_YAGO_type=[]
    
    #TEM and indexed

    for i in range(5):
        caption=image_captions[i]['sentence']
        caption_entities_index=get_textual_entities_metadata(caption)

        for row1 in caption_entities_index:
            if 'http://dbpedia.org/class/yago/' in row1[2] and 'http://www.newsreader-project.eu/time/P1D' not in row1[0]:
                
                uri_textual_entity=URIRef(vtkel[image_id]+'C'+str(i)+'/#'+row1[0][21:])
                g1.add( (uri_textual_entity, URIRef(rdf['type']), URIRef(ks['TextualEntity'])) )
                g1.add( (uri_textual_entity, URIRef(rdf['type']), URIRef(row1[2])) )
                
                uri_textual_entity_mention=URIRef(vtkel[image_id+'C'+str(i)+'/#char='+str(row1[4])+','+str(row1[5])])                    
                g1.add( (uri_textual_entity, URIRef(gaf['denotedBy']), uri_textual_entity_mention) )  
                g1.add( (uri_textual_entity_mention, URIRef(rdf['type']), URIRef(ks['TextualEntityMention'])) )
                uri_caption_id=URIRef(vtkel[image_id]+'C'+str(i)+'/')
                g1.add( (uri_textual_entity_mention, URIRef(ks['mentionOf']), uri_caption_id) )
                g1.add( (uri_textual_entity_mention, URIRef(prov['wasAttributedTo']), URIRef(vtkel['PikesAnnotator'])) )
                g1.add( (uri_textual_entity_mention, URIRef(nif['beginIndex']), Literal(row1[4])) )
                g1.add( (uri_textual_entity_mention, URIRef(nif['endIndex']), Literal(row1[5])) )                
                g1.add( (uri_textual_entity_mention, URIRef(nif['anchorOf']), Literal(row1[3])) )

                temp_textual_entity.append(row1[0][21:])
                temp_textual_entity_type.append(row1[2])
        textual_entities.append(temp_textual_entity)
        textual_entities_YAGO_type.append(temp_textual_entity_type)
        temp_textual_entity=[]
        temp_textual_entity_type=[]

    return textual_entities,textual_entities_YAGO_type                    
                    

def YAGO_taxonomy_mapping(Sparql_query_YAGO,upper_class,sub_class):
    """
    This function takes YAGO classes (i.e. Woman110787470, Person100007846) and finds if they are in the same or sub-class hierarchy by 
    mapping YAGO taxonomy.  
    
    input:
      Sparql_query_YAGO – Sparql query for YAGO taxonomy processing
      upper_class – upper class type
      sub_class – sub-class type
    
    output:
    success_flag – binary flag for hierarchy condition (true if same or sub-class success otherwise false)
    YAGO_mapping_hierarchy – the subclass hierarchy between two class extracted from YAGO taxonomy and stored in an array.
    """             
    
    upper_class=str('http://dbpedia.org/class/yago/'+upper_class)
    sub_class=str('http://dbpedia.org/class/yago/'+sub_class)    
    YAGO_mapping_hierarchy=[]
    success_flag=False
    taxonomy_loop_counter=0
    temp_upper_class=upper_class

    while success_flag!=True and taxonomy_loop_counter<2:
        taxonomy_loop_counter+=1
        if upper_class==sub_class:
            success_flag=True
            YAGO_mapping_hierarchy.append(upper_class)
            break
        elif success_flag==False:
            temp_upper_class_in_subject=upper_class
            for uri_subject,uri_property,uri_object in Sparql_query_YAGO:
                if (temp_upper_class_in_subject == uri_subject[:]) and (sub_class==uri_object[:]) and (success_flag==False):
                    success_flag=True
                    YAGO_mapping_hierarchy.append(temp_upper_class_in_subject)
                    break
                elif (temp_upper_class == uri_subject[:]) and (success_flag==False):
                    YAGO_mapping_hierarchy.append(uri_subject[:])
                    temp_upper_class_in_subject=uri_subject[:]
                    temp_upper_class=uri_object[:]
                    if temp_upper_class==sub_class and success_flag==False:
                        success_flag=True
                        YAGO_mapping_hierarchy.append(temp_upper_class)
                        break
            if success_flag==False and taxonomy_loop_counter==2:
                break
    return success_flag,YAGO_mapping_hierarchy

"""
Pre-train YOLO version 3 model for detecting visual objects.
"""    

def get_output_layers(net):
    """
    This function gets the last layer from the YOLO system and returns for further processing of visual entity detection and typing. 
    input: 
    net - YOLOv3 object detector and classification model
    
    Output:
    output_layers - Features from output layer
    """    

    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    This function takes the input image(s), class label, confidence score, and bounding boxes values from VT-LinKER and draws bounding boxes (i.e. visual objects).
    
    input:
        img – jpg image
        class_id – class id from YOLO pre-trained model
        confidence – YOLO object detected condifence score (if score is >= 0.5 object detected)
        x – left top corner of bounding box (in x-axis)
        y – left top corner of bounding box (in y-axis)
        x_plus_w – width of bounding box
        y_plus_h – height of bounding box
    Output:
        
    """

    label = str(classes[class_id])
    class_names.append(label)
    YOLO_class_names.append(label)
    class_names.append(class_id)
    class_names.append(confidence)
    class_names.append(x)
    class_names.append(y)
    class_names.append(x_plus_w)
    class_names.append(y_plus_h)
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def same_visual_mentions_handling(visual_entities):
    """
    This function assigns names into two or more than two same visual objects detected by VT-LinKEr. For example, if there are two people in an image
    with 'person' labels, it will be different into person_1 and person_2 objects.   
    
    input:
        visual_entities – Visual objects detected by YOLO 
    Output:
        visual_entities – Unique objects names
    """
    same_visual_entities = Counter(visual_entities)
    for item in same_visual_entities:
        if same_visual_entities[item]>1:
            item_count=same_visual_entities[item]
            for i in range(len(visual_entities)):
                if item==visual_entities[i] and item_count>1:
                    visual_entities[i]=item+'_'+str(item_count)
                    item_count-=1

    return visual_entities

def allignment_of_visual_textual_entities(image_captions,image_id,YOLO_class_names_gender,YOLO_in_YAGO_gender,yago_taxonomy,textual_entities,textual_entities_YAGO_type):
    """
    This function takes visual and textual entity mentions with YAGO-type detected by VT-LinKEr and makes alignment using YAGO taxonomy mapping.
     VT-LinKEr is doing the alignment by processing the visual and textual classes (i.e. visual and textual entity YAGO-type) predicted by VT-LinKER
    and mapped through YAGO taxonomy by using same-class or sub-class hierarchy. For example, a person (i.e. female) in the picture will be aligned to the woman in 
    the text by processing the YAGO type hierarchical relationship (woman sub-class of female).
     
    input:
        image_captions - Flickr30k dataset image caption text
        image_id - Flickr30k dataset image id
        YOLO_class_names - YOLO object labels (with respect to gender classification if having person label)
        YOLO_in_YAGO - YOLO label YAGO-type
        yago_taxonomy - YAGO taxonomy parsed file
        textual_entities - Textual entity mentions process by PIKES
        textual_entities_YAGO_type - Textual entity mentions typed in YAGO process by PIKES
 
     Output:
         VTC_list - return the list of aligned visual-textual entities. 
        
    """    

    VTC_list_temp=[]
    VTC_list=[]
    for i in range(5):
        temp_VED=YOLO_class_names_gender.copy()
        temp_VET=YOLO_in_YAGO_gender.copy()
        temp_TED=textual_entities[i]
        temp_TET=textual_entities_YAGO_type[i]
        switch_flag=False

        print('--------------------------------')
        
        VE_length=len(YOLO_class_names_gender)-1
        while len(temp_VED)>=1 and VE_length>=0:
            VE_temp=temp_VET[VE_length]
            index_VE=VE_length
            VE_length-=1
            
            for k in range(len(textual_entities[i])):
                mapping_hierarchy=[]
                temp_VE_class=VE_temp
                temp_TE_class=temp_TET[k][30:]
                mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(yago_taxonomy,temp_TE_class,temp_VE_class)
                
                #swtiched TET with VET for YAGO-hierarchy checking
                if mapping_flage==False:
                        mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(yago_taxonomy,temp_VE_class,temp_TE_class)
                        if mapping_flage==True:
                            switch_flag=True
                            
                if mapping_flage==True:

                    #stored CRC
                    if temp_TE_class!=temp_VE_class and switch_flag==False:
                        g1.add( (URIRef(yago[temp_TE_class]), URIRef(rdfs['subClassOf']), URIRef(yago[temp_VE_class])) )
                    elif temp_TE_class!=temp_VE_class and switch_flag==True:
                        g1.add( (URIRef(yago[temp_VE_class]), URIRef(rdfs['subClassOf']), URIRef(yago[temp_TE_class])) )
                        
                    uri_textual_entity=URIRef(vtkel[image_id+'C'+str(i)]+'/#'+temp_TED[k])
                    
                    uri_visual_entity=URIRef(vtkel[image_id+'I'+'/#'+temp_VED[index_VE]])
                    g1.add( (uri_textual_entity, URIRef(owl['sameAs']), uri_visual_entity) )
                    VTC_list_temp.append(uri_visual_entity)
                    VTC_list_temp.append(yago[temp_VE_class])
                    
                    g1.add( (uri_textual_entity, URIRef(rdf['type']), URIRef(yago[temp_TE_class])) )
                    VTC_list_temp.append(uri_textual_entity)
                    VTC_list_temp.append(URIRef(yago[temp_TE_class]))
                    VTC_list.append(VTC_list_temp)
                    VTC_list_temp=[]

                    temp_VED.pop(index_VE)
                    temp_VET.pop(index_VE)
                    temp_TED.pop(k)
                    temp_TET.pop(k)
                    break

            if mapping_flage==False:
                temp_VED.pop(index_VE)
                temp_VET.pop(index_VE)
                
    return VTC_list


def gender_injection(gender_data, bboxes,YOLO_in_YAGO_gender,YOLO_class_names_gender,bounding_boxes):

    """
    This function takes gender-classification algorithm data compared it with the YOLO person-label to classify the person class between Male 
    and Female. If the face detected by gender-classifier algorithm overlap with person bounding-box, we mark it correct (person->male/female).
     
    input:
        gender_data - gender classifier data from algorithm 
        bboxes -  face bounding boxes data
        YOLO_in_YAGO -  YAGO typed of YOLO
        YOLO_class_names - YOLO class labels
        bounding_boxes - YOLO bounding boxes data
 
     Output:
        YOLO_in_YAGO - YAGO typed YOLO class lables (with respect to gender-classifier)
        YOLO_class_names - YOLO class lables (with respect to gender-classifier)
    """            
    for i in range(len(gender_data)):
        for j in range(len(YOLO_in_YAGO_gender)):
            if 'person' in YOLO_class_names_gender[j]:
                if (bboxes[i][0]>=bounding_boxes[j][0] and bboxes[i][1]>=bounding_boxes[j][1] and (bboxes[i][2]>=bounding_boxes[j][0] and bboxes[i][2]<=bounding_boxes[j][2]) and (bboxes[i][3]>=bounding_boxes[j][1] and bboxes[i][3]<=bounding_boxes[j][3]) ):
                    if 'Male' in gender_data[i]:
                        YOLO_in_YAGO_gender[j]='Male109624168'
                        YOLO_class_names_gender[j]='male'
                    elif 'Female' in gender_data[i]:
                        YOLO_in_YAGO_gender[j]='Female109619168'
                        YOLO_class_names_gender[j]='female'

    return YOLO_in_YAGO_gender,YOLO_class_names_gender,bounding_boxes

def IOU(boxA,boxB):
    
    """
    This function takes two bounding boxes (one from benchmark-dataset and other predicted by VT-LiKEr) and calculate IOU values.
     
     input:
        boxA - VT-LiKEr predicted bounding-box values
        boxB - Bounding-box values from VTKEL dataset
 
     Output:
        iou - Intersection-over-union values (from 0-1)
        
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
    return iou
        
def Benchmark_VED_VET(Benchmark_dataset,img_id):

    """
    This function takes VTKEL dataset and extract the benchmark:
        1) Corefrence ids
        2) visual entities, bounding boxes and YAGO-type
        3) textual mentions with respect to captions ids and YAGO-type
    and stored into dictionary name 'benchmark_data'.
     
    input:
        Benchmark_dataset - VTKEL dataset file
        img_id - image id (from evaluation documents folder)
 
     Output:
        benchmark_data - visual-textual mention data (i.e. bounding box values, flickr30k ontology value, corresponding textual chain)
        
    """
    
    single_bb=[]
    Benchmark_bb=[]
    corner=''
    single_value_dic={}
    textual_entities=[]
    benchmark_data=[]
    yago_type=[]

    for state in Benchmark_dataset:

        if img_id in state[0] and 'denotedBy' in state[1] and 'http://vksflickr30k.fbk.eu/resource/'+img_id+'I/#xywh' in state[2]:
            
            DVM_xywh=state[2]
            DVM_xywh=DVM_xywh.replace('http://vksflickr30k.fbk.eu/resource/'+str(img_id)+'I/#xywh=','')

            single_value_dic['id']=state[0][:]
            for i in range(len(DVM_xywh)):
                if DVM_xywh[i]!=',':
                    corner=corner+DVM_xywh[i]
                    if i==len(DVM_xywh)-1:
                        single_bb.append(int(corner))
                        corner=''

                elif DVM_xywh[i]==',':
                    single_bb.append(int(corner))
                    corner=''

            single_value_dic['xywh']=single_bb
            Benchmark_bb.append(single_bb)
            single_bb=[]

            # extracting corresponding textual-entity (from image caption) for visual mention with YAGO-type
            for state1 in Benchmark_dataset:
                if state[0] in state1[0] and 'sameAs' in state1[1]:
                    textual_entities.append(state1[2][:])

                    for state2 in Benchmark_dataset:
                        if state1[2] in state2[0] and 'yago' in state2[2]:

                            yago_type.append(state2[2][:])
            single_value_dic['textul_entities']=textual_entities
            single_value_dic['yago_type']=yago_type
            benchmark_data.append(single_value_dic)

            textual_entities=[]
            yago_type=[]
            single_value_dic={}

    return benchmark_data

def VED_VET_Evaluations(benchmark_data,VET,VED,VE_bounding_boxes,yago_taxonomy):
    """
    This function takes the benchmark dataset (i.e. VTKEL) and visual part of VT-LinKEr to performed the visual-entity detection and typing (VED+VET) evaluations. 
        
    input:
        Benchmark_dataset - VTKEL dataset
        VED - Visual entities detected by VT-LinKEr 
        VET - Visual entities YAGO type (i.e. linked) by VT-LinKEr
        VE_bounding_boxes - Bounding boxes values predicted by VT-LinKEr
        yago_taxonomy - YAGO taxonomy parsed file
 
     Output:
        BM_total_VE - counting total benchmark visual entities
        VT_LinKEr_total_VE - counting total visual entities predicted by VT-LinKEr
        correct_VE - those visual entities matched by evaluations with respect to IOU>=0.5 and YAGO same-class/subclass hierarchy based.
    """    
    
    BM_total_VE=len(benchmark_data)
    VT_LinKEr_total_VE=len(VED)
    correct_VE=0
    
    for i in range(len(VED)):
        for data in benchmark_data:
            # first step to match IOU values of VTKEL and VT-LinKEr
            if IOU(VE_bounding_boxes[i],data['xywh'])>=0.5 :
                # second step to match same/subclass hierarchy of YAGO type of VTKEL and VT-LinKEr
                mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(yago_taxonomy,VET[i],data['yago_type'][0][30:])
                if mapping_flage==False:
                    mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(yago_taxonomy,data['yago_type'][0][30:],VET[i])
                    if mapping_flage==True:
                        correct_VE+=1

                else:
                    correct_VE+=1

    return BM_total_VE,VT_LinKEr_total_VE,correct_VE


def Benchmark_TED_TET(Benchmark_dataset,img_id):
    """
    This function takes VTKEL .ttl file and extract benchmark textual-entity-mentions with respect to image-id.
     
    input:
        Benchmark_dataset - VTKEL dataset file
        img_id - given image id (from evaluation documents folder)
 
     Output:
        benchmark_data - visual-textual-mention data (i.e. textual data with YAGO-type from captions)
        
    """
    tex_ent=[]
    tex_ent_type=[]
    # filter VTKEL dataset for caption data
    for rdf_triple in Benchmark_dataset:
        if img_id in rdf_triple[0] and 'sameAs' in rdf_triple[1]:
            for rdf_triple1 in Benchmark_dataset:
                if rdf_triple[2] in rdf_triple1[0] and 'yago' in rdf_triple1[2]:
                    tex_ent.append(rdf_triple[2])
                    tex_ent_type.append(rdf_triple1[2])

    return tex_ent,tex_ent_type

def TED_TET_Evaluations(BM_TED_list,BM_TET_list,VT_LinKEr_TED,VT_LinKEr_TET,yago_taxonomy,img_id):
    """
    This function takes textual part of benchmark dataset and matches with VT-LinKEr  for textual-entity detection and typing (TED+TET) evaluations.
 
    input:
        BM_TED_list - list of textual data from VTKEL
        BM_TET_list - list of textual data YAGO-type
        VT_LinKEr_TED - textual-entity predicted by VT-LinKEr  
        VT_LinKEr_TET - textual-entity YAGO-type Predicted BY VT-LinKEr 
        yago_taxonomy - YAGO taxonomy file
        img_id - Flickr30k dataset image id
 
     Output:
        BM_total_TED - counting total textual-entities of benchmark (i.e. VTKEL)
        VT_LinKEr_total_TED - counting total textual-entities predicted by VT-LinKEr
        correct_TE - those textual-entities matched by evaluations with respect to same/part of string and YAGO same-class/subclass hierarchy based on VTKEL and VT-LinKEr.

    """    
    BM_total_TED=len(BM_TED_list)
    VT_LinKEr_total_TED=len(VT_LinKEr_TED[0])+len(VT_LinKEr_TED[1])+len(VT_LinKEr_TED[2])+len(VT_LinKEr_TED[3])+len(VT_LinKEr_TED[4])
    correct_TE=0
    
    for i in range(5):
        for j in range(len(VT_LinKEr_TED[i])):
            for k in range(len(BM_TED_list)):
                #first step matches VT-LinKEr textual entity with VTKEL on the basis of same/part noun-phrase
                if VT_LinKEr_TED[i][j] in BM_TED_list[k] and 'C'+str(i) in BM_TED_list[k]:
                    #second step matches VT-LinKEr entity-typed with VTKEL on the basis of class/subclass hierarchy of YAGO taxonomy.
                    mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(yago_taxonomy,VT_LinKEr_TET[i][j][30:],BM_TET_list[k][30:])
                    if mapping_flage==False:
                        mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(yago_taxonomy,BM_TET_list[k][30:],VT_LinKEr_TET[i][j][30:])
                        if mapping_flage==True:
                            correct_TE+=1
                    else:
                        correct_TE+=1

    return BM_total_TED,VT_LinKEr_total_TED,correct_TE

def VTC_Evaluations(VTC_LinKEr_list,LinKEr_bb,benchmark,VEM):
    """
    This function takes the benchmark visual-textual corefernce (VTC) and matches with VT-LinKEr VTC chains. If the benchmark visual-entity matches
    with VT-LinKEr by IOU>=0.5 and textual-entity matches with same/part of string we mark it correct VTC.
 
    input:
        VTC_LinKEr_list - VTC chains predicted by VT-LinKEr
        LinKEr_bb - bounding boxes by VT-LinKEr
        benchmark - Benchmark VTC chains data 
        VEM - Visual entity mention from benchmark
 
     Output:
        VTC_LinKEr - correct VTC chains predicted by VT-LinKEr (i.e. total true positive)
        VTC_benchmark - Benchmark VTC

    """    

    VTC_benchmark=0
    VTC_LinKEr=0
    for data in benchmark:
        VTC_benchmark=VTC_benchmark+len(data['textul_entities'])

    for i in range(len(LinKEr_bb)):
        for data in benchmark:
            # matches Vep<->Veg under IOU ratio
            if IOU(LinKEr_bb[i],data['xywh'])>=0.5 :
                for j in range(len(VTC_LinKEr_list)):
                    # matches tep<->teg under equal or a substring of teg
                    if VEM[i] in VTC_LinKEr_list[j][0]:
                        VTC_LinKEr+=1

    return VTC_LinKEr, VTC_benchmark

###YAGO Taxonomy .ttl file path
yago_taxonomy_file_path="/$root-directory/yago_taxonomy-v1.1.ttl"

graph_1=ConjunctiveGraph()
yago_taxonomy=graph_1.parse(yago_taxonomy_file_path, format="turtle")

#upload image folder
image_file_counter=0
images_directory_path='/$root-directory/FlickrImagesFolder/'

#upload benchmark files
#Benchmark_dataset = graph_1.parse('/home/sdost/Desktop/PhD/VTKEL_Supervise/Evaluations/VTKEL_dataset_23008340.ttl', format='n3')
Benchmark_dataset = graph_1.parse('/$root-directory/VTKEL/VTKEL_dataset.ttl', format='n3')

#variables to count total states

#VED+VET part
BM_total_VE_total=0
VT_LinKEr_total_VE_total=0
correct_VE_total=0

#TED+TET part
BM_total_TED_total=0
VT_LinKEr_total_TED_total=0
correct_TE_total=0

#VTC part
VTC_LinKEr_count_total=0
VTC_benchmark_count_total=0

#select one by one image from Flickr30k images for VT-LinKEr evaluations
for filename in os.listdir(images_directory_path):

    image_file_counter+=1    
    if filename.endswith(".jpg"):
        
        print('====================================================\n',image_file_counter,':','image file->',filename,'---------------------')
        image_path=images_directory_path+filename
        image = cv2.imread(image_path)
        image_id=filename
                
        #==> read the image caption file
        image_id=image_id[:-4]
        image_captions=get_sentence_data('/$root-directory/Flickr30k_caption/'+image_id+'.txt')

        #==> image captions
        print('----------------------------------------------------\nImage captions processing....\n')
        print('C0:',image_captions[0]['sentence'])
        print('C1:',image_captions[1]['sentence'])
        print('C2:',image_captions[2]['sentence'])
        print('C3:',image_captions[3]['sentence'])
        print('C4:',image_captions[4]['sentence'])
        
        #Function for visual-entities detection and typing (VED+VET) evaluations
        benchmark_data_VED_VET=Benchmark_VED_VET(Benchmark_dataset,image_id)
        TED_list,TET_list=Benchmark_TED_TET(Benchmark_dataset,image_id)
                
        ###==> PIKES System Phase
        print('\n-------------------------------------------------\nPIKES processing....')
        textual_entities=[]
        textual_entities_YAGO_type=[]
        textual_entities,textual_entities_YAGO_type=Textual_entities_detetction_linking(image_id)
        
        ###==> YOLO System Phase
        print('\n-------------------------------------------------\nYOLO objects detection processing....')  
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        """
        Pre-train YOLO version 3 model for detecting visual objects.
        """    
        # read class names from text file
        classes = None
        with open('/$root-directory/YOLO/yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet('/$root-directory/YOLO/yolov3.weights', '/$root-directory/YOLO/yolov3.cfg')
        
        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        
        # set input blob for the network
        net.setInput(blob)
        
        class_names=[]
        temp_storing_bb=[]
        bounding_boxes=[]
        YOLO_class_names=[]

        outs = net.forward(get_output_layers(net))
        
        # yolo variables initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.3

        #Non-maximum suppression (NMS)
        nms_threshold = 0.4
        
        # for each visual-detetion from output YOLO-layer get the confidence, class id, bounding box params (confidence >= 0.5)
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
            temp_storing_bb.append(round(x))
            temp_storing_bb.append(round(y))
            temp_storing_bb.append(round(x+w))
            temp_storing_bb.append(round(y+h))
            bounding_boxes.append(temp_storing_bb)
            temp_storing_bb=[]

        yolo_class_in_YAGO=[]
        for i in range(len(YOLO_class_names)):
            for x in COCO_TO_YAGO:
                if YOLO_class_names[i]==x:
                    yolo_class_in_YAGO.append(COCO_TO_YAGO[x])        
        
        YOLO_class_in_YAGO=yolo_class_in_YAGO

        # Visual-Textual alignment Phase
        print('\n-------------------------------------------------\nAlignment processing....')   
                                       
        YOLO_class_names_unique=same_visual_mentions_handling(YOLO_class_names)

        # Gender identification phase
        gender_data, bboxes=Gender_main(image_path)
        YOLO_in_YAGO_gender,YOLO_class_names_gender,YOLO_bounding_boxes=gender_injection(gender_data, bboxes,YOLO_class_in_YAGO,YOLO_class_names_unique,bounding_boxes)

        # VED+VET Evaluations 
        BM_total_VE,VT_LinKEr_total_VE,correct_VE=VED_VET_Evaluations(benchmark_data_VED_VET,YOLO_in_YAGO_gender,YOLO_class_names_gender,YOLO_bounding_boxes,yago_taxonomy)
        BM_total_VE_total=BM_total_VE_total+BM_total_VE
        VT_LinKEr_total_VE_total=VT_LinKEr_total_VE_total+VT_LinKEr_total_VE
        correct_VE_total=correct_VE_total+correct_VE

        #TED+TET Evaluations
        BM_total_TED,VT_LinKEr_total_TED,correct_TE=TED_TET_Evaluations(TED_list,TET_list,textual_entities,textual_entities_YAGO_type,yago_taxonomy,image_id)
        BM_total_TED_total=BM_total_TED_total+BM_total_TED
        VT_LinKEr_total_TED_total=VT_LinKEr_total_TED_total+VT_LinKEr_total_TED
        correct_TE_total=correct_TE_total+correct_TE

        graph_3=Bounding_boxes_annotations(bounding_boxes,YOLO_class_in_YAGO,YOLO_class_names_unique,image_id)
        g1=g1+graph_3
        
        VTC_list=allignment_of_visual_textual_entities(image_captions,image_id,YOLO_class_names_gender,YOLO_in_YAGO_gender,yago_taxonomy,textual_entities,textual_entities_YAGO_type)

        #VTC Evaluations
        VTC_LinKEr_count, VTC_benchmark_count=VTC_Evaluations(VTC_list,YOLO_bounding_boxes,benchmark_data_VED_VET,YOLO_class_names_gender)
        VTC_LinKEr_count_total=VTC_LinKEr_count_total+VTC_LinKEr_count
        VTC_benchmark_count_total=VTC_benchmark_count_total+VTC_benchmark_count
        
        RDF1_s=URIRef(vtkel)
        g1.add( (URIRef(vtkel), URIRef(rdf['type']), URIRef("http://purl.org/dc/dcmitype/Software")) )

        authors=Literal("Shahi Dost & Luciano Serafini")
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/creator"), authors) )        

        t= datetime.datetime.now()
        creation_time=Literal( str(t.year)+'-'+str(t.month)+'-'+str(t.day)+':'+'-'+str(t.hour)+':'+str(t.minute)+':'+str(t.second))
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/created"), creation_time) )
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/language"), URIRef("http://lexvo.org/id/iso639-3/eng")) )

        uri_VT_LinKEr_title=Literal("Visual Textual Linker of Entities with Knowledge (VT-LinKEr)")
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/title"), uri_VT_LinKEr_title) )
        g1.add( (URIRef(vtkel["#"+image_id]), URIRef(rdf['type']), URIRef("http://purl.org/dc/dcmitype/Collection")) )
        
        g2=VTKEL_annotations(image_id,image_captions)
        g1=g1+g2
        
        # stored the resultant image file .jpg form
        cv2.imwrite('/$root-directory/folder-to-store-output-Images/'+image_id+'_yolo.jpg', image)
        cv2.destroyAllWindows()        
        

def Experinment_Statistics(image_file_counter,correct_VE_total,VT_LinKEr_total_VE_total,BM_total_VE_total,correct_TE_total,VT_LinKEr_total_TED_total,BM_total_TED_total,VTC_LinKEr_count_total,VTC_benchmark_count_total):
    """
    This function takes visual, textual and visual-textual-coreference evaluations parameters to calculate precision, recall and F1 score of:
        (i) visual-entity detection and typing (VED+VET)
        (ii) textual-entity detection and typing (TED+TET)
        (iii) visual-textual coreference (VTC).
 
        input:
        image_file_counter - total number of images used for evaluations
        correct_VE_total - true-positive (TP) entry for visual entity detection
        VT_LinKEr_total_VE_total - total visual-entities predicted by VT-LinKEr (TP+FP)
        BM_total_VE_total - Benchmark (i.e. VTKEL) total visual entities
        correct_TE_total - TP entry for textual-entity detection
        VT_LinKEr_total_TED_total - total textual-entities predicted by VT-LinKEr (TP+FP)
        BM_total_TED_total - total benchmark textual entities
        VTC_LinKEr_count_total - correct (TP) visual-textual corefernce chains predcited by VT-LinKEr
        VTC_benchmark_count_total - total benchmark visual-textual coreferencne chains
 
     Output:
        F1_VE - F1 score for visual-entity detection and typing
        F1_TE - F1 score for textual-entity detection and typing
        F1_VTC - F1 score for visual-textual corefernce

    """    
    #VET+VED statistics
    TP_VE=correct_VE_total
    FP_VE=VT_LinKEr_total_VE_total-correct_VE_total
    FN_VE=BM_total_VE_total-correct_VE_total
    
    Precision_VE=(TP_VE/(TP_VE+FP_VE))
    Recall_VE=(TP_VE/(TP_VE+FN_VE))
    
    F1_VE=((2*Precision_VE*Recall_VE)/(Precision_VE+Recall_VE))
    
    #TET+TED statistics
    TP_TE=correct_TE_total
    FP_TE=VT_LinKEr_total_TED_total-correct_TE_total
    FN_TE=BM_total_TED_total-correct_TE_total
    
    Precision_TE=(TP_TE/(TP_TE+FP_TE))
    Recall_TE=(TP_TE/(TP_TE+FN_TE))
    
    F1_TE=((2*Precision_TE*Recall_TE)/(Precision_TE+Recall_TE))
    
    #VTC statistics
    TP_VTC=VTC_LinKEr_count_total
    FP_VTC=correct_VE_total-VTC_LinKEr_count_total
    FN_VTC=VTC_benchmark_count_total-VTC_LinKEr_count_total
    
    Precision_VTC=(TP_VTC/(TP_VTC+FP_VTC))
    Recall_VTC=(TP_VTC/(TP_VTC+FN_VTC))
    
    F1_VTC=((2*Precision_VTC*Recall_VTC)/(Precision_VTC+Recall_VTC))
    
    return F1_VE,F1_TE,F1_VTC

# function call to calcuate evaluation results      
F1_VE,F1_TE,F1_VTC=Experinment_Statistics(image_file_counter,correct_VE_total,VT_LinKEr_total_VE_total,BM_total_VE_total,correct_TE_total,VT_LinKEr_total_TED_total,BM_total_TED_total,VTC_LinKEr_count_total,VTC_benchmark_count_total)
print('VT-LinKEr evaluations results:\n','VED+VET :',F1_VE,'\nTED+TET :',F1_TE,'\nVTC :',F1_VTC)    

# Store VT-LinKEr output .ttl file
g1.serialize(destination='/$root-directory/VT-LinKEr_anno.ttl', format='turtle')
print('end of VT-LinkEr evaluation!')        
        