# -*- coding: utf-8 -*-
"""
Created on Monday October 25 11:19:47 2019

@author: Shahi Dost

    This is the evaluation code for VT-LinKEr framework. We evaluate VT-LinKEr on (1) 300 (2) 1000 and,
    (3) 31K documents datasets. For more details please read our paper:TBW
"""

#---------- External libraries and functions ---------------#

import requests
import os
import cv2
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace, ConjunctiveGraph
import datetime
from collections import Counter

#---------- Integrated functions ----------------------#

from get_sentence_data import *


#---------- Prefixes used for RDF graph (triples) ----------------#

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

#------------- Microsoft COCO dataset classes with respective YAGO Ontology classes (total:80 classes) --------#

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

#---------- connect to 'PIKES server' to find knowledge graphs from texual resource ---------#

PUBLIC_PIKES_SERVER = 'https://knowledgestore2.fbk.eu/pikes-demo/api/'
LOCAL_PIKES_SERVER = 'http://localhost:8011/'

def pikes_text2rdf(img_caption):
    """
    Takes a input natural language sentence (in our case image caption) and passed through ‘PIKES server’ for knowledge graphs extraction 
    input:
      img_caption – input natural language text
    
    output:
      .ttl file – a turtle RDF format output file,  which stored the knowledge graph of natural language in Triples form

    """
    return requests.get(PUBLIC_PIKES_SERVER+"text2rdf?",{'text':img_caption})

def get_entities_from_text(img_text):
    """
    This function extract RDF graph for textual entities and their type, processed by PIKES tool.
    input:
      img_text –  Image caption from Flickr30k-entities dataset
    
    output:
     sparql query result  – stored textual entities recognized and linked by PIKES tool

    """
    pikes_answer = pikes_text2rdf(img_text.lower())
    
    g = ConjunctiveGraph()
    g.parse(data = pikes_answer.content.decode('utf-8'),format="trig")
    sparql_query = """SELECT ?subject ?object
           WHERE {
           GRAPH ?graph_1 {?subject a ks:Entity}
           GRAPH ?graph_2 {?subject rdf:type ?object}
           }"""

    return g.query(sparql_query)

def Textual_ent_detec_linking():

    """
    This function extract textual entities from image captions (five captions per image) using PIKES tool.  
    input:
        
    output:
        textual_entities - Textual entities detected by PIKES
        YAGO_type - Textual entities YAGO types linked by PIKES
        """    
    textual_entities=[]
    YAGO_type=[]
    temp_textual_entity=[]
    temp_YAGO_type=[]

    for i in range(5):
        img_caption=image_captions[i]['sentence']
        caption_entities = get_entities_from_text(img_caption)
        for row in caption_entities:
            if 'http://dbpedia.org/class/yago/' in row[1] and 'http://www.newsreader-project.eu/time/P1D' not in row[0]:
                temp_textual_entity.append(row[0][21:])
                temp_YAGO_type.append(row[1])
        textual_entities.append(temp_textual_entity)
        YAGO_type.append(temp_YAGO_type)
        temp_textual_entity=[]
        temp_YAGO_type=[]
        for row in caption_entities:
            if 'http://groundedannotationframework.org/gaf#denotedBy' in row[1] and i==0:
                print("")

    return textual_entities,YAGO_type

def YAGO_taxonomy_mapping(Sparql_query_YAGO,upper_class,sub_class):
    """
    This function takes two YAGO classes (i.e. Woman110787470 & Person100007846) and find if they are in same or sub-class hierarchy
    by mapping all YAGO taxonomy.  
    input:
      Sparql_query_YAGO – Sparql query for YAGO taxonomy
      upper_class – upper class type
      sub_class – sub-class type
    output:
    success_flag – binary flag for hierarchy condition (true if same or sub-class success otherwise false)
    mapping_hierarchy – the subclass hierarchy between two class extracted from YAGO taxonomy and stored in an array.
    """

    URI_upper_class=str('http://dbpedia.org/class/yago/'+upper_class)
    URI_sub_class=str('http://dbpedia.org/class/yago/'+sub_class)
    mapping_hierarchy=[]
    success_flag=False

    for uri_subject,uri_property,uri_object in Sparql_query_YAGO:
        if (URI_upper_class == uri_object[:]) and (URI_sub_class==uri_subject[:]):
            success_flag=True
            mapping_hierarchy.append(URI_upper_class)
            break

    return success_flag,mapping_hierarchy

"""
YOLO-version 3, aspect ration 460x460 with COCO train model for visual objects detection.
"""    

def get_output_layers(net):
    
    """
    YOLOv3-Neural Network layer for object detections, 
    input: 
    net - A neural network layer
    
    Output:
    output_layers - Features for output layer
    """    

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

#---------- Function to draw bounding boxes ----------#

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    """
        This function takes upper-top corner (i.e. x, y), width, height, image and image-id processed by YOLO model and
        draw the bounding boxes around detected objects.  

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

def same_vis_mentions(visual_entities):
    """
    This function differentiate two (or more that two) same visual objects ids detected by YOLO. For example, if YOLO
    detected two person (i.e. two people) this function will assign two ids person_1 and person_2 to person class.   

    input:
        visual_entities – Visual objects detected by YOLO 
    Output:
        visual_entities – Unique visual objects ids
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

def Allignment_of_vis_tex_ent(tex_ent_YAGO_type,Sparql_query,YOLO_in_YAGO):

    """
    This function takes visual entity mentions detected by YOLO object detector and textual entity mentions detected by 
    PIKES tool and make alignment using YAGO taxonomy mapping. YAGO taxonomy mapping is done by, processing visual-textual classes
    (visual entity type and textual entity type) and mapped through YAGO taxonomy for same-class or sub-class hierarchy detection.
     
    input:
        tex_ent_YAGO_type - Textual entity mentions typed in YAGO
        Sparql_query – Sparql query for mapping
        YOLO_in_YAGO – Corresponding YOLO class ids in YAGO class type
 
     Output:
        VTC_textual - counting of textual entities aligned visual entities(chain of visual entities with captions entities)
        
    """
    mapping_flage=False
    VTC_textual=0
    
    for caption_id in range(5):
        person_counter=0
        for i in range(len(YOLO_in_YAGO)):
            vis_class_mapping=YOLO_in_YAGO[i]

            for j in range(len(tex_ent_YAGO_type[caption_id])):       
                tex_class_mapping=tex_ent_YAGO_type[caption_id][j][30:]
                mapping_hierarchy=[]
                mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(Sparql_query,vis_class_mapping,tex_class_mapping)

                if mapping_flage==True and 'Person' in vis_class_mapping:
                    person_counter+=1
                    VTC_textual+=1
                    i+=1
                elif mapping_flage==True:
                    if vis_class_mapping!=tex_class_mapping:
                        VTC_textual+=1
                        mapping_flage=False
    return VTC_textual

def Benchmark_visual_data(Benchmark_dataset,img_id):

    """
    This function takes VTKEL dataset and extract the benchmark visual-textual-mentions data with respect to given image-id.
     
    input:
        Benchmark_dataset - VTKEL dataset file
        img_id - given image id (from evaluation documents folder)
 
     Output:
        benchmark_data - visual-textual-mention data (i.e. bounding box values, flickr30k ontology value, corresponding textual chain)
        
    """
    
    single_bb=[]
    Benchmark_bb=[]
    corner=''
    single_value_dic={}
    textual_entities=[]
    benchmark_visual_data=[]
    yago_type=[]
    for state in Benchmark_dataset:
        # filtering bounding box triples from VTKEL dataset
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
            benchmark_visual_data.append(single_value_dic)

            textual_entities=[]
            yago_type=[]
            single_value_dic={}

    return benchmark_visual_data

def Benchmark_tex_ent_data(Benchmark_dataset,img_id):

    """
    This function takes VTKEL dataset and extract benchmark textual-entity-mentions data with respect to given image-id.
     
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

def VTKEL_IOU(boxA,boxB):
    
    """
    This function takes two bounding boxes (one from benchmark dataset and second predicted by VT-LiKEr) and calculate IOU values.
     
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

def Evaluations_VTC(Sparql_query_YAGO,benchmark_data,yolo_bb,YOLO_class_names,YOLO_in_YAGO):

    """
    This function takes benchmark and predicted visual entities and find their alignment with respect to YAGO taxonomy classes
    hierarchy. First find, if the IOU values of predicted and benchmark visual entities is >=0.5, second find the alignment between
    predicted visual entities with textual entities by using YAGO taxonomy hierarchy and at the end find calculate these alignment
    with respect to benchmark visual-textual entities.
     
    input:
        Sparql_query_YAGO - YAGO taxonomy file
        benchmark_data - Benchmark visual-textual aligned data (extracted from VTKEL dataset)
        yolo_bb - VT-LinKEr predicted bounding boxes data
        YOLO_class_names - VT-LinKEr predicted class ids
        YOLO_in_YAGO - VT-LinKEr predicted corresponding YAGO type
 
     Output:
        Benchmark_tot_vis_ent - Benchmark visual entities
        Benchmark_tot_vis_tex_ent - total counted visual-textual entities
        LinKEr_tot_vis_ent - VT-LiKEr total visual-entities
        corr_vis_ent - Those visual entities whose values are >=0.5 of IOU
        corr_vis_tex_ent - corresponding aligned textual entities
        VTC_count - Visual-textual coreference count
        
    """
    
    corr_vis_ent=0
    VTC_count=0
    Benchmark_tot_vis_ent=len(benchmark_data)
    print('benchmark_data')

    LinKEr_tot_vis_ent=len(YOLO_class_names)

    corr_vis_tex_ent=0
    Benchmark_tot_vis_tex_ent=0
    for i in range(len(yolo_bb)):
        for j in range(len(benchmark_data)):
            
            #count benchmark visual-textual entities
            Benchmark_tot_vis_tex_ent=Benchmark_tot_vis_tex_ent+len(benchmark_data[j]['textul_entities'])

            if VTKEL_IOU(yolo_bb[i],benchmark_data[j]['xywh'])>=0.5 :

                corr_vis_ent+=1
                corr_vis_tex_ent=corr_vis_tex_ent+len(benchmark_data[j]['textul_entities'])

                mapping_flage=False
                mapping_hierarchy=[]
                # find the YAGO classes hierarchy matching
                mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(Sparql_query_YAGO,YOLO_in_YAGO[i],benchmark_data[j]['yago_type'][0][30:])
                if mapping_flage==True:
                    VTC_count+=1
                    #count corresponding textual entities with respect to Benchmark
                    corr_vis_tex_ent=corr_vis_tex_ent+len(benchmark_data[j]['textul_entities'])
                    mapping_flage=False
                    print('successs')
                elif mapping_flage==False:
                    # swtich the classes of visual-textual entities 
                    mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(Sparql_query_YAGO,benchmark_data[j]['yago_type'][0][30:],YOLO_in_YAGO[i])
                    if mapping_flage==True:
                        mapping_flage=False
                        print('success')
                        VTC_count+=1
                        corr_vis_tex_ent=corr_vis_tex_ent+len(benchmark_data[j]['textul_entities'])
                    else:
                        print('unsuccess')

    print('Statisctics TMD+TET+VTC:',Benchmark_tot_vis_ent,Benchmark_tot_vis_tex_ent,LinKEr_tot_vis_ent,corr_vis_ent,corr_vis_tex_ent)
    return Benchmark_tot_vis_ent,Benchmark_tot_vis_tex_ent,LinKEr_tot_vis_ent,corr_vis_ent,corr_vis_tex_ent,VTC_count

def Evaluations_tex_ent_typing(Sparql_query_YAGO,LinKEr_tex_ent,LinKEr_tex_ent_YAGO_type,Bencharmk_tex_ent,Bencharmk_tex_ent_type):

    """
    This function takes predicted textual entities mentions and compare with benchmark data for textual-mention-detection and 
    YAGO typing (i.e. TMD+TET).
     
    input:
        Sparql_query_YAGO - YAGO taxonomy file
        LinKEr_tex_ent - predicted textual-entities
        LinKEr_tex_ent_YAGO_type - corresponding linked textual-entities (with YAGO)
        Bencharmk_tex_ent - Benchmark textual-entities
        Bencharmk_tex_ent_type - corresponding linked textual-entities (with YAGO)
 
     Output:
        benchmark_tot_tex_ent - VTKEL textual entities
        tot_LinKEr_tex_ent - predicted textual entities
        corr_tex_ent - entities matched by VT-LinKEr with VTKEL dataset
        
    """    
    
    benchmark_tot_tex_ent=len(Bencharmk_tex_ent_type)
    corr_tex_ent=0
    temp_LinKEr_tex_ent_YAGO=[]
    for i in range(len(LinKEr_tex_ent_YAGO_type)):
        for j in range(len(LinKEr_tex_ent_YAGO_type[i])):
            temp_LinKEr_tex_ent_YAGO.append(LinKEr_tex_ent_YAGO_type[i][j])
    tot_LinKEr_tex_ent=len(temp_LinKEr_tex_ent_YAGO)
    for i in range(len(Bencharmk_tex_ent_type)):
        for j in range(len(temp_LinKEr_tex_ent_YAGO)):
            # check if the predicted textual entities part of caption text and thena also check their YAGO type
            if Bencharmk_tex_ent_type[i] in temp_LinKEr_tex_ent_YAGO[j]:

                mapping_flage=False
                mapping_hierarchy=[]
                mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(Sparql_query_YAGO,Bencharmk_tex_ent_type[i][30:],temp_LinKEr_tex_ent_YAGO[j][30:])
                if mapping_flage==True:
                    corr_tex_ent+=1
                    print('sucess')
                    mapping_flage=False
                    break
                elif mapping_flage==False:
                    mapping_flage,mapping_hierarchy=YAGO_taxonomy_mapping(Sparql_query_YAGO,temp_LinKEr_tex_ent_YAGO[j][30:],Bencharmk_tex_ent_type[i][30:])                
                    if mapping_flage==True:
                        print('sucess')
                        corr_tex_ent+=1
                    else:
                        print('unsucess')
                        
    print('Statisctics TMD+TET:',benchmark_tot_tex_ent,tot_LinKEr_tex_ent,corr_tex_ent)
    return benchmark_tot_tex_ent,tot_LinKEr_tex_ent,corr_tex_ent

#------------------ YAGO Taxonomy .ttl file path -------------#
yago_taxonomy_file_path="/.yago_taxonomy-v1.1.ttl"
graph_1=ConjunctiveGraph()
yago_taxonomy=graph_1.parse(yago_taxonomy_file_path, format="turtle")

#------------------ upload Flickr30k-entities image folder path -------------#
img_file_counter=-1
imgs_dir_path='/.Flickr30k-Entities/images/'

#------------------ upload Flickr30k-entities xml annotations folder path -------------#
imgs_dir_path_xml='/.Flickr30k/bounding_box_annotations/.xml'

#------------------ upload VTKEL-dataset file path (1) 300 (2) 1000 or (3) 31K documents datasets -------------#
g=ConjunctiveGraph()
Benchmark_dataset = g.parse('/.VTKEL_dataset/.ttl', format='n3')


#----------------- count evaluations scores ---------------#
Benchmark_tot_vis_ent_doc=0
tot_VTC_ent=0
tot_VTC_tex=0
Benchmark_tot_vis_tex_ent_doc=0
LinKEr_tot_vis_ent_doc=0
corr_vis_ent_doc=0
corr_vis_tex_ent_doc=0
benchmark_tot_tex_ent_doc=0
LinKEr_tot_tex_ent_doc=0
corr_tex_ent_doc=0

#----------------- sequentially upload one-by-one .xml file for evaluations ---------------#
for filename in os.listdir(imgs_dir_path_xml):

    img_file_counter+=1    
    if filename.endswith(".xml"):
        # image no.
        print('====================================================\n',img_file_counter,':','image file->',filename,'---------------------')
        image_path=imgs_dir_path+filename[:-4]+'.jpg'
        image = cv2.imread(image_path)
        img_id=filename
                
        # read the Flickr30k image-caption .txt file
        img_id=img_id[:-4]
        image_captions=get_sentence_data('.../Flickr30k_caption/'+img_id+'.txt')

        # image captions
        print('C0:',image_captions[0]['sentence'])
        print('C1:',image_captions[1]['sentence'])
        print('C2:',image_captions[2]['sentence'])
        print('C3:',image_captions[3]['sentence'])
        print('C4:',image_captions[4]['sentence'])
        
        benchmark_data=Benchmark_visual_data(Benchmark_dataset,img_id)

        Bencharmk_tex_ent,Bencharmk_tex_ent_type=Benchmark_tex_ent_data(Benchmark_dataset,img_id)

        #----------------- PIKES System Phase ---------------# 

        textual_entities=[]
        tex_ent_YAGO_type=[]
        textual_entities,tex_ent_YAGO_type=Textual_ent_detec_linking()
        
        #----------------- Object detection Phase ---------------# 
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        """
        Pre-train YOLO version 3 model for detecting visual objects.
        """    
        # read class names from text file
        classes = None
        with open('yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet('yolov3_.weights', 'yolov3.cfg')
        
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

        #----------------- Visual-Textual alignment Phase ---------------# 
        
        #----------------- Evaluations (VMD+VET) ---------------#
        Benchmark_tot_vis_ent,Benchmark_tot_vis_tex_ent,LinKEr_tot_vis_ent,corr_vis_ent,corr_vis_tex_ent,VTC_count=Evaluations_VTC(yago_taxonomy,benchmark_data,bounding_boxes,YOLO_class_names,YOLO_class_in_YAGO)

        Benchmark_tot_vis_ent_doc=Benchmark_tot_vis_ent_doc+Benchmark_tot_vis_ent
        tot_VTC_ent=tot_VTC_ent+VTC_count
        Benchmark_tot_vis_tex_ent_doc=Benchmark_tot_vis_tex_ent_doc+Benchmark_tot_vis_tex_ent
        LinKEr_tot_vis_ent_doc=LinKEr_tot_vis_ent_doc+LinKEr_tot_vis_ent
        corr_vis_ent_doc=corr_vis_ent_doc+corr_vis_ent
        corr_vis_tex_ent_doc=corr_vis_tex_ent_doc+corr_vis_tex_ent
        
        #----------------- Evaluations (TMD+TET) ---------------#
        benchmark_tot_tex_ent,LinKEr_tot_tex_ent,corr_tex_ent=Evaluations_tex_ent_typing(yago_taxonomy,textual_entities,tex_ent_YAGO_type,Bencharmk_tex_ent,Bencharmk_tex_ent_type)

        benchmark_tot_tex_ent_doc=benchmark_tot_tex_ent_doc+benchmark_tot_tex_ent
        LinKEr_tot_tex_ent_doc=LinKEr_tot_tex_ent_doc+LinKEr_tot_tex_ent
        corr_tex_ent_doc=corr_tex_ent_doc+corr_tex_ent
        # YOLO class merging
        b = set()
        unique_YOLO_class_in_YAGO = []
        for x in YOLO_class_in_YAGO:
            if x not in b:
                unique_YOLO_class_in_YAGO.append(x)
                b.add(x)
                                       
        YOLO_class_names_unique=same_vis_mentions(YOLO_class_names)
     
        VTC_textual=Allignment_of_vis_tex_ent(tex_ent_YAGO_type,yago_taxonomy,YOLO_class_in_YAGO)
        tot_VTC_tex=tot_VTC_tex+VTC_textual

        #==> stored the resultant image file .jpg form
        cv2.imwrite('.../output_images/'+img_id+'_yolo.jpg', image)
        cv2.destroyAllWindows()        
        
#----------------- Results statistics ---------------#
print('\n---------------------Evaluations Statistics-----------------------------\n')
print('Total visual entities of Benchmark:',Benchmark_tot_vis_ent_doc)
print('LinKEr total visual entities:',LinKEr_tot_vis_ent_doc)
print('LinKEr-Predicted correct visual entities',corr_vis_ent_doc)

print('\nTotal benchmark textual entities',benchmark_tot_tex_ent_doc)
print('Total LinKER-predicted textual entities',LinKEr_tot_tex_ent_doc)
print('Correct textual entities',corr_tex_ent_doc) 

print('\nTotal visual and textual entities(Benchmark Corefernce->visiual+textual):',Benchmark_tot_vis_tex_ent_doc)
print('Predicted correct visual-textual entities(Corefernce->textual+visual)',corr_vis_tex_ent_doc)
print('Total VTC (Corefernce chain wrt ID->visual)',tot_VTC_ent)
print('Total VTC (Corefernce chain wrt ID->for textual)',tot_VTC_tex)
        