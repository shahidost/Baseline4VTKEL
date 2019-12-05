# -*- coding: utf-8 -*-
"""
Created on Monday August 05 11:19:47 2019

@author: Shahi Dost

    This is the main script to run VT-LinKEr system. TBW

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
from Bounding_boxes_annotations import *
from VTKEL_annotations import *
from Gender.AgeGender import *


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
    This function takes a natural language sentence and passed through ‘PIKES server’ for knowledge graph extraction 
    input:
      img_caption – input natural language text
    
    output:
      .ttl file – a turtle RDF format output file,  which stored the knowledge graph of natural language in Triples form

    """
    return requests.get(PUBLIC_PIKES_SERVER+"text2rdf?",{'text':img_caption})

def get_entities_from_text(text):
    """
    This function extract RDF graph for textual entities, their YAGO type, beginindex and endindex, processed by PIKES tool.
    input:
      text –  Image caption of Flickr30k entities dataset
    
    output:
     sparql query result  – stored textual entities recognized and linked by PIKES

    """

    pikes_answer = pikes_text2rdf(text.lower())
    
    g = ConjunctiveGraph()
    g.parse(data = pikes_answer.content.decode('utf-8'),format="trig")
    #Sparql query for entities information extraction
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
    This function detects and links textual entities from image captions (five captions per image) using the PIKES tool and saves it into .ttl file.  
    input:
        image_id - Flickr30k dataset image id
    output:
        textual_entities - Textual entities detected by PIKES
        textual_entities_YAGO_type - Textual entities YAGO types linked by PIKES
        """    
    temp_textual_entity=[]
    temp_textual_entity_type=[]

    #TED and TET using PIKES
    textual_entities=[]
    textual_entities_YAGO_type=[]
    
    #TEM and indexed extracting and saving into .ttl file
    for i in range(5):
        caption=image_captions[i]['sentence']
        caption_entities_index=get_entities_from_text(caption)
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
    This function takes two YAGO classes (i.e. Woman110787470, Person100007846) and finds if they are in the same or sub-class hierarchy by 
    mapping YAGO taxonomy and vice-versa.  
    
    input:
      Sparql_query_YAGO – Sparql query for YAGO taxonomy
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
Pre-train YOLO version 3 (aspect ration = 416x416) model for detecting visual objects.
"""    

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
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Pre-train YOLO version 3 model for detecting visual objects.
    """    
    """
    This function takes x, y, w, h, image.jpg and image id received from YOLO model and draw the bounding boxes.  
    
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

def same_visual_mentions_handleing(visual_entities):
    """
    This function assigns names to two or more than two same visual objects detected by YOLO. For example, if there are two people in 
    visual objects, this function will differentiable person class between person_1 and person_2 objects.   
    
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

def allignment_of_visual_textual_entities(image_captions,image_id,YOLO_class_names,YOLO_in_YAGO,yago_taxonomy,textual_entities,textual_entities_YAGO_type):

    """
    This function takes visual entity mentions detected by YOLO object detector and textual entity mentions detected by PIKES
    tool and make alignment using YAGO taxonomy mapping. YAGO taxonomy mapping is done by, processing visual-textual classes
    (i.e. visual and textual entity YAGO-type) and mapped through YAGO taxonomy for same-class or sub-class hierarchy detection.
     
    input:
        image_captions - Flickr30k dataset image caption text
        image_id - Flickr30k dataset image id
        YOLO_class_names - YOLO object labe (with respect to gender classification if having person label)
        YOLO_in_YAGO - YOLO label YAGO-type
        yago_taxonomy - YAGO taxonomy parsed file
        textual_entities - Textual entity mentions process by PIKES
        textual_entities_YAGO_type - Textual entity mentions typed in YAGO process by PIKES
 
     Output:
        
    """    
    for i in range(5):
        
        #create temporary list for VED, VET, TED and TET
        temp_VED=YOLO_class_names.copy()
        temp_VET=YOLO_in_YAGO.copy()
        temp_TED=textual_entities[i]
        temp_TET=textual_entities_YAGO_type[i]
        switch_flag=False
        VE_length=len(YOLO_class_names)-1

        while len(temp_VED)>=1 and VE_length>=0:
            VE_temp=temp_VET[VE_length]
            index_VE=VE_length
            VE_length-=1
            # find alignment between VET and TET one by one
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

                    #stored CRC into .tll file for VT-LinKEr annotations
                    if temp_TE_class!=temp_VE_class and switch_flag==False:
                        g1.add( (URIRef(yago[temp_TE_class]), URIRef(rdfs['subClassOf']), URIRef(yago[temp_VE_class])) )
                    elif temp_TE_class!=temp_VE_class and switch_flag==True:
                        g1.add( (URIRef(yago[temp_VE_class]), URIRef(rdfs['subClassOf']), URIRef(yago[temp_TE_class])) )
                        
                    uri_textual_entity=URIRef(vtkel[image_id+'C'+str(i)]+'/#'+temp_TED[k])
                    uri_visual_entity=URIRef(vtkel[image_id+'I'+'/#'+temp_VED[index_VE]])
                    g1.add( (uri_textual_entity, URIRef(owl['sameAs']), uri_visual_entity) )                    
                    g1.add( (uri_textual_entity, URIRef(rdf['type']), URIRef(yago[temp_TE_class])) )

                    temp_VED.pop(index_VE)
                    temp_VET.pop(index_VE)
                    temp_TED.pop(k)
                    temp_TET.pop(k)
                    break

            if mapping_flage==False:
                temp_VED.pop(index_VE)
                temp_VET.pop(index_VE)            


def gender_classification(gender_data, bboxes,YOLO_in_YAGO,YOLO_class_names,bounding_boxes):

    """
    This function takes gender data from the Gender-classification algorithm and compared it with the YOLO person label to classify 
    the person class between Male and Female. If the face detected by gender-classifier overlap with person bounding box then we mark
    it correct (person->male/female).
     
    input:
        gender_data - gender classifier data from algorithm 
        bboxes -  gender face bounding boxes data
        YOLO_in_YAGO -  YAGO typed of YOLO
        YOLO_class_names - YOLO class labels
        bounding_boxes - YOLO bounding boxes data
 
     Output:
        YOLO_in_YAGO - YAGO typed YOLO class lables (with respect to gender-classifier)
        YOLO_class_names - YOLO class lables (with respect to gender-classifier)
    """        
    
    for i in range(len(gender_data)):
        for j in range(len(YOLO_in_YAGO)):
            if 'person' in YOLO_class_names[j]:

                if (bboxes[i][0]>=bounding_boxes[j][0] and bboxes[i][1]>=bounding_boxes[j][1] and (bboxes[i][2]>=bounding_boxes[j][0] and bboxes[i][2]<=bounding_boxes[j][2]) and (bboxes[i][3]>=bounding_boxes[j][1] and bboxes[i][3]<=bounding_boxes[j][3]) ):

                    if 'Male' in gender_data[i]:
                        YOLO_in_YAGO[j]='Male109624168'
                        YOLO_class_names[j]='male'
                    elif 'Female' in gender_data[i]:
                        YOLO_in_YAGO[j]='Female109619168'
                        YOLO_class_names[j]='female'

    return YOLO_in_YAGO,YOLO_class_names
        
#path file of YAGO Taxonomy .ttl file path
yago_taxonomy_file_path='/root-directory/yago_taxonomy-v1.1.ttl'

#parsing YAGO taxonomy file
graph_1=ConjunctiveGraph()
yago_taxonomy=graph_1.parse(yago_taxonomy_file_path, format="turtle")

image_file_counter=0

#upload image folder path
images_directory_path='/root-directory/Image-Folder/'

for filename in os.listdir(images_directory_path):

    image_file_counter+=1    
    if filename.endswith(".jpg"):
        
        print('====================================================\n',image_file_counter,':','image file->',filename,'---------------------')
        image_path=images_directory_path+filename
        image = cv2.imread(image_path)
        image_id=filename
                
        #==> read the image caption file
        image_id=image_id[:-4]
        
        #Flickr30k captions directory
        image_captions=get_sentence_data('/root-directory/Flickr30k_caption/'+image_id+'.txt')

        #==> dispaly image-captions
        print('----------------------------------------------------\nImage captions processing....\n')
        print('C0:',image_captions[0]['sentence'])
        print('C1:',image_captions[1]['sentence'])
        print('C2:',image_captions[2]['sentence'])
        print('C3:',image_captions[3]['sentence'])
        print('C4:',image_captions[4]['sentence'])
                
        # PIKES tool processing phase
        print('\n-------------------------------------------------\nPIKES processing....')
        textual_entities=[]
        textual_entities_YAGO_type=[]
        textual_entities,textual_entities_YAGO_type=Textual_entities_detetction_linking(image_id)
        
        # YOLO tool processing phase
        print('\n-------------------------------------------------\nYOLO objects detection processing....')  
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        """
        Pre-train YOLO version 3 model for detecting visual objects.
        """    
        # read class names from text file (YOLO-tool helping files paths)
        classes = None
        with open('/root-directory/yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet('/root-directory/yolov31.weights', '/root-directory/yolov3.cfg')
        
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

        # Visual-Textual alignment (VTC) phase processing
        print('\n-------------------------------------------------\nAlignment processing....')   

        # differentiate same visual entities mentions (i.e. person, person -> person_1 & person_2)                               
        YOLO_class_names_unique=same_visual_mentions_handleing(YOLO_class_names)

        # Gender identification phase processing
        gender_data, bboxes=Gender_main(image_path)
        YOLO_in_YAGO_gender,YOLO_class_names_gender=gender_classification(gender_data, bboxes,YOLO_class_in_YAGO,YOLO_class_names_unique,bounding_boxes)


        graph_2=Bounding_boxes_annotations(bounding_boxes,YOLO_class_in_YAGO,YOLO_class_names_unique,image_id)

        # add to the existing RDF graphs
        g1=g1+graph_2
                
        # Visual-textual alignment function-call
        allignment_of_visual_textual_entities(image_captions,image_id,YOLO_class_names_gender,YOLO_in_YAGO_gender,yago_taxonomy,textual_entities,textual_entities_YAGO_type)
        
        # META data information RDF triples
        RDF1_s=URIRef(vtkel)
        g1.add( (URIRef(vtkel), URIRef(rdf['type']), URIRef("http://purl.org/dc/dcmitype/Software")) )

        # autherization
        authors=Literal("Shahi Dost & Luciano Serafini")
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/creator"), authors) )        

        #time and language information
        t= datetime.datetime.now()
        creation_time=Literal( str(t.year)+'-'+str(t.month)+'-'+str(t.day)+':'+'-'+str(t.hour)+':'+str(t.minute)+':'+str(t.second))
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/created"), creation_time) )
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/language"), URIRef("http://lexvo.org/id/iso639-3/eng")) )

        # VT-LinKEr triple
        uri_VT_LinKEr_title=Literal("Visual Textual Linker of Entities with Knowledge (VT-LinKEr)")
        g1.add( (URIRef(vtkel), URIRef("http://purl.org/dc/terms/title"), uri_VT_LinKEr_title) )
        g1.add( (URIRef(vtkel["#"+image_id]), URIRef(rdf['type']), URIRef("http://purl.org/dc/dcmitype/Collection")) )
        
        # VTKEL annoations function call                      
        graph_3=VTKEL_annotations(image_id,image_captions)
        
        # add to the existing RDF graphs
        g1=g1+graph_3
        print('\nend of VT-LinkEr tool! :)')

        # stored the resultant image files into .jpg form
        cv2.imwrite('/root-directory/output-images/'+image_id+'_VT-LinKEr'+'image_file_counter'+'.jpg', image)
        cv2.destroyAllWindows()        

# stored VT-LinKEr output annotated file in .ttl formate        
g1.serialize(destination='/root-directory/annoation-folder/VT-LinKEr_.ttl', format='turtle')
        
        