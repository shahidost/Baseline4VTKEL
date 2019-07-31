# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:28:31 2019

@author: Shahi Dost
"""

def YOLO_classes_to_YAGO(YOLO_class_names0):
    """
    This function mapped YOLO system to YOLO-knowledge base system, which mean the objects detected by YOLO system will be mapped to YAGO class taxonomy and those visual objects will be linked to YAGO knowledgebase.  
    input:
        YOLO_class_names0 – Ordinary YOLO class objects 
    Output:
        YOLO_class_in_YAGO – YAGO knowledgebase objects mentions
    """           
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
    
    YOLO_class_in_YAGO=[]
    for i in range(len(YOLO_class_names0)):
        for x in COCO_TO_YAGO:
            if YOLO_class_names0[i]==x:
                YOLO_class_in_YAGO.append(COCO_TO_YAGO[x])        
           
    return YOLO_class_in_YAGO