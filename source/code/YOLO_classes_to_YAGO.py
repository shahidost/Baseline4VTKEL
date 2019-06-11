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

    YOLO_class_in_YAGO=[]
    for i in range(len(YOLO_class_names0)):
        if YOLO_class_names0[i]=='person':
            YOLO_class_in_YAGO.append('Person100007846')
        elif YOLO_class_names0[i]=='airplane':
            YOLO_class_in_YAGO.append('Airplane102691156')          
        elif YOLO_class_names0[i]=='apple':
            YOLO_class_in_YAGO.append('Apple107739125')
        elif YOLO_class_names0[i]=='backpack':
            YOLO_class_in_YAGO.append('Backpack102769748')
        elif YOLO_class_names0[i]=='banana':
            YOLO_class_in_YAGO.append('Banana112352287')
        elif YOLO_class_names0[i]=='bat':
            YOLO_class_in_YAGO.append('Bat102806379')
        elif YOLO_class_names0[i]=='glove':
            YOLO_class_in_YAGO.append('Glove103441112')
        elif YOLO_class_names0[i]=='bear':
            YOLO_class_in_YAGO.append('Bear102131653')
        elif YOLO_class_names0[i]=='bed':
            YOLO_class_in_YAGO.append('Bed102818832')
        elif YOLO_class_names0[i]=='bench':
            YOLO_class_in_YAGO.append('Bench102828884')
        elif YOLO_class_names0[i]=='bicycle':
            YOLO_class_in_YAGO.append('Bicycle102834778')
        elif YOLO_class_names0[i]=='bird':
            YOLO_class_in_YAGO.append('Bird101503061')
        elif YOLO_class_names0[i]=='boat':
            YOLO_class_in_YAGO.append('Boat102858304')
        elif YOLO_class_names0[i]=='book':
            YOLO_class_in_YAGO.append('Book106410904')
        elif YOLO_class_names0[i]=='bottle':
            YOLO_class_in_YAGO.append('Bottle102876657')
        elif YOLO_class_names0[i]=='bowl':
            YOLO_class_in_YAGO.append('Bowl102881193')
        elif YOLO_class_names0[i]=='broccoli':
            YOLO_class_in_YAGO.append('Broccoli111876803')
        elif YOLO_class_names0[i]=='bus':
            YOLO_class_in_YAGO.append('Bus1029241116')
        elif YOLO_class_names0[i]=='cake':
            YOLO_class_in_YAGO.append('Cake102937469')
        elif YOLO_class_names0[i]=='car':
            YOLO_class_in_YAGO.append('Car102958343')
        elif YOLO_class_names0[i]=='carrot':
            YOLO_class_in_YAGO.append('Carrot112937678')
        elif YOLO_class_names0[i]=='cat':
            YOLO_class_in_YAGO.append('Cat11021211620')
        elif YOLO_class_names0[i]=='phone':
            YOLO_class_in_YAGO.append('Telephone104401088')
        elif YOLO_class_names0[i]=='chair':
            YOLO_class_in_YAGO.append('Chair103001627')
        elif YOLO_class_names0[i]=='clock':
            YOLO_class_in_YAGO.append('Clock103046257')
        elif YOLO_class_names0[i]=='couch':
            YOLO_class_in_YAGO.append('Sofa104256520')
        elif YOLO_class_names0[i]=='cow':
            YOLO_class_in_YAGO.append('Cow102403454')
        elif YOLO_class_names0[i]=='cup':
            YOLO_class_in_YAGO.append('Cup103147509')
        elif YOLO_class_names0[i]=='table':
            YOLO_class_in_YAGO.append('DiningTable103201208')
        elif YOLO_class_names0[i]=='dog':
            YOLO_class_in_YAGO.append('Dog102084071')
        elif YOLO_class_names0[i]=='donut':
            YOLO_class_in_YAGO.append('Doughnut107639069')
        elif YOLO_class_names0[i]=='elephant':
            YOLO_class_in_YAGO.append('Elephant102503517')
        elif YOLO_class_names0[i]=='hydrant':
            YOLO_class_in_YAGO.append('Hydrant103550916')
        elif YOLO_class_names0[i]=='fork':
            YOLO_class_in_YAGO.append('Fork103383948')
        elif YOLO_class_names0[i]=='frisbee':
            YOLO_class_in_YAGO.append('Frisbee4387401')
        elif YOLO_class_names0[i]=='giraffe':
            YOLO_class_in_YAGO.append('Giraffe102439033')
        elif YOLO_class_names0[i]=='drier':
            YOLO_class_in_YAGO.append('Dryer103251766')
        elif YOLO_class_names0[i]=='handbag':
            YOLO_class_in_YAGO.append('Bag102774152')
        elif YOLO_class_names0[i]=='horse':
            YOLO_class_in_YAGO.append('Horse102374451')
        elif YOLO_class_names0[i]=='hotdog':
            YOLO_class_in_YAGO.append('Frank107676602')
        elif YOLO_class_names0[i]=='keyboard':
            YOLO_class_in_YAGO.append('Keboard103614007')
        elif YOLO_class_names0[i]=='kite':
            YOLO_class_in_YAGO.append('Kite113382471')
        elif YOLO_class_names0[i]=='knife':
            YOLO_class_in_YAGO.append('Knife103623556')
        elif YOLO_class_names0[i]=='laptop':
            YOLO_class_in_YAGO.append('Laptop103642806')
        elif YOLO_class_names0[i]=='microwave':
            YOLO_class_in_YAGO.append('Microwave103761084')
        elif YOLO_class_names0[i]=='motorcycle':
            YOLO_class_in_YAGO.append('Motorcycle103790512')
        elif YOLO_class_names0[i]=='mouse':
            YOLO_class_in_YAGO.append('Mouse102330245')
        elif YOLO_class_names0[i]=='orange':
            YOLO_class_in_YAGO.append('Orange107747607')
        elif YOLO_class_names0[i]=='oven':
            YOLO_class_in_YAGO.append('Oven103862676')
        elif YOLO_class_names0[i]=='parking':
            YOLO_class_in_YAGO.append('ParkingMeter103891332')
        elif YOLO_class_names0[i]=='pizza':
            YOLO_class_in_YAGO.append('Pizza107873807')
        elif YOLO_class_names0[i]=='plant':
            YOLO_class_in_YAGO.append('Plant100017222')
        elif YOLO_class_names0[i]=='refrigerator':
            YOLO_class_in_YAGO.append('Refrigerator104070727')
        elif YOLO_class_names0[i]=='remote':
            YOLO_class_in_YAGO.append('RemoteControl104074963')
        elif YOLO_class_names0[i]=='sandwich':
            YOLO_class_in_YAGO.append('Sandwich107695965')
        elif YOLO_class_names0[i]=='scissors':
            YOLO_class_in_YAGO.append('Scissors104148054')
        elif YOLO_class_names0[i]=='sheep':
            YOLO_class_in_YAGO.append('Sheep102411705')
        elif YOLO_class_names0[i]=='sink':
            YOLO_class_in_YAGO.append('Sink104223580')
        elif YOLO_class_names0[i]=='skateboard':
            YOLO_class_in_YAGO.append('Skateboard104225987')
        elif YOLO_class_names0[i]=='skis':
            YOLO_class_in_YAGO.append('Ski104228054')
        elif YOLO_class_names0[i]=='snowboard':
            YOLO_class_in_YAGO.append('Snowboard1042517911')
        elif YOLO_class_names0[i]=='spoon':
            YOLO_class_in_YAGO.append('Spoon104284002')
        elif YOLO_class_names0[i]=='ball':
            YOLO_class_in_YAGO.append('Ball102778669')
        elif YOLO_class_names0[i]=='sign':
            YOLO_class_in_YAGO.append('Signboard104217882')
        elif YOLO_class_names0[i]=='suitcase':
            YOLO_class_in_YAGO.append('Bag102773838')
        elif YOLO_class_names0[i]=='surfboard':
            YOLO_class_in_YAGO.append('Surfboard104363559')
        elif YOLO_class_names0[i]=='teddy':
            YOLO_class_in_YAGO.append('Teddy104399382')
        elif YOLO_class_names0[i]=='racket':
            YOLO_class_in_YAGO.append('Racket104039381')
        elif YOLO_class_names0[i]=='tie':
            YOLO_class_in_YAGO.append('Necktie103815615')
        elif YOLO_class_names0[i]=='toaster':
            YOLO_class_in_YAGO.append('Toaster104442312')
        elif YOLO_class_names0[i]=='toilet':
            YOLO_class_in_YAGO.append('Toilet1104446521')
        elif YOLO_class_names0[i]=='toothbrush':
            YOLO_class_in_YAGO.append('Toothbrush104453156')
        elif YOLO_class_names0[i]=='light':
            YOLO_class_in_YAGO.append('TrafficLight106874185')
        elif YOLO_class_names0[i]=='train':
            YOLO_class_in_YAGO.append('Train104468005')
        elif YOLO_class_names0[i]=='truck':
            YOLO_class_in_YAGO.append('Truck104490091')
        elif YOLO_class_names0[i]=='tv':
            YOLO_class_in_YAGO.append('TelevisionReceiver104405907')
        elif YOLO_class_names0[i]=='umbrella':
            YOLO_class_in_YAGO.append('Umbrella104507155')
        elif YOLO_class_names0[i]=='vase':
            YOLO_class_in_YAGO.append('Vase104522168')
        elif YOLO_class_names0[i]=='glass':
            YOLO_class_in_YAGO.append('Glass103438257')
        elif YOLO_class_names0[i]=='zebra':
            YOLO_class_in_YAGO.append('Zebra102391049')
    return YOLO_class_in_YAGO