# -*- coding: utf-8 -*-
"""
Created on Friday September 13 11:45:21 2019

@author: Shahi Dost
"""
#import external library
import requests
import os
import cv2
from rdflib import Graph, URIRef, Literal, Namespace, ConjunctiveGraph
import datetime
from collections import Counter
import re
from KafNafParserPy import KafNafParser
import xml.etree.ElementTree as ET

#Prefixes
g1 = Graph()
dcmit=Namespace('http://purl.org/dc/dcmitype/')
flickrOntology=Namespace('http://vksflickr30k.fbk.eu/flickrOntology/')
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

#prefix binding
g1.bind("dcmit",dcmit)
g1.bind("flickrOntology",flickrOntology)
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

## connect to 'PIKES server' for knowledge graph in RDF Trig format
PUBLIC_PIKES_SERVER = 'https://knowledgestore2.fbk.eu/pikes-demo/api/'
LOCAL_PIKES_SERVER = 'http://localhost:8011/'

def pikes_text2naf(x):
    """
    Takes a input natural language sentence and passed through ‘PIKES server’ for knowledge graph extraction 
    input:
      x – input natural language text
    
    output:
      .ttl file – a turtle RDF format output file,  which stored the knowledge graph of natural language in Triples form

    """
    return requests.get(PUBLIC_PIKES_SERVER+"text2naf?",{'text':x})


def pikes_text2rdf(x):
    """
    Takes a input natural language sentence and passed through ‘PIKES server’ for knowledge graph extraction 
    input:
      x – input natural language text
    
    output:
      .ttl file – a turtle RDF format output file,  which stored the knowledge graph of natural language in Triples form

    """
    return requests.get(PUBLIC_PIKES_SERVER+"text2rdf?",{'text':x})

def get_entities_annotations_from_text2naf(text):
    """
    Takes a input natural language sentence and passed through ‘PIKES server’ for NAF annotations extraction 
    input:
      text – input natural language text
    
    output:
      pikes_answer – PIKES answer for NAF annotations

    """
    pikes_answer = pikes_text2naf(text.lower())

    return pikes_answer

def get_entities_from_text(text):
    """
    Takes a input natural language sentence and passed through ‘PIKES server’ for entity recognitions and linking tasks 
    input:
      text – input natural language text
    
    output:
      entities_annotations – Annotated recognized and linked textual entities by PIKES

    """    
    pikes_answer = pikes_text2rdf(text.lower())
    
    g = ConjunctiveGraph()
    g.parse(data = pikes_answer.content.decode('utf-8'),format="trig")
    
    entities_annotations=g.query("""
         PREFIX a: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
         PREFIX b: <http://groundedannotationframework.org/gaf#>
         PREFIX c: <http://dkm.fbk.eu/ontologies/knowledgestore#>
         PREFIX d: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
         PREFIX e: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
         PREFIX l: <http://www.w3.org/2000/01/rdf-schema#>
         SELECT *
         WHERE{

                 ?entity a:type ?o .
                 ?entity b:denotedBy ?wchar .
                 ?wchar c:mentionOf ?caption .
                 ?wchar d:beginIndex ?bindex .
                 ?wchar e:endIndex ?eindex .
                 ?entity l:label ?Label .


         }
    """)

    return entities_annotations

def pikes_textual_entities_annotation(image_id):
    """
    Takes a input document id (image) and stored in the textual entities annotations in .ttl file for producing the VTKEL dataset 
    input:
      image_id – document id
    
    output:
      pikes_entities_data – Textual entities id for further processing and storing

    """    
    print('--------------------------PIKES-------------------------')

    pikes_entities_data=[]
    pikes_entities_data_temp=[]
    start_ind=0

    for j in range(5):        
        caption=image_captions[j]['sentence']
        entities_annotations=get_entities_from_text(caption)

        for row in entities_annotations:
            for i in range(7):                    
                if re.search(r'\b'+'http://dbpedia.org/class/yago/'+r'\b',row[i]):
                    if 'class/yago' in row[i]:
                        pikes_entities_data_temp.append(str(j))
                        pikes_ent=row[2][:]
                        pikes_entities_data_temp.append(pikes_ent)
                        anchor_of=row[3][:]
                        pikes_entities_data_temp.append(anchor_of)
                        start_ind=int(row[0][:])-len(anchor_of)
                        pikes_entities_data_temp.append(str(start_ind))
                        end_index=row[0][:]
                        pikes_entities_data_temp.append(end_index)
                        pikes_entities_data.append(pikes_entities_data_temp)
                        pikes_entities_data_temp=[]
                        yago_type=row[5]
                        pikes_entity=URIRef(vtkel[image_id+'C'+str(j)+'/'+pikes_ent[21:]])
                        g1.add(( pikes_entity, URIRef(rdf['type']),URIRef(yago_type)))

                        #TEM->Textual Entity Mention
                        TEM=URIRef(vtkel[image_id+'C'+str(j)+'#char='+str(start_ind)+','+str(end_index)])
                        g1.add(( pikes_entity, URIRef(gaf['denotedBy']),TEM ))
                        g1.add(( TEM, URIRef(rdf['type']),URIRef(ks['TextualEntityMention']) ))
                        g1.add(( TEM, URIRef(ks['mentionOf']),URIRef(vtkel[image_id+'C'+str(j)]) ))
                        g1.add(( TEM, URIRef(nif['anchorOf']),Literal(anchor_of) ))
                        g1.add(( TEM, URIRef(nif['beginIndex']),Literal(start_ind) ))                        
                        g1.add(( TEM, URIRef(nif['endIndex']),Literal(end_index) ))
                        g1.add(( TEM, URIRef(prov['wasAttributedTo']),URIRef(vtkel['PikesAnnotator']) ))
    
    return pikes_entities_data
    
def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to
    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type})

        annotations.append(sentence_data)

    return annotations

def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset
    input:
      fn - full file path to the annotations file to parse
    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info

def head_from_NAF(image_id,head_finding_temp,caption):
    """
    Takes in inputs document id (image), headers values, and image caption to performed the syntactic analysis of text and find header if noun-phrase has two or more than two nouns
    input:
        image_id - document id 
        head_finding_temp - headers information from NAF file extracted by PIKES
        caption - image caption
    
    output:
      head_id_sending - extracted header id
      head_anchor_sending - extracted header anchor
    """    
    term_saving=[]
    term_noun_saving=[]
    head_id_values=[]
    head_id_values_anchor=[]
    head_id_sending=[]
    head_anchor_sending=[]
    success_flag_2=False
    success_flag_3=False
    success_flag_4=False    
    my_NAF_parser = KafNafParser('F:/PhD/VKS Flickr30k/Nov-2008/V4/Flickr30k_naf_caption/'+image_id+'C'+str(caption)+'.naf')    

    if len(head_finding_temp)==2:
        for term_text in my_NAF_parser.get_tokens():
            if len(head_finding_temp)==2:
                for i in range(len(head_finding_temp)):
                    if head_finding_temp[i][1].lower() in term_text.get_text().lower() and head_finding_temp[i][2]==term_text.get_offset():
                        head_id_values.append(head_finding_temp[i][0])
                        head_id_values_anchor.append(head_finding_temp[i][1])
                        term=term_text.get_id().replace('w','t')
                        term_saving.append(term)
    
        for dep_object in my_NAF_parser.get_dependencies():
            if len(term_saving)==1:
                head_id_sending.append(head_id_values[0])
                head_anchor_sending.append(head_id_values_anchor[0])
                success_flag_2=True
                
            elif len(term_saving)==2:
                if term_saving[0]==dep_object.get_from() and term_saving[1]==dep_object.get_to()  and success_flag_2==False:
                    head_id_sending.append(head_id_values[0])
                    head_id_sending.append(head_id_values[1])
                    head_anchor_sending.append(head_id_values_anchor[0])
                    head_anchor_sending.append(head_id_values_anchor[1])
                    success_flag_2=True
                elif term_saving[1]==dep_object.get_from() and term_saving[0]==dep_object.get_to() and success_flag_2==False:
                    
                    head_id_sending.append(head_id_values[1])
                    head_id_sending.append(head_id_values[0])
                    head_anchor_sending.append(head_id_values_anchor[1])
                    head_anchor_sending.append(head_id_values_anchor[0])
                    success_flag_2=True

        from_term1=0
        from_term2=0
        if len(term_saving)==2 and success_flag_2==False:
            for dep_object in my_NAF_parser.get_dependencies():
                if term_saving[0]==dep_object.get_from():
                    from_term1+=1
                elif term_saving[1]==dep_object.get_from():
                    from_term2+=1
            if from_term1>=from_term2:
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[1])
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[1])
            else:
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[0])

    elif len(head_finding_temp)==3:
        for term_text in my_NAF_parser.get_tokens():
            if len(head_finding_temp)==3:
                for i in range(len(head_finding_temp)):
                    if head_finding_temp[i][1].lower() in term_text.get_text().lower() and head_finding_temp[i][2]==term_text.get_offset():
                        head_id_values.append(head_finding_temp[i][0])
                        head_id_values_anchor.append(head_finding_temp[i][1])
                        term=term_text.get_id().replace('w','t')
                        term_saving.append(term)
                        term_noun_saving.append(term_text.get_text())
                        
        for dep_object in my_NAF_parser.get_dependencies():
            if len(term_saving)==3:
                if term_saving[2]==dep_object.get_from() and term_saving[1]==dep_object.get_to() and success_flag_3==False:
                    for dep_object1 in my_NAF_parser.get_dependencies():
                        if term_saving[2]==dep_object1.get_from() and term_saving[0]==dep_object1.get_to():
                            head_id_sending.append(head_id_values[2])
                            head_id_sending.append(head_id_values[1])
                            head_id_sending.append(head_id_values[0])

                            head_anchor_sending.append(head_id_values_anchor[2])
                            head_anchor_sending.append(head_id_values_anchor[1])
                            head_anchor_sending.append(head_id_values_anchor[0])                            
                            success_flag_3=True

                elif term_saving[1]==dep_object.get_from() and term_saving[2]==dep_object.get_to() and success_flag_3==False:
                    for dep_object1 in my_NAF_parser.get_dependencies():
                        if term_saving[1]==dep_object1.get_from() and term_saving[0]==dep_object1.get_to():
                            head_id_sending.append(head_id_values[1])
                            head_id_sending.append(head_id_values[2])
                            head_id_sending.append(head_id_values[0])
                            head_anchor_sending.append(head_id_values_anchor[1])
                            head_anchor_sending.append(head_id_values_anchor[2])
                            head_anchor_sending.append(head_id_values_anchor[0])
                            success_flag_3=True

                elif term_saving[0]==dep_object.get_from() and term_saving[1]==dep_object.get_to() and success_flag_3==False:
                    for dep_object1 in my_NAF_parser.get_dependencies():
                        if term_saving[0]==dep_object1.get_from() and term_saving[2]==dep_object1.get_to():
                            head_id_sending.append(head_id_values[0])
                            head_id_sending.append(head_id_values[1])
                            head_id_sending.append(head_id_values[2])
                            head_anchor_sending.append(head_id_values_anchor[0])
                            head_anchor_sending.append(head_id_values_anchor[1])
                            head_anchor_sending.append(head_id_values_anchor[2])
                            success_flag_3=True
        from_term1=0
        from_term2=0
        from_term3=0
    
        if len(term_saving)==3 and success_flag_3==False:
            for dep_object in my_NAF_parser.get_dependencies():
                if term_saving[0]==dep_object.get_from():
                    from_term1+=1
                elif term_saving[1]==dep_object.get_from():
                    from_term2+=1
                elif term_saving[2]==dep_object.get_from():
                    from_term3+=1
            if from_term1>=from_term2 and from_term1>=from_term3:
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[2])
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[2])
            elif from_term2>=from_term3 and from_term2>=from_term1:
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[2])
                head_id_sending.append(head_id_values[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[2])
                head_anchor_sending.append(head_id_values_anchor[0])
            elif from_term3>=from_term1 and from_term3>=from_term2:
                head_id_sending.append(head_id_values[2])
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[0])
                head_anchor_sending.append(head_id_values_anchor[2])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[0])

            else:
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[2])
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[2])

    elif len(head_finding_temp)>=4:
        for term_text in my_NAF_parser.get_tokens():
            if len(head_finding_temp)>=4:
                for i in range(len(head_finding_temp)):
                    if head_finding_temp[i][1].lower() in term_text.get_text().lower() and head_finding_temp[i][2]==term_text.get_offset():
                        head_id_values.append(head_finding_temp[i][0])
                        head_id_values_anchor.append(head_finding_temp[i][1])
                        term=term_text.get_id().replace('w','t')
                        term_saving.append(term)
                        term_noun_saving.append(term_text.get_text())

        for dep_object in my_NAF_parser.get_dependencies():
            if len(term_saving)>=4:
                if term_saving[0]==dep_object.get_from() and term_saving[1]==dep_object.get_to() and success_flag_4==False:
                    for dep_object1 in my_NAF_parser.get_dependencies():
                        if term_saving[0]==dep_object1.get_from() and term_saving[2]==dep_object1.get_to():
                            for dep_object2 in my_NAF_parser.get_dependencies():
                                if term_saving[0]==dep_object2.get_from() and term_saving[3]==dep_object2.get_to():
                                    head_id_sending.append(head_id_values[0])
                                    head_id_sending.append(head_id_values[1])
                                    head_id_sending.append(head_id_values[2])
                                    head_id_sending.append(head_id_values[3])                                    
                                    head_anchor_sending.append(head_id_values_anchor[0])
                                    head_anchor_sending.append(head_id_values_anchor[1])
                                    head_anchor_sending.append(head_id_values_anchor[2])
                                    head_anchor_sending.append(head_id_values_anchor[3])
                                    success_flag_4=True
                elif term_saving[1]==dep_object.get_from() and term_saving[0]==dep_object.get_to() and success_flag_4==False:
                    for dep_object1 in my_NAF_parser.get_dependencies():
                        if term_saving[1]==dep_object1.get_from() and term_saving[2]==dep_object1.get_to():
                            for dep_object2 in my_NAF_parser.get_dependencies():
                                if term_saving[1]==dep_object2.get_from() and term_saving[3]==dep_object2.get_to():
                                    head_id_sending.append(head_id_values[1])
                                    head_id_sending.append(head_id_values[0])
                                    head_id_sending.append(head_id_values[2])
                                    head_id_sending.append(head_id_values[3])
                                    
                                    head_anchor_sending.append(head_id_values_anchor[1])
                                    head_anchor_sending.append(head_id_values_anchor[0])
                                    head_anchor_sending.append(head_id_values_anchor[2])
                                    head_anchor_sending.append(head_id_values_anchor[3])
                                    success_flag_4=True

                elif term_saving[2]==dep_object.get_from() and term_saving[0]==dep_object.get_to() and success_flag_4==False:
                    for dep_object1 in my_NAF_parser.get_dependencies():
                        if term_saving[2]==dep_object1.get_from() and term_saving[1]==dep_object1.get_to():
                            for dep_object2 in my_NAF_parser.get_dependencies():
                                if term_saving[2]==dep_object2.get_from() and term_saving[3]==dep_object2.get_to():
                                    head_id_sending.append(head_id_values[2])
                                    head_id_sending.append(head_id_values[0])
                                    head_id_sending.append(head_id_values[1])
                                    head_id_sending.append(head_id_values[3])
                                    
                                    head_anchor_sending.append(head_id_values_anchor[2])
                                    head_anchor_sending.append(head_id_values_anchor[0])
                                    head_anchor_sending.append(head_id_values_anchor[1])
                                    head_anchor_sending.append(head_id_values_anchor[3])
                                    success_flag_4=True
                elif term_saving[3]==dep_object.get_from() and term_saving[0]==dep_object.get_to() and success_flag_4==False:
                    for dep_object1 in my_NAF_parser.get_dependencies():
                        if term_saving[3]==dep_object1.get_from() and term_saving[1]==dep_object1.get_to():
                            for dep_object2 in my_NAF_parser.get_dependencies():
                                if term_saving[3]==dep_object2.get_from() and term_saving[2]==dep_object2.get_to():
                                    head_id_sending.append(head_id_values[3])
                                    head_id_sending.append(head_id_values[0])
                                    head_id_sending.append(head_id_values[1])
                                    head_id_sending.append(head_id_values[2])
                                    
                                    head_anchor_sending.append(head_id_values_anchor[3])
                                    head_anchor_sending.append(head_id_values_anchor[0])
                                    head_anchor_sending.append(head_id_values_anchor[1])
                                    head_anchor_sending.append(head_id_values_anchor[2])
                                    success_flag_4=True

        from_term1=0
        from_term2=0
        from_term3=0
        from_term4=0
    
        if len(term_saving)>=4 and success_flag_3==False:
            for dep_object in my_NAF_parser.get_dependencies():
                if term_saving[0]==dep_object.get_from():
                    from_term1+=1
                elif term_saving[1]==dep_object.get_from():
                    from_term2+=1
                elif term_saving[2]==dep_object.get_from():
                    from_term3+=1
                elif term_saving[3]==dep_object.get_from():
                    from_term4+=1
            if from_term1>=from_term2 and from_term1>=from_term3 and from_term1>=from_term4:
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[2])
                head_id_sending.append(head_id_values[3])
                
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[2])
                head_anchor_sending.append(head_id_values_anchor[3])
            elif from_term2>=from_term1 and from_term2>=from_term3 and from_term2>=from_term4:
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[2])
                head_id_sending.append(head_id_values[3])
                
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[2])
                head_anchor_sending.append(head_id_values_anchor[3])
            elif from_term3>=from_term1 and from_term3>=from_term2 and from_term3>=from_term4:
                head_id_sending.append(head_id_values[2])
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[3])
                
                head_anchor_sending.append(head_id_values_anchor[2])
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[3])

            elif from_term4>=from_term1 and from_term4>=from_term2 and from_term4>=from_term3:
                head_id_sending.append(head_id_values[3])
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[2])
                
                head_anchor_sending.append(head_id_values_anchor[3])
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[2])

            else:
                head_id_sending.append(head_id_values[0])
                head_id_sending.append(head_id_values[1])
                head_id_sending.append(head_id_values[2])
                head_id_sending.append(head_id_values[3])
                
                head_anchor_sending.append(head_id_values_anchor[0])
                head_anchor_sending.append(head_id_values_anchor[1])
                head_anchor_sending.append(head_id_values_anchor[2])
                head_anchor_sending.append(head_id_values_anchor[3])

    return head_id_sending,head_anchor_sending

def Alignment_PIKES_Flickr(flickr_NP_data_temp_Caption,pikes_entities_data_Caption,image_id):
    """
    This function takes in input Flickr30k noun-phrase, captions and PIKES recognized and linked entities for 
    alignment with visual bounding boxes and also received and stored the header information. At the end stored 
    all these annotations into RDF graphs for .ttl file storage.
    input:
        flickr_NP_data_temp_Caption - Flickr30k noun-phrases 
        pikes_entities_data_Caption - PIKES recognized and linked textual entities with respect to caption(s)
        image_id - image ID
    
    output:

    """    
    np_ids_Caption=[]
    pikes_flickr_temp=[]
    pikes_flickr_align=[]
    
    for i in range(len(pikes_entities_data_Caption)):
        for j in range(5):
            for k in range(len(flickr_NP_data_temp_Caption[j])):
                if pikes_entities_data_Caption[i][2].lower() in flickr_NP_data_temp_Caption[j][k][1].lower() and int(pikes_entities_data_Caption[i][0])==j and int(pikes_entities_data_Caption[i][3])>=flickr_NP_data_temp_Caption[j][k][2] and int(pikes_entities_data_Caption[i][3])<=flickr_NP_data_temp_Caption[j][k][3]:
                    pikes_flickr_temp.append(pikes_entities_data_Caption[i][0])
                    pikes_flickr_temp.append(pikes_entities_data_Caption[i][1])
                    pikes_flickr_temp.append(pikes_entities_data_Caption[i][2])
                    pikes_flickr_temp.append(pikes_entities_data_Caption[i][3])
                    pikes_flickr_temp.append(pikes_entities_data_Caption[i][4])
                    pikes_flickr_temp.append(flickr_NP_data_temp_Caption[j][k][0])
                    pikes_flickr_temp.append(flickr_NP_data_temp_Caption[j][k][1])
                    pikes_flickr_temp.append(flickr_NP_data_temp_Caption[j][k][2])
                    pikes_flickr_temp.append(flickr_NP_data_temp_Caption[j][k][3])
                    pikes_flickr_align.append(pikes_flickr_temp)
                    pikes_flickr_temp=[]


    local1=0
    np_ids1=[]
    for i in range(len(pikes_flickr_align)):        
        if int(pikes_flickr_align[i][0])==local1:
            np_ids1.append(pikes_flickr_align[i][5])
            if i==len(pikes_flickr_align)-1:
                np_ids_Caption.append(np_ids1)
                np_ids1=[]             
        
        elif int(pikes_flickr_align[i][0])!=local1:
            np_ids_Caption.append(np_ids1)
            np_ids1=[]
            np_ids1.append(pikes_flickr_align[i][5])
            local1+=1
    head_finding_temp=[]
    head_finding=[]
    entity_count=0
    for c in range(len(np_ids_Caption)):
        same_ids_count=Counter(np_ids_Caption[c])
        for np_id in same_ids_count:
            if same_ids_count[np_id]<2:
                for i in range(len(pikes_flickr_align)):
                    if np_id in pikes_flickr_align[i][5] and c==int(pikes_flickr_align[i][0]) and int(pikes_flickr_align[i][4])>=int(pikes_flickr_align[i][7]):

                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+pikes_flickr_align[i][1][21:]])
                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

            elif same_ids_count[np_id]==2:

                entity_count=0

                for i in range(len(pikes_flickr_align)):
                    if np_id in pikes_flickr_align[i][5] and c==int(pikes_flickr_align[i][0]) and int(pikes_flickr_align[i][4])>=int(pikes_flickr_align[i][7]):

                        if c<=4:
                            entity_count+=1
                            head_finding_temp.append(pikes_flickr_align[i][1][21:])
                            head_finding_temp.append(pikes_flickr_align[i][2])
                            head_finding_temp.append(pikes_flickr_align[i][3])
                            head_finding_temp.append(pikes_flickr_align[i][4])
                            head_finding.append(head_finding_temp)
                            head_finding_temp=[]
                            if entity_count==2:
                                head_returned,head_anchor_sending=head_from_NAF(image_id,head_finding,c)
                                if len(head_returned)==1:
                                    sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                    sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[0]])
                                    g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                elif len(head_finding)==2 and len(head_returned)==2:

                                    if head_anchor_sending[0]==head_anchor_sending[1]:
                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[0]])
                                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[1]]) 
                                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                    else:
                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[0]])
                                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[1]]) 
                                        g1.add(( sameAs_entity_id, URIRef(owl['partOf']),sameAs_entity ))


                                head_finding=[]
                                head_returned=[]
                                entity_count=0

            elif same_ids_count[np_id]==3:

                head_finding=[]
                head_returned=[]
                for i in range(len(pikes_flickr_align)):
                    if np_id in pikes_flickr_align[i][5] and c==int(pikes_flickr_align[i][0]) and int(pikes_flickr_align[i][4])>=int(pikes_flickr_align[i][7]):

                        if c<=4:
                            entity_count+=1

                            head_finding_temp.append(pikes_flickr_align[i][1][21:])
                            head_finding_temp.append(pikes_flickr_align[i][2])
                            head_finding_temp.append(pikes_flickr_align[i][3])
                            head_finding_temp.append(pikes_flickr_align[i][4])

                            head_finding.append(head_finding_temp)
                            head_finding_temp=[]

                            if entity_count==3:

                                head_returned,head_anchor_sending=head_from_NAF(image_id,head_finding,c)

                                if len(head_returned)==3:

                                    if head_anchor_sending[0]==head_anchor_sending[1] or head_anchor_sending[0]==head_anchor_sending[2]:
                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[0]])
                                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[1]]) 
                                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[2]]) 
                                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                    else :
                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[0]])
                                        g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))

                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[1]]) 
                                        g1.add(( sameAs_entity_id, URIRef(owl['partOf']),sameAs_entity ))

                                        sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                        sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[2]]) 
                                        g1.add(( sameAs_entity_id, URIRef(owl['partOf']),sameAs_entity ))




                                head_finding=[]
                                head_returned=[]
                                entity_count=0

            elif same_ids_count[np_id]>=4:
                print('case 4+:',c,same_ids_count)
                head_finding=[]
                head_returned=[]
                for i in range(len(pikes_flickr_align)):
                    if np_id in pikes_flickr_align[i][5] and c==int(pikes_flickr_align[i][0]) and int(pikes_flickr_align[i][4])>=int(pikes_flickr_align[i][7]):

                        if c<=4:
                            entity_count+=1
                            head_finding_temp.append(pikes_flickr_align[i][2])
                            head_finding_temp.append(pikes_flickr_align[i][3])
                            head_finding_temp.append(pikes_flickr_align[i][4])
                            head_finding.append(head_finding_temp)
                            head_finding_temp=[]
                            if entity_count==4:
                                print(head_finding)
                                head_returned,head_anchor_sending=head_from_NAF(image_id,head_finding,c)
                                if len(head_returned)==4:

                                    sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                    sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[0]])
                                    g1.add(( sameAs_entity_id, URIRef(owl['sameAs']),sameAs_entity ))
 
                                    sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                    sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[1]]) 
                                    g1.add(( sameAs_entity_id, URIRef(owl['partOf']),sameAs_entity ))

                                    sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                    sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[2]]) 
                                    g1.add(( sameAs_entity_id, URIRef(owl['partOf']),sameAs_entity ))

                                    sameAs_entity_id=URIRef(vtkel[image_id+'#'+pikes_flickr_align[i][5]])
                                    sameAs_entity=URIRef(vtkel[str(image_id)+'C'+str(c)+'/#'+head_returned[3]]) 
                                    g1.add(( sameAs_entity_id, URIRef(owl['partOf']),sameAs_entity ))


                                head_finding=[]
                                head_returned=[]
                                entity_count=0

## Flickr30k images directory
image_file_counter=0
images_directory_path='F:/PhD/VKS Flickr30k/Nov-2008/V4/VTKEL and Flickr30k annotations/script/input images/insert/'
g1.add( (URIRef(vtkel), URIRef(rdf['type']), URIRef(dcmit['Software'])) )

## resource meta data information triples
t= datetime.datetime.now()
creation_time=Literal( str(t.year)+'-'+str(t.month)+'-'+str(t.day)+':'+'-'+str(t.hour)+':'+str(t.minute)+':'+str(t.second))
g1.add( (URIRef(vtkel), URIRef(dct['created']), creation_time) )
g1.add( (URIRef(vtkel), URIRef(dct['language']), URIRef("http://lexvo.org/id/iso639-3/eng")) )

uri_VT_LinKEr_title=Literal("Visual Textual Knowledge Entity Linking dataset")
g1.add( (URIRef(vtkel), URIRef(dct['title']), uri_VT_LinKEr_title) )

authors=Literal("Shahi Dost & Luciano Serafini")
g1.add( (URIRef(vtkel), URIRef(dct['creator']), authors) )        

g1.add( (URIRef(vtkel['FlickrAnnotator']), URIRef(rdf['type']), URIRef(prov['Agent'])) )
g1.add( (URIRef(vtkel['PikesAnnotator']), URIRef(rdf['type']), URIRef(prov['Agent'])) )

##takes one by one image file from image directory to produced VTKEL dataset
for filename in os.listdir(images_directory_path):

    image_file_counter+=1    
    if filename.endswith(".jpg"):
        
        print('====================================================\n',image_file_counter,':','image file->',filename,'---------------------')
        image_path=images_directory_path+filename
        image = cv2.imread(image_path)
        image_id=filename
                        
        #==> read the image caption file
        g1.add( (URIRef(vtkel[image_id]), URIRef(rdf['type']),URIRef(ks['Representation'])))
        g1.add( (URIRef(vtkel[image_id]), URIRef(dc['format']),Literal("image/jpeg") ))

        image_id=image_id[:-4]

        g1.add( (URIRef(vtkel['#'+str(image_id)]), URIRef(rdf['type']),URIRef(dcmit['Collection'])))

        image_captions=get_sentence_data('F:/PhD/VKS Flickr30k/Nov-2008/V4/Flickr30k_caption/'+image_id+'.txt')

        #==> image captions info storage
        flickr_NP_data_temp=[]
        flickr_NP_data_temp_C0=[]
        flickr_NP_data_temp_Caption=[]
        for c in range(5):
            for i in range(len(image_captions[c]['phrases'])):  
                caption=image_captions[c]['sentence']
                caption=caption.split()
    
                g1.add( (URIRef(vtkel[image_id+'C'+str(c)+'.txt']), URIRef(rdf['type']),URIRef(ks['Representation']) ))
                g1.add( (URIRef(vtkel[image_id+'C'+str(c)+'.txt']), URIRef(dc['format']),Literal("text/plain") ))
    
                g1.add( ( URIRef(vtkel[image_id+'C'+str(c)]), URIRef(rdf['type']),URIRef(ks['Resource']) ))
                g1.add( ( URIRef(vtkel[image_id+'C'+str(c)]), URIRef(rdf['type']),URIRef(ks['Text']) ))
                g1.add( ( URIRef(vtkel[image_id+'C'+str(c)]), URIRef(ks['storedAs']),URIRef(vtkel[image_id+'C'+str(c)+'.txt']) ))
                g1.add( ( URIRef(vtkel[image_id+'C'+str(c)]), URIRef(nif['isString']),Literal(image_captions[c]['sentence']) ))
                g1.add( ( URIRef(vtkel[image_id+'C'+str(c)]), URIRef(dct['identifier']),Literal(image_id+'C'+str(c)) ))
                g1.add( ( URIRef(vtkel[image_id+'C'+str(c)]), URIRef(dct['isPartOf']),URIRef(vtkel[image_id]) ))
    
                flickr_textual_entity_id=URIRef(vtkel[str(image_id)+'#'+str(image_captions[c]['phrases'][i]['phrase_id'])])
                g1.add( (flickr_textual_entity_id, URIRef(rdf['type']),URIRef(ks['Entity']) ))
                flickr_type=image_captions[c]['phrases'][i]['phrase_type'][0]
                g1.add( (flickr_textual_entity_id, URIRef(rdf['type']),URIRef(flickrOntology[flickr_type]) ))
                
                
                word_len=0
                start_index=0
                end_index=0
                start_index=image_captions[c]['phrases'][i]['first_word_index']
                for j in range(image_captions[c]['phrases'][i]['first_word_index']):
                    word_len=word_len+len(caption[j])
                start_index=start_index+word_len
                end_index=start_index+(len(image_captions[c]['phrases'][i]['phrase']))

                char_noun_phrase=str(image_id)+'C'+str(c)+'/#char='+str(start_index)+','+str(end_index)
                g1.add( (flickr_textual_entity_id, URIRef(gaf['denotedBy']),URIRef(vtkel[char_noun_phrase]) ))            
    
                g1.add( (URIRef(vtkel[char_noun_phrase]), URIRef(rdf['type']),URIRef(ks['TextualEntityMention'] ) ))    
                g1.add( (URIRef(vtkel[char_noun_phrase]), URIRef(nif['anchorOf']), Literal(image_captions[c]['phrases'][i]['phrase']) ))

                g1.add( (URIRef(vtkel[char_noun_phrase]), URIRef(nif['beginIndex']), Literal(start_index) ))   
                g1.add( (URIRef(vtkel[char_noun_phrase]), URIRef(nif['endIndex']), Literal(end_index) ))   
                g1.add( (URIRef(vtkel[char_noun_phrase]), URIRef(prov['wasAttributedTo']), URIRef(vtkel['FlickrAnnotator']) ))   
    
                g1.add( ( URIRef(vtkel[image_id+'C'+str(c)]), URIRef(ks['hasMention']), URIRef(vtkel[char_noun_phrase]) ))    

                flickr_NP_data_temp.append(image_captions[c]['phrases'][i]['phrase_id'])
                flickr_NP_data_temp.append(image_captions[c]['phrases'][i]['phrase'])
                flickr_NP_data_temp.append(start_index)
                flickr_NP_data_temp.append(end_index)
                flickr_NP_data_temp_C0.append(flickr_NP_data_temp)
                flickr_NP_data_temp=[]
            flickr_NP_data_temp_Caption.append(flickr_NP_data_temp_C0)
            flickr_NP_data_temp_C0=[]

        ##==> flickr30k bounding boxes phase and storing

        image_bb_annotations=get_annotations('F:/PhD/VKS Flickr30k/Nov-2008/V4/Flickr30k_bb/Annotations/'+image_id+'.xml')  

        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(rdf['type']),URIRef(ks['Resource']) ))
        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(rdf['type']),URIRef(dcmit['Image']) ))
        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(ks['storedAs']),URIRef(vtkel[image_id+'.jpg/']) ))
        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(dct['identifier']),Literal(image_id) ))
        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(dct['isPartOf']),URIRef(vtkel[image_id]) ))
        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(nfo['channels']),Literal(3) ))
        
        image = cv2.imread(image_path)
        Width = image.shape[1]
        Height = image.shape[0]
        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(nfo['horizontalResolution']),Literal(Width) ))
        g1.add( ( URIRef(vtkel[image_id+'I']), URIRef(nfo['verticalResolution']),Literal(Height) ))

        for bb_id in image_bb_annotations['boxes']:

            for bb_v in range(len(image_bb_annotations['boxes'][bb_id])):
                flickr_visual_entity_id=URIRef(vtkel[str(image_id)+'#'+str(bb_id)])
                xywh=str(image_bb_annotations['boxes'][bb_id][bb_v][0]+1)+','+str(image_bb_annotations['boxes'][bb_id][bb_v][1]+1)+','+str(image_bb_annotations['boxes'][bb_id][bb_v][2]+1)+','+str(image_bb_annotations['boxes'][bb_id][bb_v][3]+1)
                flickr_visual_entity_xywh=URIRef(vtkel[str(image_id)+'I/#xywh='+xywh])
                g1.add( (flickr_visual_entity_id, URIRef(gaf['denotedBy']),flickr_visual_entity_xywh ))

                g1.add( (flickr_visual_entity_xywh, URIRef(rdf['type']),URIRef(ks['VisualEntityMention']) ))
                g1.add( (flickr_visual_entity_xywh, URIRef(ks['xmin']),Literal(image_bb_annotations['boxes'][bb_id][bb_v][0]+1) ))
                g1.add( (flickr_visual_entity_xywh, URIRef(ks['ymin']),Literal(image_bb_annotations['boxes'][bb_id][bb_v][1]+1) ))
                g1.add( (flickr_visual_entity_xywh, URIRef(ks['xmax']),Literal(image_bb_annotations['boxes'][bb_id][bb_v][2]+1) ))
                g1.add( (flickr_visual_entity_xywh, URIRef(ks['ymax']),Literal(image_bb_annotations['boxes'][bb_id][bb_v][3]+1) ))
                g1.add( (flickr_visual_entity_xywh, URIRef(prov['wasAttributedTo']), URIRef(vtkel['FlickrAnnotator']) ))
                
                g1.add( (URIRef(vtkel[image_id+'I']), URIRef(ks['hasVisualMention']), flickr_visual_entity_xywh ))
                

            for i in range(5):
                for j in range(len(image_captions[i]['phrases'])):  
                    if bb_id==image_captions[i]['phrases'][j]['phrase_id']:
                        g1.add( ( URIRef(vtkel[image_id+'C'+str(i)]), URIRef(rdf['type']),URIRef(ks['Resource']) ))

        ### PIKES System Phase
        textual_entities=[]
        textual_entities_YAGO_type=[]
        
        pikes_entities_data_Caption=pikes_textual_entities_annotation(image_id)

        Alignment_PIKES_Flickr(flickr_NP_data_temp_Caption,pikes_entities_data_Caption,image_id)
        
        
g1.serialize(destination='F:/PhD/VKS Flickr30k/Nov-2008/V4/1 VTK ttl/1 big ttl/VTKEL_dataset.ttl', format='turtle')

