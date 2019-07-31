# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:36:44 2019

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
                 
g1 = Graph()
g1.bind("vtkel",vtkel1)
g1.bind("ks",ks1)
g1.bind("owl",owl1)
g1.bind("gaf",gaf1)
g1.bind("nif",nif1)
g1.bind("prov",prov1)
#==> End of Prefixes to use for RDF graph (triples)

def Coreference_rdfTripes_from_PIKES(pikes_entities,pikes_entities_YAGO,image_id_annotation):
    out_fun=get_sentence_data('F:/PhD/VKS Flickr30k/Nov-2008/V4/Flickr30k_caption/'+image_id_annotation+'.txt')
    """
    This function takes textual entity mentions and their typed extracted by PIKES, process these entities and stored RDF triple forms.
    Input:
        pikes_entities – textual mentions extracted by PIKES for first caption
        pikes_entities_YAGO – textual mentions extracted by PIKES for second caption
        image_id_annotation – textual mentions extracted by PIKES for third caption

    Output:
        g1:return textual entity mentions and their typed extracted by PIKES in RDFs graph form

    """    
    for i in range(5):
        temp_mention_c01=''
        anchor=''
        for j in range(len(pikes_entities[i])):
            RDFC1_s=URIRef(vtkel1[image_id_annotation]+'C'+str(i)+'/#'+pikes_entities[i][j])
            RDFC1_p=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
            RDFC1_o=URIRef(ks1['TextualEntity'])
            g1.add( (RDFC1_s, RDFC1_p, RDFC1_o) )

            RDFC2_o=URIRef(pikes_entities_YAGO[i][j])
            g1.add( (RDFC1_s, RDFC1_p, RDFC2_o) )
            anchor_word_c0=''
            if pikes_entities[i][j][-2:]=='_2':
                temp_mention_c01=pikes_entities[i][j][:-2]
                anchor=temp_mention_c01
                caption_c0=out_fun[i]['sentence']
                caption_c0=TextBlob(caption_c0)
                caption_c0_words_list=caption_c0.words
                anchor_count=0

                for k in range(len(caption_c0_words_list)):
                    if anchor in caption_c0_words_list[k]:
                        anchor_count+=1
                        if anchor_count==2:
                            anchor_word_c0=caption_c0_words_list[k]
                                                      
                start_index=0
                start_index=out_fun[i]['sentence'].find(temp_mention_c01,start_index)
                start_index=out_fun[i]['sentence'].find(anchor_word_c0,start_index+1)
                
                end_index=len(anchor_word_c0)+start_index
                if len(anchor_word_c0)>0:
                    RDFC3_s=RDFC1_s
                    RDFC3_p=URIRef(gaf1['denotedBy'])
                    RDFC3_o=URIRef(vtkel1[image_id_annotation+'C'+str(i)+'/#char='+str(start_index)+','+str(end_index)])
                    g1.add( (RDFC3_s, RDFC3_p, RDFC3_o) )  

                    RDFC4_s=RDFC3_o
                    RDFC4_p=RDFC1_p
                    RDFC4_o=URIRef(ks1['TextualEntityMention'])
                    g1.add( (RDFC4_s, RDFC4_p, RDFC4_o) )
                
                    RDFC5_s=RDFC3_o
                    RDFC5_p=URIRef(ks1['mentionOf'])
                    RDFC5_o=URIRef(vtkel1[image_id_annotation]+'C'+str(i)+'/')
                    g1.add( (RDFC5_s, RDFC5_p, RDFC5_o) )
                
                    RDFC6_s=RDFC3_o
                    RDFC6_p=URIRef(nif1['anchorOf'])
                    RDFC6_o=Literal(anchor_word_c0)
                    g1.add( (RDFC6_s, RDFC6_p, RDFC6_o) )
                
                    RDFC7_s=RDFC3_o
                    RDFC7_p=URIRef(nif1['beginIndex'])
                    RDFC7_o=Literal(start_index)
                    g1.add( (RDFC7_s, RDFC7_p, RDFC7_o) )

                    RDFC8_s=RDFC3_o
                    RDFC8_p=URIRef(nif1['endIndex'])
                    RDFC8_o=Literal(end_index)
                    g1.add( (RDFC8_s, RDFC8_p, RDFC8_o) )
                
                    RDFC9_s=RDFC3_o
                    RDFC9_p=URIRef(prov1['wasAttributedTo'])
                    RDFC9_o=URIRef(vtkel1['PikesAnnotator'])
                    g1.add( (RDFC9_s, RDFC9_p, RDFC9_o) )   

            else:
                temp_mention_c01=pikes_entities[i][j]
                anchor=temp_mention_c01
                caption_c0=out_fun[i]['sentence']
                caption_c0=TextBlob(caption_c0)
                caption_c0_words_list=caption_c0.words
                anchor_count=0
                for k in range(len(caption_c0_words_list)):
                    if anchor in caption_c0_words_list[k]:
                        anchor_count+=1
                        if anchor_count==1:
                            anchor_word_c0=caption_c0_words_list[k]

                start_index=0
                start_index=out_fun[i]['sentence'].find(anchor_word_c0,start_index)
                end_index=len(anchor_word_c0)+start_index
                if len(anchor_word_c0)>0:
                    RDFC3_s=RDFC1_s
                    RDFC3_p=URIRef(gaf1['denotedBy'])
                    RDFC3_o=URIRef(vtkel1[image_id_annotation+'C'+str(i)+'/#char='+str(start_index)+','+str(end_index)])
                    g1.add( (RDFC3_s, RDFC3_p, RDFC3_o) )  
#                    print(RDFC3_s,RDFC3_p,RDFC3_o)
                    RDFC4_s=RDFC3_o
                    RDFC4_p=RDFC1_p
                    RDFC4_o=URIRef(ks1['TextualEntityMention'])
                    g1.add( (RDFC4_s, RDFC4_p, RDFC4_o) )
                
                    RDFC5_s=RDFC3_o
                    RDFC5_p=URIRef(ks1['mentionOf'])
                    RDFC5_o=URIRef(vtkel1[image_id_annotation]+'C'+str(i)+'/')
                    g1.add( (RDFC5_s, RDFC5_p, RDFC5_o) )
                
                    RDFC6_s=RDFC3_o
                    RDFC6_p=URIRef(nif1['anchorOf'])
                    RDFC6_o=Literal(anchor_word_c0)
                    g1.add( (RDFC6_s, RDFC6_p, RDFC6_o) )
                
                    RDFC7_s=RDFC3_o
                    RDFC7_p=URIRef(nif1['beginIndex'])
                    RDFC7_o=Literal(start_index)
                    g1.add( (RDFC7_s, RDFC7_p, RDFC7_o) )

                    RDFC8_s=RDFC3_o
                    RDFC8_p=URIRef(nif1['endIndex'])
                    RDFC8_o=Literal(end_index)
                    g1.add( (RDFC8_s, RDFC8_p, RDFC8_o) )
                
                    RDFC9_s=RDFC3_o
                    RDFC9_p=URIRef(prov1['wasAttributedTo'])
                    RDFC9_o=URIRef(vtkel1['PikesAnnotator'])
                    g1.add( (RDFC9_s, RDFC9_p, RDFC9_o) )   
                    
    return g1