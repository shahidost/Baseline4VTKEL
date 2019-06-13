# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:36:44 2019

@author: Shahi Dost
"""

#External libraries and functions data input/out
import cv2
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from YAGO_texonomy import yago_taxo_fun, image_id_input
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

image_id=image_id_input
#==> Read image and save Id of respective image
image = cv2.imread(image_id)
image_id=image_id[85:]
image_id_annotation=image_id[:-4]
out_fun=get_sentence_data('F:/PhD/VKS Flickr30k/Nov-2008/V4/Flickr30k_caption/'+image_id_annotation+'.txt')
def Coreference_rdfTripes_from_PIKES(YOLO_class_names,YOLO_class_in_YAGO,total_number_mentions,total_mentions_YAGO_class,mention_align_pikes_c0,mention_align_pikes_c1,mention_align_pikes_c2,mention_align_pikes_c3,mention_align_pikes_c4):
    """
    This function takes textual data from PIKES, visual data from YOLO and corresponding mapped entity mentions from YAGO and process these data for entity alignment and visual-textual coreference chains resolution. All the aligned visual-textual entity mentions are stored in a RDF .ttl file in graph form with coreference chains information. This function process caption wise alignment with visual information and recognized and linked to YAGO knowledgebase.
    Input:
        YOLO_class_names – visual detected objects by YOLO
        YOLO_class_in_YAGO – corresponding YAGO class 
        total_number_mentions – complete entity mention for single image captions
        total_mentions_YAGO_class –  corresponding YAGO mapped classes
        mention_align_pikes_c0 – textual mentions extracted by PIKES for first caption
        mention_align_pikes_c1 – textual mentions extracted by PIKES for second caption
        mention_align_pikes_c2 – textual mentions extracted by PIKES for third caption
        mention_align_pikes_c3 – textual mentions extracted by PIKES for fourth caption
        mention_align_pikes_c4 – textual mentions extracted by PIKES for fifth caption

    Output:
        total_number_mentions – total textual mentions
        total_mentions_YAGO_class – corresponding YAGO classes
    """
    temp_mention_c0=''
    for i in range(len(total_number_mentions)):
        temp_mention_c0=total_number_mentions[i]
        RDF0_s=URIRef(vtkel1[image_id_annotation+'#t'+str(i+1)])
        RDF0_p=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
        RDF0_o=URIRef(ks1['TextualEntity'])
#        g1.add( (RDF0_s, RDF0_p, RDF0_o) )
        
        RDF01_s=RDF0_s
        RDF01_p=URIRef(owl1['sameAs'])
        temp_mention_c01=''
        anchor=''
###==> Caption 0:
        for j in range(len(mention_align_pikes_c0)):
            if temp_mention_c0 in mention_align_pikes_c0[j]:
                RDF01_s=URIRef(vtkel1[image_id_annotation]+'C0/#'+total_number_mentions[i])
                RDF01_p=RDF0_p
                RDF01_o=RDF0_o
                g1.add( (RDF01_s, RDF01_p, RDF01_o) )
                RDF1_s=RDF01_s
                RDF1_p=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
                RDF1_o=URIRef(total_mentions_YAGO_class[i])
                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                anchor_word_c0=''
                if temp_mention_c0[-2:]=='_2':
                    temp_mention_c01=temp_mention_c0[:-2]
                    anchor=temp_mention_c01
                    caption_c0=out_fun[0]['sentence']
                    caption_c0=TextBlob(caption_c0)
                    caption_c0_words_list=caption_c0.words
                    anchor_count=0
                    for k in range(len(caption_c0_words_list)):
                        if anchor in caption_c0_words_list[k]:
                            anchor_count+=1
                            if anchor_count==2:
#                                print('anchor->',anchor,temp_mention_c0[:],caption_c0_words_list[k])
                                anchor_word_c0=caption_c0_words_list[k]
                                                          
                    start_index=0
                    start_index=out_fun[0]['sentence'].find(temp_mention_c01,start_index)
                    start_index=out_fun[0]['sentence'].find(anchor_word_c0,start_index+1)
                    
                    end_index=len(anchor_word_c0)+start_index
                    if len(anchor_word_c0)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C0'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )  
                    
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                    
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C0/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                    
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c0)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                    
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )

                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                    
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )                    
                    
                else:
                    anchor=temp_mention_c0
                    caption_c0=out_fun[0]['sentence']
                    caption_c0=TextBlob(caption_c0)
                    caption_c0_words_list=caption_c0.words
                    anchor_count=0
                    for k in range(len(caption_c0_words_list)):
                        if anchor in caption_c0_words_list[k]:
                            anchor_count+=1
                            if anchor_count==1:
                                anchor_word_c0=caption_c0_words_list[k]

                    start_index=0
                    start_index=out_fun[0]['sentence'].find(anchor_word_c0,start_index)
                    
                    end_index=len(anchor_word_c0)+start_index
                    if len(anchor_word_c0)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C0'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )                      

                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                    
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C0/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                    
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c0)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                    
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )

                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                    
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )                    
                    
###==> Caption 1:
        for j in range(len(mention_align_pikes_c1)):
            if temp_mention_c0 in mention_align_pikes_c1[j]:
                RDF01_s=URIRef(vtkel1[image_id_annotation]+'C1/#'+total_number_mentions[i])
                RDF01_p=RDF0_p
                RDF01_o=RDF0_o
                g1.add( (RDF01_s, RDF01_p, RDF01_o) )
                RDF1_s=RDF01_s
                RDF1_p=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
                RDF1_o=URIRef(total_mentions_YAGO_class[i])
                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                anchor_word_c1=''
                if temp_mention_c0[-2:]=='_2':
                    temp_mention_c01=temp_mention_c0[:-2]
                    anchor=temp_mention_c01
                    caption_c1=out_fun[1]['sentence']
                    caption_c1=TextBlob(caption_c1)
                    caption_c1_words_list=caption_c1.words
                    anchor_count=0
                    for k in range(len(caption_c1_words_list)):
                        if anchor in caption_c1_words_list[k]:
                            anchor_count+=1
                            if anchor_count==2:
                                anchor_word_c1=caption_c1_words_list[k]
                                                          
                    start_index=0
                    start_index=out_fun[1]['sentence'].find(temp_mention_c01,start_index)
                    start_index=out_fun[1]['sentence'].find(anchor_word_c1,start_index+1)
                    
                    end_index=len(anchor_word_c1)+start_index
                    if len(anchor_word_c1)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C1'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )  
                        
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C1/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c1)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )                    
                    
                    
                else:
                    anchor=temp_mention_c0
                    caption_c1=out_fun[1]['sentence']
                    caption_c1=TextBlob(caption_c1)
                    caption_c1_words_list=caption_c1.words
                    anchor_count=0
                    for k in range(len(caption_c1_words_list)):
                        if anchor in caption_c1_words_list[k]:
                            anchor_count+=1
                            if anchor_count==1:
                                anchor_word_c1=caption_c1_words_list[k]

                    start_index=0
                    start_index=out_fun[1]['sentence'].find(anchor_word_c1,start_index)
                    end_index=len(anchor_word_c1)+start_index
                    if len(anchor_word_c1)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C1'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )                      
    
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C1/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c1)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )

###==> Caption 2:
        for j in range(len(mention_align_pikes_c2)):
            if temp_mention_c0 in mention_align_pikes_c2[j]:
                RDF01_s=URIRef(vtkel1[image_id_annotation]+'C2/#'+total_number_mentions[i])
                RDF01_p=RDF0_p
                RDF01_o=RDF0_o
                g1.add( (RDF01_s, RDF01_p, RDF01_o) )
                RDF1_s=RDF01_s
                RDF1_p=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
                RDF1_o=URIRef(total_mentions_YAGO_class[i])
                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                anchor_word_c2=''
                if temp_mention_c0[-2:]=='_2':
                    temp_mention_c01=temp_mention_c0[:-2]
                    anchor=temp_mention_c01
                    caption_c2=out_fun[2]['sentence']
                    caption_c2=TextBlob(caption_c2)
                    caption_c2_words_list=caption_c2.words
                    anchor_count=0
                    for k in range(len(caption_c2_words_list)):
                        if anchor in caption_c2_words_list[k]:
                            anchor_count+=1
                            if anchor_count==2:
                                anchor_word_c2=caption_c2_words_list[k]
                                                          
                    start_index=0
                    start_index=out_fun[2]['sentence'].find(temp_mention_c01,start_index)
                    start_index=out_fun[2]['sentence'].find(anchor_word_c2,start_index+1)
                    
                    end_index=len(anchor_word_c2)+start_index
                    if len(anchor_word_c2)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C2'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )  
                        
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C2/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c2)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )                    
                    
                    
                else:
                    anchor=temp_mention_c0
                    caption_c2=out_fun[2]['sentence']
                    caption_c2=TextBlob(caption_c2)
                    caption_c2_words_list=caption_c2.words
                    anchor_count=0
                    for k in range(len(caption_c2_words_list)):
                        if anchor in caption_c2_words_list[k]:
                            anchor_count+=1
                            if anchor_count==1:
                                anchor_word_c2=caption_c2_words_list[k]

                    start_index=0
                    start_index=out_fun[2]['sentence'].find(anchor_word_c2,start_index)
                    
                    end_index=len(anchor_word_c2)+start_index
                    if len(anchor_word_c2)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C2'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )                      
    
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id[:-4]]+'C2/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c2)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )
###==> Caption 3:
        for j in range(len(mention_align_pikes_c3)):
            if temp_mention_c0 in mention_align_pikes_c3[j]:
                RDF01_s=URIRef(vtkel1[image_id_annotation]+'C3/#'+total_number_mentions[i])
                RDF01_p=RDF0_p
                RDF01_o=RDF0_o
                g1.add( (RDF01_s, RDF01_p, RDF01_o) )
                RDF1_s=RDF01_s
                RDF1_p=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
                RDF1_o=URIRef(total_mentions_YAGO_class[i])
                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                anchor_word_c3=''
                if temp_mention_c0[-2:]=='_2':
                    temp_mention_c01=temp_mention_c0[:-2]
                    anchor=temp_mention_c01
                    caption_c3=out_fun[3]['sentence']
                    caption_c3=TextBlob(caption_c3)
                    caption_c3_words_list=caption_c3.words
                    anchor_count=0
                    for k in range(len(caption_c3_words_list)):
                        if anchor in caption_c3_words_list[k]:
                            anchor_count+=1
                            if anchor_count==2:
                                anchor_word_c3=caption_c3_words_list[k]
                                                          
                    start_index=0
                    start_index=out_fun[3]['sentence'].find(temp_mention_c01,start_index)
                    start_index=out_fun[3]['sentence'].find(anchor_word_c3,start_index+1)
                    
                    end_index=len(anchor_word_c3)+start_index
                    if len(anchor_word_c3)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C3'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )  
                        
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C3/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c3)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )                    
                    
                    
                else:
                    anchor=temp_mention_c0
                    caption_c3=out_fun[3]['sentence']
                    caption_c3=TextBlob(caption_c3)
                    caption_c3_words_list=caption_c3.words
                    anchor_count=0
                    for k in range(len(caption_c3_words_list)):
                        if anchor in caption_c3_words_list[k]:
                            anchor_count+=1
                            if anchor_count==1:
                                anchor_word_c3=caption_c3_words_list[k]

                    start_index=0
                    start_index=out_fun[3]['sentence'].find(anchor_word_c3,start_index)
                    
                    end_index=len(anchor_word_c3)+start_index
                    if len(anchor_word_c3)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C3'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )                      
    
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C3/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c3)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )
###==> Caption 4:
        for j in range(len(mention_align_pikes_c4)):
            if temp_mention_c0 in mention_align_pikes_c4[j]:
                RDF01_s=URIRef(vtkel1[image_id_annotation]+'C4/#'+total_number_mentions[i])
                RDF01_p=RDF0_p
                RDF01_o=RDF0_o
                g1.add( (RDF01_s, RDF01_p, RDF01_o) )
                RDF1_s=RDF01_s
                RDF1_p=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
                RDF1_o=URIRef(total_mentions_YAGO_class[i])
                g1.add( (RDF1_s, RDF1_p, RDF1_o) )
                anchor_word_c4=''
                if temp_mention_c0[-2:]=='_2':
                    temp_mention_c01=temp_mention_c0[:-2]
                    anchor=temp_mention_c01
                    caption_c4=out_fun[4]['sentence']
                    caption_c4=TextBlob(caption_c4)
                    caption_c4_words_list=caption_c4.words
                    anchor_count=0
                    for k in range(len(caption_c4_words_list)):
                        if anchor in caption_c4_words_list[k]:
                            anchor_count+=1
                            if anchor_count==2:
                                anchor_word_c4=caption_c4_words_list[k]
                                                          
                    start_index=0
                    start_index=out_fun[4]['sentence'].find(temp_mention_c01,start_index)
                    start_index=out_fun[4]['sentence'].find(anchor_word_c4,start_index+1)
                    
                    end_index=len(anchor_word_c4)+start_index
                    if len(anchor_word_c4)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C4'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )  
                        
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C4/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c4)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )                    
                    
                    
                else:
                    anchor=temp_mention_c0
                    caption_c4=out_fun[4]['sentence']
                    caption_c4=TextBlob(caption_c4)
                    caption_c4_words_list=caption_c4.words
                    anchor_count=0
                    for k in range(len(caption_c4_words_list)):
                        if anchor in caption_c4_words_list[k]:
                            anchor_count+=1
                            if anchor_count==1:
                                anchor_word_c4=caption_c4_words_list[k]

                    start_index=0
                    start_index=out_fun[4]['sentence'].find(anchor_word_c4,start_index)
                    
                    end_index=len(anchor_word_c4)+start_index
                    if len(anchor_word_c4)>0:
                        RDF2_s=RDF01_s
                        RDF2_p=URIRef(gaf1['denotedBy'])
                        RDF2_o=URIRef(vtkel1[image_id_annotation+'C4'+'/#char='+str(start_index)+','+str(end_index)])
                        g1.add( (RDF2_s, RDF2_p, RDF2_o) )                      
    
                        RDF3_s=RDF2_o
                        RDF3_p=RDF1_p
                        RDF3_o=URIRef(ks1['TextualEntityMention'])
                        g1.add( (RDF3_s, RDF3_p, RDF3_o) )
                        
                        RDF4_s=RDF2_o
                        RDF4_p=URIRef(ks1['mentionOf'])
                        RDF4_o=URIRef(vtkel1[image_id_annotation]+'C4/')
                        g1.add( (RDF4_s, RDF4_p, RDF4_o) )
                        
                        RDF5_s=RDF2_o
                        RDF5_p=URIRef(nif1['anchorOf'])
                        RDF5_o=Literal(anchor_word_c4)
                        g1.add( (RDF5_s, RDF5_p, RDF5_o) )
                        
                        RDF6_s=RDF2_o
                        RDF6_p=URIRef(nif1['beginIndex'])
                        RDF6_o=Literal(start_index)
                        g1.add( (RDF6_s, RDF6_p, RDF6_o) )
    
                        RDF7_s=RDF2_o
                        RDF7_p=URIRef(nif1['endIndex'])
                        RDF7_o=Literal(end_index)
                        g1.add( (RDF7_s, RDF7_p, RDF7_o) )
                        
                        RDF8_s=RDF2_o
                        RDF8_p=URIRef(prov1['wasAttributedTo'])
                        RDF8_o=URIRef(vtkel1['PikesAnnotator'])
                        g1.add( (RDF8_s, RDF8_p, RDF8_o) )
                    
    return g1,total_number_mentions,total_mentions_YAGO_class