# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:40:11 2019

@author: Shahi Dost
"""
def removed_duplicate_mentions(total_number_mentions,total_mentions_YAGO_class):
    """
    This function removed the duplicate entity from captions with YAGO class and stored in two arrays (one for entity mentions and second for respective YAGO class).
    Input:
        total_number_mentions – total textual entity mentions
        total_mentions_YAGO_class – respective YAGO class of entity mentions

    output:
        rdf_mentions_pikes – mentions of PIKES (after removing duplicate textual mentions)
        rdf_mentions_yago – respective YAGO class
    """
    rdf_mentions_pikes=[]
    rdf_mentions_yago=[]
    temp_mention=''
    temp_mention_yago=''
    for i in range(len(total_number_mentions)):
        temp_mention=total_number_mentions[i]
        temp_mention_yago=total_mentions_YAGO_class[i]
        
        mention_id_flag=False
        for j in range(len(rdf_mentions_pikes)):
            if temp_mention in rdf_mentions_pikes[j]:
                mention_id_flag=True
        if mention_id_flag==True:
            print('')
        else:
    #        print(temp_mention_yago)
            rdf_mentions_pikes.append(temp_mention)
            rdf_mentions_yago.append(temp_mention_yago)


    return rdf_mentions_pikes,rdf_mentions_yago

