# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:36:55 2019

@author: Shahi Dost
"""

from collections import Counter

def duplicate_visual_mentions(YOLO_class_names):
    """
    This function assigns names to two same visual objects detected by YOLO. For example if there are two people in visual objects, function will differentiable person class to to person_1 and person_2 objects.   
    input:
        YOLO_class_names – Objects detected by YOLO system 
    Output:
        list2 – Unique objects names
    """

    list1=YOLO_class_names
    list2=[]
    counts=Counter(list1)
    
    #for two cases
    First2=False
    Second2=False
    
    #for three objects
    First3=False
    Second3=False
    Third3=False
    
    #for four objects
    First4=False
    Second4=False
    Third4=False
    Fourth4=False
    
    #for five objects
    #for four objects
    First5=False
    Second5=False
    Third5=False
    Fourth5=False
    Five5=False
    Five6=False
    #double case resolving
    double_case=False
    temp_word1=[]
    
    for i in range(len(list1)):
        if counts[list1[i]]==1:
            list2.append(list1[i])       
        elif counts[list1[i]]==2:            

            if First2==False:
                First2=True
                list2.append(list1[i])
                temp_word=list1[i]
            elif Second2==False and temp_word==list1[i]:
                Second2=True
                double_case=True
                list2.append(list1[i]+'_2')
    
            elif Second2==False and temp_word!=list1[i]:
                Second2=True
                double_case=True
                list2.append(list1[i])
                temp_word1=list1[i]
                
            elif (First2==True) and (Second2==True) and (double_case==True) and (temp_word!=list1[i]):
                if temp_word!=temp_word1:
                    list2.append(list1[i]+'_2') 
                    temp_word=list1[i]
                elif temp_word==temp_word1:
                    list2.append(list1[i]+'_2') 
                 
            elif (First2==True) and (Second2==True) and (double_case==True) and (temp_word==list1[i]):
                if temp_word==list1[i] and temp_word!=temp_word1:
                    list2.append(list1[i]+'_2') 
                elif temp_word==list1[i] and temp_word!=temp_word1:
                    temp_word=list1[i]+'_2'           
    
        elif counts[list1[i]]==3:
            if First3==False:
                First3=True
                list2.append(list1[i])
            elif Second3==False:
                Second3=True
                list2.append(list1[i]+'_2')        
            elif Third3==False:
                Third3=True
                list2.append(list1[i]+'_3')   
        elif counts[list1[i]]==4:
            if First4==False:
                First4=True
                list2.append(list1[i])
            elif Second4==False:
                Second4=True
                list2.append(list1[i]+'_2')        
            elif Third4==False:
                Third4=True
                list2.append(list1[i]+'_3')        
            elif Fourth4==False:
                Fourth4=True
                list2.append(list1[i]+'_4')
        elif counts[list1[i]]==5:
            if First5==False:
                First5=True
                list2.append(list1[i])
            elif Second5==False:
                Second5=True
                list2.append(list1[i]+'_2')        
            elif Third5==False:
                Third5=True
                list2.append(list1[i]+'_3')        
            elif Fourth5==False:
                Fourth5=True
                list2.append(list1[i]+'_4')
            elif Five5==False:
                Five5=True
                list2.append(list1[i]+'_5')
        elif counts[list1[i]]>=6:
            if First5==False:
                First5=True
                list2.append(list1[i])
            elif Second5==False:
                Second5=True
                list2.append(list1[i]+'_2')        
            elif Third5==False:
                Third5=True
                list2.append(list1[i]+'_3')        
            elif Fourth5==False:
                Fourth5=True
                list2.append(list1[i]+'_4')
            elif Five5==False:
                Five5=True
                list2.append(list1[i]+'_5')
            elif Five6==False:
                Five5=True
                list2.append(list1[i]+'_6')
    return list2
