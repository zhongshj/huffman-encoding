#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:43:36 2017

@author: abk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
def split_prob(list_prob):
    
    print(list_prob)
    #list_prob = list([0.2,0.4,0.3,0.1])
    
    # initialize variables
    list_symbol = list(range(1,len(list_prob)+1))
    df = pd.DataFrame({'prob':list_prob,'symbol':list_symbol})    

    #get the merge list
    merge_list0 = list([])
    merge_list1 = list([])
    while len(df) > 1:
        #print("len:",len(df))
        df = df.sort_values(by="prob")
        df = df.reset_index(drop=True)
        merge_list0.append(df['symbol'][0])
        merge_list1.append(df['symbol'][1])
        df.ix[1,'prob'] = df.ix[1,'prob'] + df.ix[0,'prob']
        df = df.drop([0])
        
    print("mlist0:", merge_list0)
    print("mlist1:", merge_list1)
    #initialize code list
    list_code = list([])    
    for i in range(len(list_prob)):
        list_code.append([])
    
    group = np.array(range(1,len(list_prob)+1))
    
    #compute huffman codes
    for i in range(len(merge_list0)):   
        print("merging",merge_list0[i],"and",merge_list1[i])       
        for j in range(np.size(group)): 
            if group[j] == merge_list0[i]:
                print("insert 0 for", j+1)
                list_code[j].insert(0,0)
            if group[j] == merge_list1[i]:
                print("insert 1 for", j+1)
                list_code[j].insert(0,1)
        temp = group[merge_list0[i]-1]
        for k in range(np.size(group)):
            if group[k] == temp:
                group[k] = group[merge_list1[i]-1]

        print("group update", group)
            
    print("code:",list_code)
    return list_code
        
        
        
        
        
        