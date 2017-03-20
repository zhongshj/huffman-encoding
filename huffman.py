#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:43:36 2017

@author: abk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wave
import math
import collections
import string

#%%
get_bin = lambda x, n: format(x, 'b').zfill(n)

def get_freq_array(wave_data):
    #return a sorted frequency array and corresponding index
    array_prob = np.zeros(65536)
    for i in wave_data:
        array_prob[i+32768] = array_prob[i+32768] + 1
        print("add:",i)
    index = np.argsort(array_prob)
    array_prob = array_prob[index]
    return array_prob[::-1], index[::-1]-32768
    
#def get_freq_array(wave_data):
#    array_prob = np.zeros(65536)
#    for i in wave_data:
#        array_prob[i+32768] = array_prob[i+32768] + 1
#        print("add:",i)
#    #index = np.argsort(array_prob)
#    #array_prob = array_prob[index]
#    return array_prob
    
def get_part_prob(array_prob,k):
    sum_freq = 0
    for i in range(k):
        sum_freq = sum_freq + array_prob[i]
    return sum_freq/sum(array_prob)



def average_length(list_prob,list_code):
    #calculate average code length
    list_prob = list_prob/sum(list_prob)
    addup = 0
    for i in range(len(list_prob)):
        addup = addup + list_prob[i] * len(list_code[i])
    return addup


def entropy(list_prob):
    addup = 0
    list_prob = list_prob/sum(list_prob)
    for i in list_prob:
        addup = addup - i * math.log(i,2)
    return addup


def gen_coding(list_prob):
    #return huffman codes
    print(list_prob)
    #list_prob = list([0.2,0.4,0.3,0.1])
    
    # initialize variables
    list_symbol = list(range(1,len(list_prob)+1))
    df = pd.DataFrame({'prob':list_prob,'symbol':list_symbol})    
    
    #get the merge list
    merge_list0 = list([])
    merge_list1 = list([])
    while len(df) > 1:
        print("len:",len(df))
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

    
def gen_dict(list_code_o,index):
    #generate dictionary that only contains huffman codes
    list_code = list_code_o.copy()
    dic = {}
    for i in range(-32768,32768):
        try:
            dic[i] = list_code[np.where(index==i)[0][0]]
            print("exist:",i)
        except BaseException as e:
            
            #dic[i] = list(map(int,list(np.binary_repr(i, width=16))))
            print("noexist:",i)
    return dic
    
def encoding(dic,wave_data):
    #if huffman coding, code with 0 + huffman codes
    #if amplitude coding, code with 1 + (int to binary)
    bit_stream = []
    for i in wave_data:
        try:
            l = dic[i].copy()
            l.insert(0,0)
            bit_stream.extend(l)
        except BaseException as e:
            bit_stream.append(1)
            #bit_stream.extend(list(map(int,list(np.binary_repr(i, width=16)))))
            bit_stream.extend([int(value) for value in list(get_bin(i+32768, 16))])
    return bit_stream
    
def decoding(bit_stream_o,list_code,index):
    wave_list = []
    bit_stream = bit_stream_o.copy()
    count = 0
    while len(bit_stream)!=0:
        print(len(bit_stream))
        dis = bit_stream.pop(0)
        if dis == 0:
            temp = []
            find = False
            count2 = 0
            while find == False:
                bit = bit_stream.pop(0)
                count2 = count2 + 1
                if count2 > 100:
                    print(count2)
                temp.append(bit)
                #print(bit)
                for i in range(len(list_code)):
                    #print("serch list_code",i)
                    
                    if temp == list_code[i]:
                        print("found")
                        count = count+1
                        wave_list.append(index[i])
                        find = True
                        break
        else:
            print("slajdflksd")
            temp = []
            for i in range(16):
                temp.append(bit_stream.pop(0))
            wave_list.append(int(''.join(str(e) for e in temp),2)-32768)
            count = count+1
    return wave_list
    
    
#%% read
f = wave.open(r"/Users/shijianzhong/Documents/github/huffman encoding/male/SA1.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = f.readframes(nframes)
f.close()
sa1m = np.fromstring(str_data, dtype=np.short)

#%% get probability
array_prob_f1,index_f1 = get_freq_array(sa1f)
array_prob_f2,index_f2 = get_freq_array(sa2f)
array_prob_m1,index_m1 = get_freq_array(sa1m)
array_prob_m2,index_m2 = get_freq_array(sa2m)
#%% get codes
list_code_all = gen_coding(list(array_prob_all[0:4096]))



#%% get dics for encoding
dic_f1 = gen_dict(list_code_f1,index_f1)
dic_f2 = gen_dict(list_code_f2,index_f2)
dic_m1 = gen_dict(list_code_m1,index_m1)
dic_m2 = gen_dict(list_code_m2,index_m2)


#%%
bit_stream = encoding(dic_all,sa1f)
de = decoding(bit_stream,list_code_all,index_all)


#%% write
f = wave.open(r"output8.wav", "wb")
f.setnchannels(nchannels)
f.setsampwidth(sampwidth)
f.setframerate(framerate)
save_data = target_data.astype(np.short)
f.writeframes(save_data.tostring())
f.close()
        
