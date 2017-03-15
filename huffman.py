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


#%%


def l2_norm(array):
    #compute sum of squares
    addup = 0
    for i in array:
        addup = addup + i**2
    return addup


def snr(wave_data,quantized_data):
    #compute quantization snr
    return 10 * math.log(l2_norm(wave_data)/l2_norm(wave_data-quantized_data),2)


def get_prob_list(quantized_data):
    #calculate list_prob from quantized_data
    for_prob = np.append(quantized_data,np.array(gen_quant_array()))
    d = collections.Counter(for_prob)
    a = np.array(list(d.items()))
    newarray = sorted(a, key=lambda x:(x[0]))
    a = np.array(newarray)
    list_prob = list(a.T[1])
    return list_prob


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



def gen_quant_array():
    #generate quantization array
    quant_vector = list([0])
    max_volume = 32768
    sum_step = 0
    for i in range(8):
        step = 0.5**(8-i)*max_volume
        for j in range(8):
            quant_vector.append(sum_step+step*(1/8)*(j+1))
        sum_step = quant_vector[-1]
    
    double_quant = quant_vector.copy()
    double_quant.reverse()
    double_quant =  list(np.array(double_quant) * (-1))
    double_quant.extend(quant_vector[1:])
    return double_quant


def half_search(quantizer,vol):
    #a half search function, return the nearest level for vol
    l = len(quantizer)
    if l == 2:
        if quantizer[0] + quantizer[1] > 2 * vol:
            return quantizer[0]
        else:
            return quantizer[1]
    elif vol > quantizer[math.floor(l/2)]:
        return half_search(quantizer[math.floor(l/2):],vol)
    else:
        return half_search(quantizer[:math.ceil(l/2)],vol)


def quantization(wave_data):
    #
    quantizer = gen_quant_array()
    quantized_data = []
    for i in wave_data:
        quantized_data.append(half_search(quantizer,i))
    
    quantized_data = np.array(quantized_data)
    quantization_error = l2_norm(quantized_data-wave_data)/np.size(quantized_data)
    print("quantization error:", quantization_error)
    return quantized_data, quantization_error


    
    
#%% read
f = wave.open(r"/Users/shijianzhong/Documents/github/huffman encoding/female/SX368.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = f.readframes(nframes)
f.close()
wave_data = np.fromstring(str_data, dtype=np.short)

#%% process
target_data, error = quantization(wave_data)
list_prob = get_prob_list(target_data)
list_code = gen_coding(list_prob)
entropy = entropy(list_prob)
print("entropy:",entropy)
average_length = average_length(list_prob,list_code)
print("average length:",average_length)
#%% write
f = wave.open(r"output.wav", "wb")
f.setnchannels(nchannels)
f.setsampwidth(sampwidth)
f.setframerate(framerate)
save_data = target_data.astype(np.short)
f.writeframes(save_data.tostring())
f.close()
        
