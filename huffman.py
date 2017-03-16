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

def a_law_comp(array):
    a = 87.6
    e = 2.71828182845
    max_amp = 32768
    array = array/max_amp
    a_array = np.zeros(np.size(array))
    for i in range(np.size(array)):
        if array[i] >= 0 and array[i] <= 1/a:
            a_array[i] = a*array[i]/(1+math.log(a,e))
        elif array[i] > 1/a:
            a_array[i] = (1+math.log(a*array[i],e))/(1+math.log(a,e))
        elif array[i] < 0 and array[i] >= -1/a:
            a_array[i] = -1*a*array[i]/(1+math.log(a,e))
        elif array[i] < -1/a:
            a_array[i] = -1*(1+math.log(-1*a*array[i],e))/(1+math.log(a,e))
        else:
            a_array[i] = 0
    return a_array
    
def a_law_decomp(array):
    a = 87.6
    e = 2.71828182845
    b = 1+math.log(a,e)
    array = array/max(abs(array))
    a_array = np.zeros(np.size(array))
    for i in range(np.size(array)):
        if array[i] >= 0 and array[i] <= 1/b:
            a_array[i] = array[i]*b/a
        elif array[i] > 1/b:
            a_array[i] = (e**(array[i]*b-1))/a
        elif array[i] < 0 and array[i] >= -1/b:
            a_array[i] = array[i]*b/a
        elif array[i] < -1/b:
            a_array[i] = -1*(e**(-1*array[i]*b-1))/a
        else:
            a_array[i] = 0
    return a_array
    
    
def gen_quant_array(bit):
    #generate quantization array
    num = 2**bit
    interval = 2/num
    quant_array = np.arange(-1,1,interval)
    quant_array = quant_array + interval/2
    return quant_array


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


def quantization(wave_data,bit=8):
    #
    quantizer = gen_quant_array(bit)
    quantized_data = []
    for i in wave_data:
        quantized_data.append(half_search(quantizer,i))
    
    quantized_data = np.array(quantized_data)
    quantization_error = l2_norm(quantized_data-wave_data)/np.size(quantized_data)
    print("quantization error:", quantization_error)
    return quantized_data/max(abs(quantized_data))


    
    
#%% read
f = wave.open(r"/Users/shijianzhong/Documents/github/huffman encoding/male/SA1.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = f.readframes(nframes)
f.close()
sa1m = np.fromstring(str_data, dtype=np.short)

#%% process
sa1m_q = quantization(a_law_comp(sa1m))
sa2m_q = quantization(a_law_comp(sa2m))
sa1f_q = quantization(a_law_comp(sa1f))
sa2f_q = quantization(a_law_comp(sa2f))
#%%
plt.hist(sa2m_q,bins=20)
plt.title("Quantized amplitude destribution for SA2.wav(male)")
plt.savefig("2m.eps")
#%% write
f = wave.open(r"output8.wav", "wb")
f.setnchannels(nchannels)
f.setsampwidth(sampwidth)
f.setframerate(framerate)
save_data = target_data.astype(np.short)
f.writeframes(save_data.tostring())
f.close()
        
