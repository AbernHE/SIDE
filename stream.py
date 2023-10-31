import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs


max_d = 205
max_p = 545

## RNA seq 增添的代码
#编码三个为子结构:
from functools import reduce

max_r = 67

rna_dict = {}
a = ['A','C','U','G','N']  # 4种字符，匹配5种字符
k = 3
count = 1
for i in reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [a] * k):
    rna_dict[i] = count
    count = count + 1
# for i in reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [a] * c):
#     rna_dict[i] = count
#     count = count + 1
# for i in reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [a] * d):
#     rna_dict[i] = count
#     count = count + 1
# rna_dict_2 = {}
# c, d = ['A','C','U','G'], 5  # 4种字符，匹配5种字符
# count_2 = 1
# for i in reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [c] * d):
#     rna_dict_2[i] = count_2
#     count_2 = count_2 + 1

def rna_2emb_encoder(x):
    rna_len = 67
    rna = x + x[-1]
    result = []
    for i in range(0, len(rna), 3):
        query = rna[i:i+3]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * rna_len

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x) - 2):
        query = x[i:i+3]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x) - 2)

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_5ker_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x) - 4):
        query = x[i:i+5]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x) - 4)

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_1ker_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x)):
        query = x[i:i+1]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x))

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_2ker_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x)-1):
        query = x[i:i+2]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x)-1)

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_4ker_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x)-3):
        query = x[i:i+4]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x)-3)

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_5ker_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x)-4):
        query = x[i:i+5]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x)-4)

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_6ker_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x)-5):
        query = x[i:i+6]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x)-5)

    return np.asarray(result), np.asarray(input_mask)

def rna_2emb_encoder_7ker_s(x):
    # print(len(x))
    result = []
    for i in range(0, len(x)-6):
        query = x[i:i+7]
        if query not in rna_dict:
            result.append(0)
        else:
            result.append(rna_dict[query])
    
    input_mask = [1] * int(len(x)-6)

    return np.asarray(result), np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]

        #RNA相关的
        r1 = self.df.iloc[index]['Seq1']
        r2 = self.df.iloc[index]['Seq2']

        # r1_v_a, input_mask_r1_a = rna_2emb_encoder_1ker_s(r1)
        # r2_v_a, input_mask_r2_a = rna_2emb_encoder_1ker_s(r2)

        # r1_v_b, input_mask_r1_b = rna_2emb_encoder_2ker_s(r1)
        # r2_v_b, input_mask_r2_b = rna_2emb_encoder_2ker_s(r2)

        r1_v_c, input_mask_r1_c = rna_2emb_encoder_s(r1)
        r2_v_c, input_mask_r2_c = rna_2emb_encoder_s(r2)

        # r1_v = np.concatenate((r1_v_a,r1_v_b,r1_v_c),axis=0)
        # input_mask_r1 = np.concatenate((input_mask_r1_a,input_mask_r1_b,input_mask_r1_c),axis=0)
        # r2_v = np.concatenate((r2_v_a,r2_v_b,r2_v_c),axis=0)
        # input_mask_r2 = np.concatenate((input_mask_r2_a,input_mask_r2_b,input_mask_r2_c),axis=0)
        
        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.labels[index]
        return r1_v_c, r2_v_c, input_mask_r1_c, input_mask_r2_c, y