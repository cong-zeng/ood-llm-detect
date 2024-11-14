import hashlib
import pandas as pd
import random
import tqdm
import re
import numpy as np
import os
import json
from datasets import load_dataset

# dataset = load_dataset("yaful/DeepfakeTextDetect")
deepfake_name_dct = {'OpenAI-GPT':['gpt-3.5-trubo','text-davinci-002', 'text-davinci-003'],\
            'Meta-LLaMA':['13B', '30B', '65B', '7B'],\
            'GLM-130B':['GLM130B'],\
            'Google-FLAN-T5':['flan_t5_base', 'flan_t5_large','flan_t5_small', 'flan_t5_xl', 'flan_t5_xxl'],\
            'Facebook-OPT':['opt_1.3b', 'opt_125m', 'opt_13b', 'opt_2.7b', 'opt_30b', 'opt_350m', 'opt_6.7b', 'opt_iml_30b','opt_iml_max_1.3b'],\
            'BigScience':['bloom_7b','t0_11b', 't0_3b'],\
            'EleutherAI':['gpt_j','gpt_neox'],\
            'human':['human']}
deepfake_model_set ={'OpenAI-GPT':0,'Meta-LLaMA':1,'GLM-130B':2,'Google-FLAN-T5':3,\
            'Facebook-OPT':4,'BigScience':5,'EleutherAI':6,'human':7}

def find_substring_and_return_remainder(full_string, substring):
    index = full_string.find(substring)
    if index != -1:
        return full_string[index + len(substring):]
    else:
        if 'human' in full_string:
            return 'human'
    return None   

def stable_long_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    long_long_hash = (int_hash & ((1 << 63) - 1))
    return long_long_hash

def find_lst(folder_path):
    if "unseen" in folder_path:
        file_names = ["valid.csv", "train.csv", "test.csv", "test_ood.csv"]
    else:
        file_names = ["valid.csv", "train.csv", "test.csv"]
    return file_names

def process_data(dataset):
    data_list=[]
    for i in range(len(dataset)):
        text,label,src=dataset[i]['text'],str(dataset[i]['label']),dataset[i]['src']
        data_list.append((text,label,src,stable_long_hash(text)))#
    return data_list

def load_deepfake(folder_path=None):
    data_new = {
        'train': [],
        'test': [],
        'valid': [] ,# add val set
        'test_ood': []
    }
    file_names = find_lst(folder_path)
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=["text", "label", "src"])
            data_partition= file_name[:-4]
            txt_lst = df["text"].tolist()
            label_lst = df["label"].tolist()
            src_lst = df["src"].tolist()
            for index in range(len(txt_lst)):
                dct = {}
                dct['text'] = txt_lst[index]
                dct['label'] = label_lst[index]
                dct['src'] = src_lst[index]
                data_new[data_partition].append(dct)       
    
    for key in data_new:
        data_new[key] = process_data(data_new[key])

    return data_new
