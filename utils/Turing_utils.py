import pandas as pd
import random
import tqdm
import re
import numpy as np
import os
import json

turing_name_dct = {
    'OpenAI-GPT': ['gpt3', 'gpt2_large', 'gpt2_pytorch', 'gpt2_medium', 'gpt2_small', 'gpt2_xl', 'gpt1'],
    'Grover': ['grover_mega', 'grover_large', 'grover_base'],
    'XLNet': ['xlnet_base', 'xlnet_large'],
    'Facebook-FAIR': ['fair_wmt19', 'fair_wmt20'],
    'PPLM': ['pplm_gpt2', 'pplm_distil'],
    'Xlm': ['xlm'],
    'Ctrl': ['ctrl'],
    'Transfo_xl': ['transfo_xl'],
    'Human': ['human']
}
turing_model_set = {'OpenAI-GPT': 0, 'Grover': 1, 'XLNet': 2, 'Facebook-FAIR': 3, 'PPLM': 4, 'Xlm': 5, 'Ctrl': 6,
             'Transfo_xl': 7, 'Human': 8}

def trim_quotes(s):
    return s.strip("\"'")

def process_spaces(text):
    text=text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return trim_quotes(text)

def load_Turing(file_folder=None):
    data={
        'train':[],
        'test':[],
        'valid':[]
    }
    folder=os.listdir(file_folder)
    for now in folder:
        if now[-3:]!='csv':
            continue
        full_path=os.path.join(file_folder,now)
        keyname=now.split('.')[0]
        assert keyname in data.keys(), f'{keyname} is not in data.keys()'
        now_data=pd.read_csv(full_path)
        for i in range(len(now_data)):
            text,src=now_data.iloc[i]['Generation'],now_data.iloc[i]['label']
            label= '1' if src=='human' else '0'
            data[keyname].append((process_spaces(str(text)),label,src,i))
    
    return data
