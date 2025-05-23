import pandas as pd
import random
import tqdm
import re
import numpy as np
import os
import json

M4_model_set = {'chatGPT': 0, 'bloomz': 1, "dolly": 2, "davinci": 3, "cohere": 4, "llama2-fine-tuned": 5,  "jais-30b": 6, "GPT4": 7, "human": 8}

def load_M4(filefoleder, machine_text_only=False):
    data_new = {
    }
    folder = os.listdir(filefoleder)
    for entry in folder:
        full_path = os.path.join(filefoleder, entry)
        tt=entry[:-6]
        parts = tt.split('_')
        if len(parts)==3:
            keyname = f"{parts[-1]}_{parts[-2]}"
            data_new[keyname] = []
        else:
            keyname = parts[-1]
            data_new[keyname] = []
        with open(full_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                try:
                    data = json.loads(line)
                    # machine_text  model
                    dct = {}
                    dct['text'] = data["text"]
                    # reverse the label set machine label 0, human label 1
                    if data["label"] != 0:
                        dct['label'] = 0
                    else:
                        dct['label'] = 1
                    dct['src'] = data["model"]
                    # if machine_text_only then skip human text
                    if machine_text_only and dct['label'] == 1:
                        continue
                    data_new[keyname].append(dct)
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i}: {e}")
                    continue
    for key in data_new:
        data_new[key] = process_data_MGT(data_new[key])
        # only use 10% of the data
        # random.shuffle(data_new[key])
        # data_new[key] = data_new[key][:int(len(data_new[key]) * 0.01)]
    return data_new
  
           

def process_data_MGT(dataset):
    data_list=[]
    for i in range(len(dataset)):
        text,label,src=dataset[i]['text'],str(dataset[i]['label']),dataset[i]['src']
        data_list.append((text,label,src,i))
    return data_list
