from torch.utils.data import Dataset
import numpy as np
from utils.Deepfake_utils import deepfake_model_set,deepfake_name_dct
from utils.Turing_utils import turing_model_set,turing_name_dct
from utils.M4_utils import M4_model_set
from utils.raid_utils import raid_model_set,raid_name_dct

class PassagesDataset(Dataset):
    def __init__(self, dataset, model_set_idx=None, mode='deepfake', need_ids=False):
        self.mode=mode
        self.dataset = dataset
        self.need_ids=need_ids
        self.classes=[]
        self.model_name_set={}  # { "model_name": (model_idx, model_set_idx),
                                # ... 
                                # }

        if mode=='deepfake':
            cnt=0
            for model_set_name,model_set in deepfake_name_dct.items():
                for name in model_set:
                    self.model_name_set[name]=(cnt,deepfake_model_set[model_set_name])
                    self.classes.append(name)
                    cnt+=1
        elif mode=='Turing':
            cnt=0
            for model_set_name,model_set in turing_name_dct.items():
                for name in model_set:
                    self.model_name_set[name]=(cnt,turing_model_set[model_set_name])
                    self.classes.append(name)
                    cnt+=1
        elif mode=='raid':
            cnt=0
            for model_set_name,model_set in raid_name_dct.items():
                for name in model_set:
                    self.model_name_set[name]=(cnt,raid_model_set[model_set_name])
                    self.classes.append(name)
                    cnt+=1
        elif mode=='M4':
            model_name=set()
            for item in self.dataset:
                model_name.add(item[2])
            for i,name in enumerate(model_name):
                self.model_name_set[name]=(i,M4_model_set[name])
                self.classes.append(name)

        else:
            LLM_name=set()
            for item in self.dataset:
                LLM_name.add(item[2])
            for i,name in enumerate(LLM_name):
                self.model_name_set[name]=(i,i)
                self.classes.append(name)
        
        if model_set_idx is not None:
            print(f" -------Loading {mode} dataset for model_set_idx:{model_set_idx} -------")
            self.dataset = []
            for data in dataset:
                text,label,src,id = data
                # From data's scr to model_idx model_set_idx 
                write_model,write_model_set = self._scr_to_model(src)

                if write_model_set != model_set_idx:
                    continue
                else:
                    self.dataset.append(data)
    
        print(f'Totally, there are {len(self.classes)} classes, the classes are {self.classes}')
        print(f'there are {len(self.dataset)} samples')
    
    def get_class(self):
        return self.classes
    
    def _scr_to_model(self,src):
        write_model,write_model_set=1000,1000
        for name in self.model_name_set.keys():
            if name in src:
                write_model,write_model_set=self.model_name_set[name]
                break
        assert write_model!=1000,f'write_model is empty,src is {src}'
        return write_model,write_model_set
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text,label,src,id=self.dataset[idx]
        write_model,write_model_set = self._scr_to_model(src)
        if self.need_ids:
            return text,int(label),int(write_model),int(write_model_set),int(id)
        else:
            return text,int(label),int(write_model),int(write_model_set)

