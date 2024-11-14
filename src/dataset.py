from torch.utils.data import Dataset
from utils.Deepfake_utils import deepfake_model_set,deepfake_name_dct
from utils.Turing_utils import turing_model_set,turing_name_dct

class PassagesDataset(Dataset):
    def __init__(self, dataset,mode='deepfake',need_ids=False):
        self.mode=mode
        self.dataset = dataset
        self.need_ids=need_ids
        self.classes=[]
        self.model_name_set={}
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
        else:
            LLM_name=set()
            for item in self.dataset:
                LLM_name.add(item[2])
            for i,name in enumerate(LLM_name):
                self.model_name_set[name]=(i,i)
                self.classes.append(name)
        
        print(f'there are {len(self.classes)} classes in {mode} dataset')
        print(f'the classes are {self.classes}')
    
    def get_class(self):
        return self.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text,label,src,id=self.dataset[idx]
        write_model,write_model_set=1000,1000
        for name in self.model_name_set.keys():
            if name in src:
                write_model,write_model_set=self.model_name_set[name]
                break
        assert write_model!=1000,f'write_model is empty,src is {src}'

        if self.need_ids:
            return text,int(label),int(write_model),int(write_model_set),int(id)
        else:
            return text,int(label),int(write_model),int(write_model_set)

