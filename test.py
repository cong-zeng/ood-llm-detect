import os
import pickle
import random
from matplotlib import pyplot as plt
from src.index import Indexer
from utils.utils import compute_metrics
import torch
import numpy as np
from torch.utils.data import DataLoader
from lightning import Fabric
from tqdm import tqdm
import argparse
# from src.deep_SVDD import SimCLR_Classifier_SCL
from utils.Turing_utils import load_Turing
from utils.Deepfake_utils import load_deepfake
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4
from utils.raid_utils import load_raid
from src.dataset  import PassagesDataset
from utils.utils import best_threshold_by_f1
from transformers import AutoTokenizer
from torch.utils.data.dataloader import default_collate

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, \
                    accuracy_score, precision_score, f1_score, recall_score

def collate_fn(batch):
    text,label,write_model,write_model_set = default_collate(batch)
    encoded_batch = tokenizer.batch_encode_plus(
        text,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True,
        )
    return encoded_batch,label,write_model,write_model_set

def infer(passages_dataloder,fabric,tokenizer,model):
    if fabric.global_rank == 0 :
        passages_dataloder=tqdm(passages_dataloder,total=len(passages_dataloder))
    model.model.eval()
    model.eval()
    with torch.no_grad():
        preds_list = []
        test_labels = []
        for batch in passages_dataloder:
            encoded_batch, label, write_model, write_model_set = batch
            encoded_batch = { k: v.cuda() for k, v in encoded_batch.items()}
            # output = model(**encoded_batch).last_hidden_state
            # embeddings = pooling(output, encoded_batch)  
            # print(encoded_batch)
            loss, out, k_out, k_outlabel = model(encoded_batch, write_model, write_model_set,label)
            # print(encoded_batch['input_ids'].shape)
            if fabric.global_rank == 0 :
                preds_list.append(out.cpu())
                test_labels.append(k_outlabel.cpu())
    return preds_list, test_labels

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def test(opt):
    if opt.ood_type == "deepsvdd":
        from src.deep_SVDD import SimCLR_Classifier_SCL
    elif opt.ood_type == "energy":
        from src.energy import SimCLR_Classifier_SCL
    else:
        AssertionError("Only support deepsvdd and energy")
    if opt.device_num>1:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num,strategy='ddp')
    else:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num)
    fabric.launch()
    model = SimCLR_Classifier_SCL(opt, fabric)
    state_dict = torch.load(opt.model_path, map_location="cpu")
    # new_state_dict={}
    # for key in state_dict.keys():
    #     if key.startswith('model.'):
    #         new_state_dict[key[6:]]=state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer=model.model.tokenizer
    if opt.mode=='deepfake':
        test_database = load_deepfake(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='OUTFOX':
        test_database = load_OUTFOX(opt.test_dataset_path,opt.attack)[opt.test_dataset_name]
    elif opt.mode=='Turing':
        test_database = load_Turing(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='M4':
        test_database = load_M4(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='raid':
        test_database = load_raid()[opt.test_dataset_name]
        
    # database = load_deepfake('/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models')['train']
    test_dataset = PassagesDataset(test_database,mode=opt.mode)

    test_dataloder = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    test_dataloder=fabric.setup_dataloaders(test_dataloder)
    model=fabric.setup(model)

    preds_list, test_labels = infer(test_dataloder,fabric,tokenizer,model)
    fabric.barrier()

    if fabric.global_rank == 0:
        pred_np = torch.cat(preds_list).view(-1).numpy()
        label_np = torch.cat(test_labels).view(-1).numpy()
        # auc = roc_auc_score(label_np, pred_np)
        # other metrics
        fpr, tpr, _ = roc_curve(label_np, pred_np)
        roc_auc = auc(fpr, tpr)

        precision_, recall_, _ = precision_recall_curve(label_np, pred_np)
        pr_auc = auc(recall_, precision_)

        target_fpr = 0.05
        tpr_at_fpr_5 = np.interp(target_fpr, fpr, tpr)

        threshold, f1 = best_threshold_by_f1(label_np, pred_np)
        y_pred = np.where(pred_np>threshold,1,0)
        acc = accuracy_score(label_np, y_pred)
        precision = precision_score(label_np, y_pred)
        recall = recall_score(label_np, y_pred)
        f1 = f1_score(label_np, y_pred)
        print(f"Val, AUC: {roc_auc}, pr_auc: {pr_auc}, tpr_at_fpr_5: {tpr_at_fpr_5}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--projection_size', type=int, default=768, help="Pretrained model output dim")

    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")

    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--b', type=float, default=1) 
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--d', type=float, default=1,help="classifier loss weight")
    parser.add_argument('--classifier_dim', type=int, default=2,help="classifier out dim")
    parser.add_argument("--AA",action='store_true',help="task for finding text source")

    parser.add_argument("--R", type=float, default=0.0, help="DeepSVDD HP R")
    parser.add_argument("--nu", type=float, default=0.1, help="DeepSVDD HP nu")
    parser.add_argument("--C", default=None, help="DeepSVDD HP C")
    parser.add_argument("--objective", type=str, default="one-class", help="one-class,soft-boundary")
    parser.add_argument("--out_dim", type=int, default=128, help="output dim and dim of c")
    parser.add_argument("--only_classifier", action='store_true',help="only use classifier, no contrastive loss")



    parser.add_argument('--mode', type=str, default='deepfake', help="deepfake,MGT or MGTDetect_CoCo")
    parser.add_argument('--ood_type', type=str, default='deepsvdd', help="deepsvdd, energy")

    parser.add_argument("--test_dataset_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--test_dataset_name', type=str, default='test', help="train,valid,test,test_ood")
    parser.add_argument("--attack", type=str, default="none", help="Attack type only for OUTFOX dataset, none,outfox,dipper")
    parser.add_argument("--model_path", type=str, default="/home/heyongxin/detect-LLM-text/DAT/pth/unseen_model/model_best_gpt35.pth",\
                         help="Path to the embedding model checkpoint")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base", help="Model name")

    parser.add_argument('--max_K', type=int, default=5, help="Search [1,K] nearest neighbors,choose the best K")
    parser.add_argument('--pooling', type=str, default="average", help="Pooling method, average or cls")
    
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    set_seed(opt.seed)
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)

    test(opt)