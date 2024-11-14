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
from src.text_embedding import TextEmbeddingModel
from utils.Turing_utils import load_Turing
from utils.Deepfake_utils import load_deepfake
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4
from src.dataset  import PassagesDataset

def infer(passages_dataloder,fabric,tokenizer,model):
    if fabric.global_rank == 0 :
        passages_dataloder=tqdm(passages_dataloder,total=len(passages_dataloder))
        allids, allembeddings,alllabels= [],[],[]
    model.model.eval()
    with torch.no_grad():
        for batch in passages_dataloder:
            text,label,write_model,write_model_set,ids= batch
            encoded_batch = tokenizer.batch_encode_plus(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        padding="max_length",
                        # padding=True,
                        truncation=True,
                    )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            # output = model(**encoded_batch).last_hidden_state
            # embeddings = pooling(output, encoded_batch)  
            # print(encoded_batch)
            embeddings = model(encoded_batch)
            # print(encoded_batch['input_ids'].shape)
            embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(1))
            label = fabric.all_gather(label).view(-1)
            ids = fabric.all_gather(ids).view(-1)
            if fabric.global_rank == 0 :
                allembeddings.append(embeddings.cpu())
                allids.extend(ids.cpu().tolist())
                alllabels.extend(label.cpu().tolist())
    if fabric.global_rank == 0 :
        allembeddings = torch.cat(allembeddings, dim=0)
        epsilon = 1e-6
        emb_dict,label_dict={},{}
        norms  = torch.norm(allembeddings, dim=1, keepdim=True) + epsilon
        allembeddings= allembeddings / norms
        for i in range(len(allids)):
            emb_dict[allids[i]]=allembeddings[i]
            label_dict[allids[i]]=alllabels[i]
        allids,allembeddings,alllabels=[],[],[]
        for key in emb_dict:
            allids.append(key)
            allembeddings.append(emb_dict[key])
            alllabels.append(label_dict[key])
        allembeddings = torch.stack(allembeddings, dim=0)
        return allids, allembeddings.numpy(),alllabels
    else:
        return [],[],[]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def test(opt):
    if opt.device_num>1:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num,strategy='ddp')
    else:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num)
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    new_state_dict={}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]]=state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer=model.tokenizer
    if opt.mode=='deepfake':
        database = load_deepfake(opt.database_path)[opt.database_name]
        test_database = load_deepfake(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='OUTFOX':
        database=load_OUTFOX(opt.database_path,opt.attack)[opt.database_name]
        test_database = load_OUTFOX(opt.test_dataset_path,opt.attack)[opt.test_dataset_name]
    elif opt.mode=='Turing':
        database=load_Turing(opt.database_path)[opt.database_name]
        test_database = load_Turing(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='M4':
        database=load_M4(opt.database_path)[opt.database_name]+load_M4(opt.database_path)[opt.database_name.replace('train','dev')]
        test_database = load_M4(opt.test_dataset_path)[opt.test_dataset_name]
        
    # database = load_deepfake('/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models')['train']
    passage_dataset = PassagesDataset(database,mode=opt.mode,need_ids=True)
    test_dataset = PassagesDataset(test_database,mode=opt.mode,need_ids=True)

    passages_dataloder = DataLoader(passage_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True)
    test_dataloder = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True)
    passages_dataloder,test_dataloder=fabric.setup_dataloaders(passages_dataloder,test_dataloder)
    model=fabric.setup(model)

    test_ids, test_embeddings,test_labels = infer(test_dataloder,fabric,tokenizer,model)
    train_ids, train_embeddings,train_labels = infer(passages_dataloder,fabric,tokenizer,model)
    fabric.barrier()

    if fabric.global_rank == 0:
        index = Indexer(opt.embedding_dim)
        index.index_data(train_ids, train_embeddings)
        label_dict={}
        for i in range(len(train_ids)):
            label_dict[train_ids[i]]=train_labels[i]

        if opt.save_database:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            index.serialize(opt.save_path)
            #save label_dict using pickle
            with open(os.path.join(opt.save_path, 'label_dict.pkl'), 'wb') as f:
                pickle.dump(label_dict, f)

        test_labels=[str(test_labels[i]) for i in range(len(test_labels))]

        preds= {i: [] for i in range(1,opt.max_K+1)}
        if len(test_embeddings.shape) == 1:
            test_embeddings = test_embeddings.reshape(1, -1)
        top_ids_and_scores = index.search_knn(test_embeddings, opt.max_K)
        for i, (ids, scores) in enumerate(top_ids_and_scores):
            zero_num,one_num=0,0
            # 将scores排序，返回排好序的下标
            sorted_scores = np.argsort(scores)
            # 从大到小排序
            sorted_scores = sorted_scores[::-1]
            for k in range(1,opt.max_K+1):
                id = ids[sorted_scores[k-1]]
                if label_dict[int(id)]==0:
                    zero_num+=1
                else:
                    one_num+=1
                if zero_num>one_num:
                    preds[k].append('0')
                else:
                    preds[k].append('1')
        K_values = list(range(1, opt.max_K+1))
        human_recs = []
        machine_recs = []
        avg_recs = []
        accs = []
        precisions = []
        recalls = []
        f1_scores = []

        for k in range(1,opt.max_K+1):
            human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(test_labels, preds[k],test_ids)
            print(f"K={k}, HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")
            human_recs.append(human_rec)
            machine_recs.append(machine_rec)
            avg_recs.append(avg_rec)
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        # Plotting each metric in a separate subplot
        axs[0, 0].plot(K_values, human_recs, marker='o', label='Human Recognition Rate')
        axs[0, 0].set_title('Human Recognition Rate')
        axs[0, 0].grid(True)

        axs[0, 1].plot(K_values, machine_recs, marker='x', label='Machine Recognition Rate')
        axs[0, 1].set_title('Machine Recognition Rate')
        axs[0, 1].grid(True)

        axs[0, 2].plot(K_values, avg_recs, marker='^', label='Average Recognition Rate')
        axs[0, 2].set_title('Average Recognition Rate')
        axs[0, 2].grid(True)

        axs[1, 0].plot(K_values, accs, marker='s', label='Accuracy')
        axs[1, 0].set_title('Accuracy')
        axs[1, 0].grid(True)

        axs[1, 1].plot(K_values, precisions, marker='p', label='Precision')
        axs[1, 1].set_title('Precision')
        axs[1, 1].grid(True)

        axs[1, 2].plot(K_values, recalls, marker='*', label='Recall')
        axs[1, 2].set_title('Recall')
        axs[1, 2].grid(True)

        axs[2, 0].plot(K_values, f1_scores, marker='D', label='F1 Score')
        axs[2, 0].set_title('F1 Score')
        axs[2, 0].grid(True)

        # Hide empty subplots
        for i in range(2, 3):
            for j in range(1, 3):
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.savefig('performance_metrics_subplot.png', dpi=300)
        max_ids=0
        for i in range(1,opt.max_K):
            if avg_recs[i]>avg_recs[max_ids]:
                max_ids=i
        print(f"Find opt.max_K is {max_ids+1}")
        print(f"HumanRec: {human_recs[max_ids]}, MachineRec: {machine_recs[max_ids]}, AvgRec: {avg_recs[max_ids]}, Acc:{accs[max_ids]}, Precision:{precisions[max_ids]}, Recall:{recalls[max_ids]}, F1:{f1_scores[max_ids]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=768)

    parser.add_argument('--mode', type=str, default='deepfake', help="deepfake,MGT or MGTDetect_CoCo")
    parser.add_argument("--database_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--database_name', type=str, default='train', help="train,valid,test,test_ood")
    parser.add_argument("--test_dataset_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--test_dataset_name', type=str, default='test', help="train,valid,test,test_ood")
    parser.add_argument("--attack", type=str, default="none", help="Attack type only for OUTFOX dataset, none,outfox,dipper")
    parser.add_argument("--model_path", type=str, default="/home/heyongxin/detect-LLM-text/DAT/pth/unseen_model/model_best_gpt35.pth",\
                         help="Path to the embedding model checkpoint")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base", help="Model name")

    parser.add_argument('--max_K', type=int, default=5, help="Search [1,K] nearest neighbors,choose the best K")
    parser.add_argument('--pooling', type=str, default="average", help="Pooling method, average or cls")
    parser.add_argument("--save_database",action='store_true',help="Save the database using faiss")
    parser.add_argument("--save_path", type=str, default="database", help="Path to save the database")
    
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    set_seed(opt.seed)
    test(opt)