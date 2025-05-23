import sys
sys.path.append('./')
import random
random.seed(42)
from tqdm import tqdm
import numpy as np
import json
import os
import argparse
from transformers import AutoTokenizer
from src.index import Indexer
from utils.utils import compute_metrics,calculate_metrics,best_threshold_by_f1
import torch
from src.dataset import PassagesDataset
from torch.utils.data import DataLoader
from src.hrn import SimCLR_Classifier_SCL

from lightning import Fabric
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import yaml
from utils.Turing_utils import load_Turing
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4
from utils.Deepfake_utils import load_deepfake
from utils.raid_utils import load_raid
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve

from utils.Deepfake_utils import deepfake_model_set,deepfake_name_dct
from utils.M4_utils import M4_model_set
from utils.raid_utils import raid_model_set,raid_name_dct

torch.random.manual_seed(42)
np.random.seed(42)

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

def train_single_classifier(model, model_set_idx, model_set_name, opt, fabric: Fabric):
    # Load training data by model set
    if opt.dataset=='deepfake':
        dataset = load_deepfake(opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='deepfake', model_set_idx=model_set_idx)
        passages_dataloder = DataLoader(passages_dataset, batch_size=opt.per_gpu_batch_size,\
                                     num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode='deepfake', model_set_idx=None)
        val_dataloder = DataLoader(val_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                            num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    elif opt.dataset=='M4':
        dataset = load_M4(opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name]+dataset[opt.database_name.replace('train','dev')],mode='M4', model_set_idx=model_set_idx)
        passages_dataloder = DataLoader(passages_dataset, batch_size=opt.per_gpu_batch_size,\
                                     num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode='M4', model_set_idx=None)
        val_dataloder = DataLoader(val_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                            num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    elif opt.dataset=='raid':
        dataset = load_raid()
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='raid', model_set_idx=model_set_idx)
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode='raid', model_set_idx=None)
        passages_dataloder = DataLoader(passages_dataset, batch_size=opt.per_gpu_batch_size,\
                                        num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
        val_dataloder = DataLoader(val_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                                num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
        
    if opt.only_classifier:
        opt.a=opt.b=opt.c=0
        opt.d=1
        opt.one_loss=True
    
    # Initialize model

    if opt.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            param.requires_grad=False
            # if 'emb' in name:
                # param.requires_grad=False
                
    if opt.d==0:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad=False

    passages_dataloder= fabric.setup_dataloaders(passages_dataloder)
    val_dataloder = fabric.setup_dataloaders(val_dataloder)
    

    # Set up optimizer and scheduler
    num_batches_per_epoch = len(passages_dataloder)
    print("num_batches_per_epoch, passage: ", len(passages_dataloder))
    print("Total samples: ", num_batches_per_epoch * opt.per_gpu_batch_size)
    warmup_steps = opt.warmup_steps
    lr = opt.lr
    total_steps = opt.total_epoch * num_batches_per_epoch- warmup_steps
    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=lr/10)
    
    optimizer = fabric.setup_optimizers(optimizer)

    # Training loop
    max_auc = 0
    print(" -------Training Model Set: {}--Model Set idx: {}-------".format(model_set_name, model_set_idx))  
    for epoch in range(opt.total_epoch):
        model.train()
        avg_loss = 0
        pbar = enumerate(passages_dataloder)

        if fabric.global_rank == 0:   
            pbar = tqdm(pbar, total = len(passages_dataloder))
            print(('\n' + '%11s' *(7)) % ('Epoch', 'GPU_mem', 'lr', 'Cur_loss', 'avg_loss', 'loss_clr', 'l_classify'))

        for i, batch in pbar:
            # gradient reset
            optimizer.zero_grad() 

            # learning rate warmup
            current_step = epoch * num_batches_per_epoch + i
            if current_step < warmup_steps:
                current_lr = lr * current_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            current_lr = optimizer.param_groups[0]['lr']

            # load batch data to GPU
            encoded_batch, label, write_model, write_model_set = batch
            encoded_batch = { k: v.cuda() for k, v in encoded_batch.items()}

            # forward pass: for different model_index Load one model and corresponding training data
            loss, loss_label, loss_classfiy, k_out, k_outlabel  = model(encoded_batch, model_set_idx, write_model, write_model_set,label)

            # backward pass and optimization
            avg_loss = (avg_loss * i + loss.item()) / (i+1)
            fabric.backward(loss) 
            # fabric.clip_gradients(model, optimizer, max_norm=1.0, norm_type=2)
            optimizer.step() 

            # log
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            if fabric.global_rank == 0:
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * 5) %
                    (f'{epoch + 1}/{opt.total_epoch}', mem, current_lr, loss.item(),avg_loss, loss_label, loss_classfiy))
                # if current_step%10==0:
                #     writer.add_scalar('model_set_idx', model_set_idx, current_step)
                #     writer.add_scalar('lr', current_lr, current_step)
                #     writer.add_scalar('loss', loss.item(), current_step)
                #     writer.add_scalar('avg_loss', avg_loss, current_step)
                #     writer.add_scalar('loss_label', loss_label.item(), current_step)
                #     writer.add_scalar('loss_classfiy', loss_classfiy.item(), current_step)
        
        #Validation
        with torch.no_grad():
            model.eval()
            pbar = enumerate(val_dataloder)
            if fabric.global_rank == 0:
                pbar = tqdm(pbar, total = len(val_dataloder))
                print(('\n' + '%11s' *(3)) % ('Model_set_id', 'GPU_mem', 'loss'))
                test_labels, pred_list = [], []


            for i, batch in pbar:
                encoded_batch, label, write_model, write_model_set = batch
                encoded_batch = { k: v.cuda() for k, v in encoded_batch.items()}
                loss, scores, k_out, k_outlabel, _ = model(encoded_batch, model_set_idx, write_model, write_model_set,label)
                
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                if fabric.global_rank == 0:
                    pred_list.append(scores.cpu().detach())
                    test_labels.append(k_outlabel.cpu().detach())
                    pbar.set_description(
                            ('%11s' * 2 + '%11.4g') % 
                            (f'{model_set_idx}', mem, loss.item()))
            # use roc_auc_score and other metrics to evaluate the performance 
            if fabric.global_rank == 0:
                pred_np = torch.cat(pred_list).view(-1)
                label_np = torch.cat(test_labels).view(-1)
                label_np = 1 - (torch.abs(torch.sign(label_np - model_set_idx)))

                auc = roc_auc_score(label_np, pred_np)
                # other metrics
                threshold, f1 = best_threshold_by_f1(label_np, pred_np)
                y_pred = np.where(pred_np>threshold,1,0)
                acc = accuracy_score(label_np, y_pred)
                precision = precision_score(label_np, y_pred)
                recall = recall_score(label_np, y_pred)
                f1 = f1_score(label_np, y_pred)
                print(f"Val, AUC: {auc}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")
                # writer.add_scalar('val_auc', auc, epoch)
                # writer.add_scalar('val_acc', acc, epoch)
                # writer.add_scalar('val_precision', precision, epoch)
                # writer.add_scalar('val_recall', recall, epoch)
                # writer.add_scalar('val_f1', f1, epoch)
                # writer.add_scalar('val_threshold', threshold, epoch)
                # writer.add_scalar('val_f1', f1, epoch)

        torch.cuda.empty_cache()
        fabric.barrier()

    #     # Save model
    #     if fabric.global_rank == 0:
    #         if auc > max_auc:
    #             max_auc = auc
    #             torch.save(model.state_dict(), os.path.join(opt.savedir, f"model_classifier_{model_set_name}_best.pth"))
    #             print("Model saved at: ", os.path.join(opt.savedir, f"model_classifier_{model_set_name}_best.pth")) 
    #         # writer.add_scalar('max_AUC', max_auc, epoch)
    #         print("[Epoch %d/%d/%d]  [loss: %0.2f] [MaxAUC: %0.4f]" %
    #                 (epoch + 1, opt.total_epoch, model_set_idx + 1, loss, max_auc))


    # # Return best model and best AUC
    # best_model = SimCLR_Classifier_SCL(opt,fabric)
    # best_model.load_state_dict(torch.load(os.path.join(opt.savedir, f"model_classifier_{model_set_name}_best.pth")))

    return model, None, None

def train(opt):
    # Initialize fabric 
    torch.set_float32_matmul_precision("medium")
    if opt.device_num>1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num,strategy=ddp_strategy)#
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num)
    fabric.launch()
    
    # Set up tensorboard writer and save directory
    if fabric.global_rank == 0 :
        for num in range(10000):
            if os.path.exists(os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num)))==False:
                opt.savedir=os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num))
                os.makedirs(opt.savedir)
                break
        if os.path.exists(os.path.join(opt.savedir,'runs'))==False:
            os.makedirs(os.path.join(opt.savedir,'runs'))
        writer = SummaryWriter(os.path.join(opt.savedir,'runs'))
        #save opt to yaml
        opt_dict = vars(opt)
        with open(os.path.join(opt.savedir,'config.yaml'), 'w') as file:
            yaml.dump(opt_dict, file, sort_keys=False)

    # Train model for each model set
    if opt.dataset=='deepfake':
        num_models = len(deepfake_model_set) - 1
        model_set_names = list(deepfake_model_set.keys())
    elif opt.dataset=='M4':
        num_models = 5
        model_set_names = list(M4_model_set.keys())
    elif opt.dataset=='raid':
        num_models = len(raid_model_set) - 1
        model_set_names = list(raid_model_set.keys())
    
    models = {}
    model = SimCLR_Classifier_SCL(opt, num_models, fabric)
    if opt.resum:
        state_dict = torch.load(opt.pth_path, map_location="cpu")
        model.model.load_state_dict(state_dict)
    model = fabric.setup_module(model)
    
    for model_set_idx in range(num_models):
        model_set_name = model_set_names[model_set_idx]
        print("Start Training Model Set: ", model_set_name)

        model, pred_np, _ = train_single_classifier(model, model_set_idx, model_set_name, opt, fabric)
    
    # Testing
    if opt.dataset=='deepfake':
        dataset = load_deepfake(opt.path)
        test_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode='deepfake', model_set_idx=None)
        test_dataloder = DataLoader(test_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                                num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    elif opt.dataset=='M4':
        dataset = load_M4(opt.path)
        test_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode='M4', model_set_idx=None)
        test_dataloder = DataLoader(test_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                                num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    elif opt.dataset=='raid':
        dataset = load_raid()
        test_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode='raid', model_set_idx=None)
        test_dataloder = DataLoader(test_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                                num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    preds_models = {}

    with torch.no_grad():
        pbar = enumerate(test_dataloder)
        if fabric.global_rank == 0:
            pbar = tqdm(pbar, total = len(test_dataloder))
            test_labels, pred_list, label_machine = [], [], []

        for i, batch in pbar:
            scores = 0
            encoded_batch, label, write_model, write_model_set = batch
            encoded_batch = { k: v.cuda() for k, v in encoded_batch.items()}
            # model = models[model_set_idx]
            model.eval()
            loss, scores, k_out, _, k_outlabel = model(encoded_batch, model_set_idx, write_model, write_model_set,label, run_all=True)
            # scores = scores / len(models)
            
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            if fabric.global_rank == 0:
                pred_list.append(scores.cpu().detach())
                # label_machine.append(k_outlabel.cpu().detach())
                test_labels.append(k_outlabel.cpu().detach())
                pbar.set_description(
                        ('%11s' * 2 + '%11.4g') % 
                        (f'{model_set_idx}', mem, loss.item()))
        
        if fabric.global_rank == 0:
            pred_np = 1 - torch.cat(pred_list).view(-1).numpy()
            label_np = torch.cat(test_labels).view(-1).numpy()
            # label_np = 1 - (torch.abs(torch.sign(label_np - model_set_idx)))
            # auc = roc_auc_score(label_np, pred_np)
            # print("Test AUC: ", auc)
            fpr, tpr, _ = roc_curve(label_np, pred_np)
            roc_auc = auc(fpr, tpr)

            precision_, recall_, _ = precision_recall_curve(label_np, pred_np)
            pr_auc = auc(recall_, precision_)

            target_fpr = 0.05
            tpr_at_fpr_5 = np.interp(target_fpr, fpr, tpr)
            target_tpr = 0.95                               # the TPR you care about
            fpr_at_tpr_95 = np.interp(target_tpr, tpr, fpr)
            # other metrics
            threshold, f1 = best_threshold_by_f1(label_np, pred_np)
            y_pred = np.where(pred_np>threshold,1,0)
            acc = accuracy_score(label_np, y_pred)
            precision = precision_score(label_np, y_pred)
            recall = recall_score(label_np, y_pred)
            f1 = f1_score(label_np, y_pred)
            print(f"Test, AUC:{roc_auc}, pr_auc: {pr_auc}, tpr_at_fpr_5: {tpr_at_fpr_5},fpr_at_tpr_95: {fpr_at_tpr_95}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")

            # Save test results
            test_results = {
                'auc': roc_auc,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            test_results_path = os.path.join(opt.savedir, f"test_results_{opt.dataset}_{opt.method}.json")
            with open(test_results_path, 'w') as f: 
                json.dump(test_results, f)
            print("Test results saved at: ", test_results_path)
            # Save model predictions
            preds_models[model_set_idx] = pred_np
            
            torch.save(model.state_dict(), os.path.join(opt.savedir, f"model_classifier_hrn.pth"))
            print("Model saved at: ", os.path.join(opt.savedir, f"model_clssifier_hrn.pth")) 
    
    torch.cuda.empty_cache()
    fabric.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=8, help="GPU number to use")
    parser.add_argument('--projection_size', type=int, default=768, help="Pretrained model output dim")
    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
    parser.add_argument('--num_workers', type=int, default=8, help="num_workers for dataloader")
    parser.add_argument("--per_gpu_batch_size", default=32, type=int, help="Batch size per GPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU for evaluation."
    )
    parser.add_argument("--R", type=float, default=0.0, help="DeepSVDD HP R")
    parser.add_argument("--nu", type=float, default=0.1, help="DeepSVDD HP nu")
    parser.add_argument("--C", default=None, help="DeepSVDD HP C")
    parser.add_argument("--objective", type=str, default="one-class", help="one-class,soft-boundary")
    parser.add_argument("--out_dim", type=int, default=128, help="output dim and dim of c")

    parser.add_argument("--dataset", type=str, default="deepfake", help="deepfake,OUTFOX,TuringBench,M4")
    parser.add_argument("--path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--database_name', type=str, default='train', help="train,valid,test,test_ood")
    parser.add_argument('--valid_dataset_name', type=str, default='valid', help="train,valid,test,test_ood")
    parser.add_argument('--test_dataset_name', type=str, default='test', help="train,valid,test,test_ood")
    parser.add_argument('--topk', type=int, default=10, help="Search topk nearest neighbors for validation")

    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--b', type=float, default=1) 
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--d', type=float, default=1,help="classifier loss weight")
    parser.add_argument('--classifier_dim', type=int, default=2,help="classifier out dim")
    parser.add_argument("--AA",action='store_true',help="task for finding text source")

    parser.add_argument("--total_epoch", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
    parser.add_argument("--eps", type=float, default=1e-6, help="eps")
    parser.add_argument("--savedir", type=str, default="./runs")
    parser.add_argument("--name", type=str, default='deepfake')

    parser.add_argument("--resum", type=bool, default=False)
    parser.add_argument("--pth_path", type=str, default='', help="resume embedding model path")

    #google/flan-t5-base 768
    #mixedbread-ai/mxbai-embed-large-v1 1024
    #princeton-nlp/unsup-simcse-roberta-base 768
    #princeton-nlp/unsup-simcse-bert-base-uncased 768
    #BAAI/bge-base-en-v1.5
    #e5-base-unsupervised 768
    #nomic-ai/nomic-embed-text-v1-unsupervised 768
    #facebook/mcontriever 768
    parser.add_argument('--model_name', type=str, default='princeton-nlp/unsup-simcse-roberta-base')
    # parser.add_argument('--freeze_layer', type=int, default=0, help="freeze layer, 0 means no freeze, 12 means all freeze,10 means freeze first 10 layers")
    parser.add_argument("--freeze_embedding_layer",action='store_true',help="freeze embedding layer")
    parser.add_argument("--one_loss",action='store_true',help="only use single contrastive loss")
    parser.add_argument("--only_classifier", action='store_true',help="only use classifier, no contrastive loss")

    parser.add_argument("--method", type=str, choices=["simclr", "dsvdd", "hrn", "energy"], default="simclr", help="Method to use")
    
    opt = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    train(opt)