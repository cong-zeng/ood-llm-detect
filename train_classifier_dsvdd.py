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

from src.deep_SVDD import SimCLR_Classifier_SCL


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

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().detach().cpu().numpy()), 1 - nu)

def train(opt):
    # Initialize fabric and set up data loaders
    torch.set_float32_matmul_precision("medium")
    if opt.device_num>1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num,strategy=ddp_strategy)#
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num)
    fabric.launch()
    # load dataset and dataloader
    if opt.dataset=='deepfake':
        dataset = load_deepfake(opt.path)
        machine_dataset = load_deepfake(opt.path, machine_text_only=True)
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='deepfake')
        machine_passages_dataset = PassagesDataset(machine_dataset[opt.database_name],mode='deepfake')
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='deepfake')
    elif opt.dataset=='TuringBench':
        dataset = load_Turing(file_folder=opt.path)
        machine_dataset = load_Turing(file_folder=opt.path, machine_text_only=True)

        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='Turing')
        machine_passages_dataset = PassagesDataset(machine_dataset[opt.database_name],mode='Turing')

        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='Turing')
        print("TuringBench dataset loaded!,len:",len(machine_passages_dataset))
    elif opt.dataset=='OUTFOX':
        dataset = load_OUTFOX(opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='OUTFOX')
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='OUTFOX')
    elif opt.dataset=='M4':
        dataset = load_M4(opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name]+dataset[opt.database_name.replace('train','dev')],mode='M4')

        machine_dataset = load_M4(opt.path, machine_text_only=True)
        machine_passages_dataset = PassagesDataset(machine_dataset[opt.database_name]+machine_dataset[opt.database_name.replace('train','dev')],mode='M4')
        
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='M4')
    elif opt.dataset=='raid':
        dataset = load_raid()
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='raid')
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='raid')

        machine_dataset = load_raid(machine_text_only=True)
        machine_passages_dataset = PassagesDataset(machine_dataset[opt.database_name],mode='raid')


    if opt.AA:
        opt.classifier_dim=len(passages_dataset.model_name_set)

    passages_dataloder = DataLoader(passages_dataset, batch_size=opt.per_gpu_batch_size,\
                                     num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
    machine_passages_dataloder = DataLoader(machine_passages_dataset, batch_size=opt.per_gpu_batch_size,\
                                     num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
    val_dataloder = DataLoader(val_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                            num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)

    if opt.only_classifier:
        opt.a=opt.b=opt.c=0
        opt.d=1
        opt.one_loss=True
    
    # Initialize model
    model = SimCLR_Classifier_SCL(opt,fabric)

    if opt.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            if 'emb' in name:
                param.requires_grad=False
                
    if opt.d==0:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad=False

    passages_dataloder,val_dataloder, machine_passages_dataloder = fabric.setup_dataloaders(passages_dataloder,val_dataloder, machine_passages_dataloder)
    
    if fabric.global_rank == 0 :
        for num in range(10000):
            if os.path.exists(os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num)))==False:
                opt.savedir=os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num))
                os.makedirs(opt.savedir)
                break
        if os.path.exists(os.path.join(opt.savedir,'runs'))==False:
            os.makedirs(os.path.join(opt.savedir,'runs'))
        writer = SummaryWriter(os.path.join(opt.savedir,'runs'))
        index = Indexer(opt.projection_size)
        #save opt to yaml
        opt_dict = vars(opt)
        with open(os.path.join(opt.savedir,'config.yaml'), 'w') as file:
            yaml.dump(opt_dict, file, sort_keys=False)

    # Set up optimizer and scheduler
    num_batches_per_epoch = len(passages_dataloder)
    print("passage: ", len(passages_dataloder))
    print("machine_passages_dataloder: ", len(machine_passages_dataloder))

    warmup_steps = opt.warmup_steps
    lr = opt.lr
    total_steps = opt.total_epoch * num_batches_per_epoch- warmup_steps
    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=lr/10)
    # model, optimizer = fabric.setup(model, optimizer)
    model = fabric.setup_module(model)

    # (DeepSVDD Setting)
    model.mark_forward_method('initialize_center_c')

    optimizer = fabric.setup_optimizers(optimizer)
    warm_up_n_epochs = 5  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
    
    #(DeepSVDD Setting) initialize center_c
    print("Initialize center_c!")
    model.initialize_center_c(machine_passages_dataloder)

    # Training loop
    max_auc=0
    for epoch in range(opt.total_epoch):
        model.train()
        avg_loss = 0
        pbar = enumerate(passages_dataloder)

        if fabric.global_rank == 0:
            pbar = tqdm(pbar, total = len(passages_dataloder))
            print(('\n' + '%11s' *(5)) % ('Epoch', 'GPU_mem', 'Cur_loss', 'avg_loss','lr'))

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

            # forward pass and loss calculation
            if opt.one_loss:
                loss, loss_label, loss_DeepSVDD, k_out, k_outlabel  = model(encoded_batch, write_model, write_model_set,label)
            else:
                loss, loss_model, loss_set, loss_label, loss_DeepSVDD, loss_human, k_out, k_outlabel  = model(encoded_batch,write_model,write_model_set,label)
        
            # backward pass and optimization
            avg_loss = (avg_loss * i + loss.item()) / (i+1)
            fabric.backward(loss) 

            # fabric.clip_gradients(model, optimizer, max_norm=1.0, norm_type=2)
            optimizer.step() 

            # When using the soft-boundary objective, update the radius R after warm-up epochs
            if current_step >= warmup_steps:
                schedule.step()

            # (DeepSVDD Setting)Update hypersphere radius R on mini-batch distances 
            if opt.objective == "soft-boundary" and (epoch >= warm_up_n_epochs):
                loss_DeepSVDD = fabric.all_gather(loss_DeepSVDD).mean() 
                model.R.data = torch.tensor(get_radius(loss_DeepSVDD, model.nu), device=model.device)

            # log
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            if fabric.global_rank == 0:
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * 3) %
                    (f'{epoch + 1}/{opt.total_epoch}', mem, loss.item(),avg_loss, current_lr)
                )

                if current_step%10==0:
                    writer.add_scalar('lr', current_lr, current_step)
                    writer.add_scalar('loss', loss.item(), current_step)
                    writer.add_scalar('avg_loss', avg_loss, current_step)
                    writer.add_scalar('loss_label', loss_label.item(), current_step)
                    writer.add_scalar('loss_DeepSVDD', loss_DeepSVDD.item(), current_step)
                    if opt.one_loss==False:
                        writer.add_scalar('loss_model', loss_model.item(), current_step)
                        writer.add_scalar('loss_model_set', loss_set.item(), current_step)
                        writer.add_scalar('loss_human', loss_human.item(), current_step)
        
        # Validation
        with torch.no_grad():
            test_loss = 0
            model.eval()
            pbar = enumerate(val_dataloder)
            if fabric.global_rank == 0 :
                test_labels, preds_list = [],[]           
                pbar = tqdm(pbar, total=len(val_dataloder))
                print(('\n' + '%11s' *(3)) % ('Epoch', 'GPU_mem', 'loss'))
            
            for i, batch in pbar:
                encoded_batch, label, write_model, write_model_set = batch
                encoded_batch = { k: v.cuda() for k, v in encoded_batch.items()}
                loss, out, k_out, k_outlabel = model(encoded_batch, write_model, write_model_set,label)        
                
                # (DeepSVDD Setting)Update hypersphere radius R on mini-batch distances
                if opt.objective == 'soft-boundary':
                    out = out - model.R ** 2
                else:
                    out = out
                    
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                if fabric.global_rank == 0 :
                    preds_list.append(out.cpu())
                    test_labels.append(k_outlabel.cpu())
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g') % 
                        (f'{epoch + 1}/{opt.total_epoch}', mem, loss.item())
                    )
            # use roc_auc_score and other metrics to evaluate the performance 
            if fabric.global_rank == 0 :
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
                target_tpr = 0.95                               # the TPR you care about
                fpr_at_tpr_95 = np.interp(target_tpr, tpr, fpr)

                threshold, f1 = best_threshold_by_f1(label_np, pred_np)
                y_pred = np.where(pred_np>threshold,1,0)
                acc = accuracy_score(label_np, y_pred)
                precision = precision_score(label_np, y_pred)
                recall = recall_score(label_np, y_pred)
                f1 = f1_score(label_np, y_pred)
                print(f"Val, AUC: {roc_auc}, pr_auc: {pr_auc}, tpr_at_fpr_5: {tpr_at_fpr_5}, fpr_at_tpr_95: {fpr_at_tpr_95}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")
                writer.add_scalar('val_auc', roc_auc, epoch)
                writer.add_scalar('val_acc', acc, epoch)
                writer.add_scalar('val_precision', precision, epoch)
                writer.add_scalar('val_recall', recall, epoch)
                writer.add_scalar('val_f1', f1, epoch)
                writer.add_scalar('val_threshold', threshold, epoch)
                writer.add_scalar('val_f1', f1, epoch)


        torch.cuda.empty_cache()
        fabric.barrier()
        # save model
        if fabric.global_rank == 0:
            if roc_auc>max_auc:
                max_auc=roc_auc
                torch.save(model.get_encoder().state_dict(), os.path.join(opt.savedir,'model_best.pth'))
                torch.save(model.state_dict(), os.path.join(opt.savedir,'model_classifier_best.pth'))
                print('Save model to {}'.format(os.path.join(opt.savedir,'model_best.pth'.format(epoch))), flush=True)
                # save the best test result
                test_results = {
                    'epoch': epoch,
                    'auc': auc,
                    'acc': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
                test_results_path = os.path.join(opt.savedir, f"test_results_{opt.dataset}_{opt.method}.json")
                with open(test_results_path, 'w') as f:
                    json.dump(test_results, f, indent=4)
                print(f"Best test results saved to {test_results_path}")
            
            if epoch%10==0:
                torch.save(model.get_encoder().state_dict(), os.path.join(opt.savedir,'model_{}.pth'.format(epoch)))
                torch.save(model.state_dict(), os.path.join(opt.savedir,'model_classifier_{}.pth'.format(epoch)))
                print('Save model to {}'.format(os.path.join(opt.savedir,'model_{}.pth'.format(epoch))), flush=True)
            
            torch.save(model.get_encoder().state_dict(), os.path.join(opt.savedir,'model_last.pth'))
            torch.save(model.state_dict(), os.path.join(opt.savedir,'model_classifier_last.pth'))
            print('Save model to {}'.format(os.path.join(opt.savedir,'model_last.pth'.format(epoch))), flush=True)        
        
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