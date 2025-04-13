import sys
sys.path.append('./')
import random
random.seed(42)
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer
from src.index import Indexer
from utils.utils import compute_metrics,calculate_metrics
import torch
from src.dataset import PassagesDataset
from torch.utils.data import DataLoader
from src.simclr import SimCLR_Classifier,SimCLR_Classifier_SCL
from lightning import Fabric
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import yaml
from utils.Turing_utils import load_Turing
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4
from utils.Deepfake_utils import load_deepfake
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data.dataloader import default_collate

def process_top_ids_and_scores(top_ids_and_scores, label_dict):
    preds=[]
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        zero_num,one_num=0,0
        for id in ids:
            if label_dict[int(id)]==0:
                zero_num+=1
            else:
                one_num+=1
        if zero_num>one_num:
            preds.append('0')
        else:
            preds.append('1')
    return preds

def process_top_ids_and_scores_AA(top_ids_and_scores, label_dict):
    preds=[]
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        num_dict={}
        max_num,max_id=0,0
        for id in ids:
            if label_dict[int(id)] not in num_dict:
                num_dict[label_dict[int(id)]]=1
            else:
                num_dict[label_dict[int(id)]]+=1
            if num_dict[label_dict[int(id)]]>max_num:
                max_num=num_dict[label_dict[int(id)]]
                max_id=label_dict[int(id)]
        preds.append(str(max_id))
    return preds

def collate_fn(batch):
    # 首先使用default_collate处理大部分情况
    text,label,write_model,write_model_set = default_collate(batch)
    encoded_batch = tokenizer.batch_encode_plus(
        text,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True,
        )
    return encoded_batch,label,write_model,write_model_set

def train(opt):
    torch.set_float32_matmul_precision("medium")
    if opt.device_num>1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num,strategy=ddp_strategy)#
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num)
    fabric.launch()

    if opt.dataset=='deepfake':
        dataset = load_deepfake(opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='deepfake')
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='deepfake')
    elif opt.dataset=='TuringBench':
        dataset = load_Turing(file_folder=opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='Turing')
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='Turing')
    elif opt.dataset=='OUTFOX':
        dataset = load_OUTFOX(opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name],mode='OUTFOX')
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='OUTFOX')
    elif opt.dataset=='M4':
        dataset = load_M4(opt.path)
        passages_dataset = PassagesDataset(dataset[opt.database_name]+dataset[opt.database_name.replace('train','dev')],mode='M4')
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name],mode='M4')

    if opt.AA:
        opt.classifier_dim=len(passages_dataset.model_name_set)

    passages_dataloder = DataLoader(passages_dataset, batch_size=opt.per_gpu_batch_size,\
                                     num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
    
    val_dataloder = DataLoader(val_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                            num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    
    if opt.only_classifier:
        opt.a=opt.b=opt.c=0
        opt.d=1
        opt.one_loss=True

    if opt.one_loss:
        model = SimCLR_Classifier_SCL(opt,fabric).train()
    else:
        model = SimCLR_Classifier(opt,fabric).train()
    
    # assert opt.freeze_layer<=12 and opt.freeze_layer>=0, "freeze_layer should be in [0,12]"

    # if opt.freeze_layer>0 or opt.freeze_embedding_layer:
    #     name_list=[]
    #     for i in range(opt.freeze_layer,12):
    #         for name, param in model.model.named_parameters():
    #             if name.startswith(f'encoder.layer.{i}'):
    #                 name_list.append(name)

    #     for name, param in  model.model.named_parameters():
    #         if name in name_list:
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    if opt.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            if 'emb' in name:
                param.requires_grad=False
                
    if opt.d==0:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad=False

    passages_dataloder,val_dataloder=fabric.setup_dataloaders(passages_dataloder,val_dataloder)
    
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

    
    num_batches_per_epoch = len(passages_dataloder)
    warmup_steps=opt.warmup_steps
    lr = opt.lr
    total_steps = opt.total_epoch * num_batches_per_epoch- warmup_steps
    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=lr/10)
    model, optimizer = fabric.setup(model, optimizer)
    max_avg_rec=0
    for epoch in range(opt.total_epoch):
        model.train()
        avg_loss=0
        pbar = enumerate(passages_dataloder)
        if fabric.global_rank == 0:
            # print("the model has {} parameters".format(sum(p.numel() for p in model.parameters()))
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name)
            if opt.one_loss:
                print("Train with one loss!")
            index.reset()
            allembeddings, alllabels= [],[]
            label_dict={}            
            pbar = tqdm(pbar, total=len(passages_dataloder))
            print(('\n' + '%11s' *(5)) % ('Epoch', 'GPU_mem', 'Cur_loss', 'avg_loss','lr'))
        for i,batch in pbar:
            optimizer.zero_grad()
            current_step=epoch*num_batches_per_epoch+i
            if current_step < warmup_steps:
                current_lr = lr * current_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            current_lr = optimizer.param_groups[0]['lr']

            encoded_batch,label,write_model,write_model_set = batch
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
        
            if opt.one_loss:
                loss,loss_label,loss_classfiy,k_out,k_outlabel  = model(encoded_batch,write_model,write_model_set,label)
            else:
                loss,loss_model,loss_set,loss_label,loss_classfiy,loss_human,k_out,k_outlabel  = model(encoded_batch,write_model,write_model_set,label)
            avg_loss=(avg_loss*i+loss.item())/(i+1)
            fabric.backward(loss)
            # fabric.clip_gradients(model, optimizer, max_norm=1.0, norm_type=2)
            optimizer.step()
            if current_step >= warmup_steps:
                schedule.step()

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            if fabric.global_rank == 0:
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * 3) %
                    (f'{epoch + 1}/{opt.total_epoch}', mem, loss.item(),avg_loss, current_lr)
                )
                allembeddings.append(k_out.cpu())
                alllabels.extend(k_outlabel.cpu().tolist())            
                if current_step%10==0:
                    writer.add_scalar('lr', current_lr, current_step)
                    writer.add_scalar('loss', loss.item(), current_step)
                    writer.add_scalar('avg_loss', avg_loss, current_step)
                    writer.add_scalar('loss_label', loss_label.item(), current_step)
                    writer.add_scalar('loss_classfiy', loss_classfiy.item(), current_step)
                    if opt.one_loss==False:
                        writer.add_scalar('loss_model', loss_model.item(), current_step)
                        writer.add_scalar('loss_model_set', loss_set.item(), current_step)
                        writer.add_scalar('loss_human', loss_human.item(), current_step)
        
        with torch.no_grad():
            test_loss=0
            model.eval()
            pbar=enumerate(val_dataloder)
            if fabric.global_rank == 0 :
                test_embeddings,test_labels = [],[]           
                pbar = tqdm(pbar, total=len(val_dataloder))
                print(('\n' + '%11s' *(5)) % ('Epoch', 'GPU_mem', 'Cur_acc', 'avg_acc','loss'))

            right_num, tot_num= 0,0
            for i, batch in pbar:
                encoded_batch,label,write_model,write_model_set = batch
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                loss,out,k_out,k_outlabel= model(encoded_batch,write_model,write_model_set,label)
                preds = torch.argmax(out, dim=1)
                # print(preds.shape,k_outlabel.shape)
                cur_right_num = (preds == k_outlabel).sum().item()
                cur_num = k_outlabel.shape[0]

                right_num+=cur_right_num
                tot_num+=cur_num

                test_loss=(test_loss*i+loss.item())/(i+1)

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                if fabric.global_rank == 0 :
                    test_embeddings.append(k_out.cpu())
                    test_labels.extend(k_outlabel.cpu().tolist())
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * 3) %
                        (f'{epoch + 1}/{opt.total_epoch}', mem, cur_right_num/cur_num, right_num/tot_num,loss.item())
                    )
                    
        torch.cuda.empty_cache()
        fabric.barrier()
        if opt.only_classifier==False and fabric.global_rank == 0:
            print("Add embeddings to index!")
            allembeddings = torch.cat(allembeddings, dim=0)
            epsilon = 1e-6
            norms  = torch.norm(allembeddings, dim=1, keepdim=True) + epsilon
            allembeddings= allembeddings / norms
            allids=range(len(alllabels))
            for i in range(len(allids)):
                label_dict[allids[i]]=alllabels[i]
            index.index_data(allids,allembeddings.numpy())
            print("Add embeddings to index done!")
        
            print("Search knn!")
            test_embeddings = torch.cat(test_embeddings, dim=0)
            test_labels=[str(test_labels[i]) for i in range(len(test_labels))]
            epsilon = 1e-6
            norms  = torch.norm(test_embeddings, dim=1, keepdim=True) + epsilon
            test_embeddings= test_embeddings / norms
            if len(test_embeddings.shape) == 1:
                test_embeddings = test_embeddings.reshape(1, -1)
            test_embeddings=test_embeddings.numpy()
            top_ids_and_scores = index.search_knn(test_embeddings, opt.topk)
            if opt.AA:
                preds=process_top_ids_and_scores_AA(top_ids_and_scores, label_dict)
            else:
                preds=process_top_ids_and_scores(top_ids_and_scores, label_dict)
            print("Search knn done!")
            if opt.AA:
                # print(test_labels[:10],preds[:10])
                accuracy, avg_f1,avg_rec=calculate_metrics(test_labels, preds)
                print(f"Validation Accuracy: {accuracy}, AvgF1: {avg_f1}, AvgRecall: {avg_rec}")
                writer.add_scalar('val/val_loss', test_loss, epoch)
                writer.add_scalar('val/val_acc', accuracy, epoch)
                writer.add_scalar('val/val_avg_f1', avg_f1, epoch)
                writer.add_scalar('val/val_avg_recall', avg_rec, epoch)
            else:
                human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(test_labels, preds)
                print(f"Validation HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")
                writer.add_scalar('val/val_loss', test_loss, epoch)
                writer.add_scalar('val/val_acc', acc, epoch)
                writer.add_scalar('val/val_precision', precision, epoch)
                writer.add_scalar('val/val_recall', recall, epoch)
                writer.add_scalar('val/val_f1', f1, epoch)
                writer.add_scalar('val/val_human_rec', human_rec, epoch)
                writer.add_scalar('val/val_machine_rec', machine_rec, epoch)
                writer.add_scalar('val/val_avg_rec', avg_rec, epoch)

        if fabric.global_rank == 0:
            writer.add_scalar('val/acc_classifier', right_num/tot_num, epoch)
            if opt.only_classifier:
                avg_rec=right_num/tot_num
            if avg_rec>max_avg_rec:
                max_avg_rec=avg_rec
                torch.save(model.get_encoder().state_dict(), os.path.join(opt.savedir,'model_best.pth'))
                torch.save(model.state_dict(), os.path.join(opt.savedir,'model_classifier_best.pth'))
                print('Save model to {}'.format(os.path.join(opt.savedir,'model_best.pth'.format(epoch))), flush=True)
            
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
    opt = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    train(opt)