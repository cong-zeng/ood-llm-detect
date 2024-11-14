import torch
import torch.nn as nn
import torch.nn.functional as F
from src.text_embedding import TextEmbeddingModel

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, in_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.dense1 = nn.Linear(in_dim, in_dim//4)
        self.dense2 = nn.Linear(in_dim//4, in_dim//16)
        self.out_proj = nn.Linear(in_dim//16, out_dim)

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.dense1.bias, std=1e-6)
        nn.init.normal_(self.dense2.bias, std=1e-6)
        nn.init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, features):
        x = features
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class SimCLR_Classifier_SCL(nn.Module):
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_SCL, self).__init__()
        
        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        self.device=self.model.model.device
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.esp=torch.tensor(1e-6,device=self.device)
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        
        self.a=torch.tensor(opt.a,device=self.device)
        self.d=torch.tensor(opt.d,device=self.device)
        self.only_classifier=opt.only_classifier


    def get_encoder(self):
        return self.model

    def _compute_logits(self, q,q_index1, q_index2,q_label,k,k_index1,k_index2,k_label):
        def cosine_similarity_matrix(q, k):

            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        
        logits=cosine_similarity_matrix(q,k)/self.temperature

        q_labels=q_label.view(-1, 1)# N,1
        k_labels=k_label.view(1, -1)# 1,N+K

        same_label=(q_labels==k_labels)# N,N+K

        #model:model set
        pos_logits_model = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_model=logits*torch.logical_not(same_label)
        logits_model=torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1) 

        return logits_model
    
    def forward(self, batch, indices1, indices2,label):
        bsz = batch['input_ids'].size(0)
        q = self.model(batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index1 = self.fabric.all_gather(indices1).view(-1)
        k_index2 = self.fabric.all_gather(indices2).view(-1)
        #q:N
        #k:4N
        logits_label = self._compute_logits(q,indices1, indices2,label,k,k_index1,k_index2,k_label)
        
        out = self.classifier(q)
        
        if self.opt.AA:
            loss_classfiy = F.cross_entropy(out, indices1)
        else:
            loss_classfiy = F.cross_entropy(out, label)

        gt = torch.zeros(bsz, dtype=torch.long,device=logits_label.device)

        if self.only_classifier:
            loss_label = torch.tensor(0,device=self.device)
        else:
            loss_label = F.cross_entropy(logits_label, gt)

        loss = self.a*loss_label+self.d*loss_classfiy
        if self.training:
            return loss,loss_label,loss_classfiy,k,k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            return loss,out,k,k_label


class SimCLR_Classifier_test(nn.Module):
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_test, self).__init__()
        
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        self.device=self.model.model.device
    
    def forward(self, batch):
        q = self.model(batch)
        out = self.classifier(q)
        return out

class SimCLR_Classifier(nn.Module):
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier, self).__init__()

        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric

        self.model = TextEmbeddingModel(opt.model_name)
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.model.device)
            self.model.load_state_dict(state_dict)
  
        self.device=self.model.model.device
        self.esp=torch.tensor(1e-6,device=self.device)
        self.a = torch.tensor(opt.a,device=self.device)
        self.b = torch.tensor(opt.b,device=self.device)
        self.c = torch.tensor(opt.c,device=self.device)
        self.d = torch.tensor(opt.d,device=self.device)

        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        self.only_classifier = opt.only_classifier


    def get_encoder(self):
        return self.model

    def _compute_logits(self, q,q_index1, q_index2,q_label,k,k_index1,k_index2,k_label):
        def cosine_similarity_matrix(q, k):

            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        
        logits=cosine_similarity_matrix(q,k)/self.temperature

        q_index1=q_index1.view(-1, 1)# N,1
        q_index2=q_index2.view(-1, 1)# N,1
        q_labels=q_label.view(-1, 1)# N,1

        k_index1=k_index1.view(1, -1)# 1,N+K
        k_index2=k_index2.view(1, -1)
        k_labels=k_label.view(1, -1)# 1,N+K

        same_model=(q_index1==k_index1)
        same_set=(q_index2==k_index2)# N,N+K
        same_label=(q_labels==k_labels)# N,N+K

        is_human=(q_label==1).view(-1)
        is_machine=(q_label==0).view(-1)

        pos_logits_human = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_human=logits*torch.logical_not(same_label)
        logits_human=torch.cat((pos_logits_human.unsqueeze(1), neg_logits_human), dim=1)
        logits_human=logits_human[is_human]

        #model:model set
        pos_logits_model = torch.sum(logits*same_model,dim=1)/torch.max(torch.sum(same_model,dim=1),self.esp)# N
        neg_logits_model=logits*torch.logical_not(same_model)# N,N+K
        logits_model=torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1)
        logits_model=logits_model[is_machine]
        #model set:label
        pos_logits_set = torch.sum(logits*torch.logical_xor(same_set,same_model),dim=1)/torch.max(torch.sum(torch.logical_xor(same_set,same_model),dim=1),self.esp)
        neg_logits_set=logits*torch.logical_not(same_set)
        logits_set=torch.cat((pos_logits_set.unsqueeze(1), neg_logits_set), dim=1)
        logits_set=logits_set[is_machine]      
        #label:label
        pos_logits_label = torch.sum(logits*torch.logical_xor(same_set,same_label),dim=1)/torch.max(torch.sum(torch.logical_xor(same_set,same_label),dim=1),self.esp)
        neg_logits_label=logits*torch.logical_not(same_label)
        logits_label=torch.cat((pos_logits_label.unsqueeze(1), neg_logits_label), dim=1)
        logits_label=logits_label[is_machine]        

        return logits_model,logits_set,logits_label,logits_human
    
    def forward(self, encoded_batch, indices1, indices2,label):
        # print(len(text))
        q = self.model(encoded_batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index1 = self.fabric.all_gather(indices1).view(-1)
        k_index2 = self.fabric.all_gather(indices2).view(-1)
        #q:N
        #k:4N
        logits_model,logits_set,logits_label,logits_human = self._compute_logits(q,indices1, indices2,label,k,k_index1,k_index2,k_label)
        out = self.classifier(q)
        
        if self.opt.AA:
            loss_classfiy = F.cross_entropy(out, indices1)
        else:
            loss_classfiy = F.cross_entropy(out, label)

        gt_model = torch.zeros(logits_model.size(0), dtype=torch.long,device=logits_model.device)
        gt_set = torch.zeros(logits_set.size(0), dtype=torch.long,device=logits_set.device)
        gt_label = torch.zeros(logits_label.size(0), dtype=torch.long,device=logits_label.device)
        gt_human = torch.zeros(logits_human.size(0), dtype=torch.long,device=logits_human.device)


        loss_model =  F.cross_entropy(logits_model, gt_model)
        loss_set = F.cross_entropy(logits_set, gt_set)
        loss_label = F.cross_entropy(logits_label, gt_label)
        if logits_human.numel()!=0:
            loss_human = F.cross_entropy(logits_human.to(torch.float64), gt_human)
        else:
            loss_human=torch.tensor(0,device=self.device)

        loss = self.a*loss_model + self.b*loss_set + self.c*loss_label+(self.a+self.b+self.c)*loss_human+self.d*loss_classfiy
        if self.training:
            if self.opt.AA:
                return loss,loss_model,loss_set,loss_label,loss_human,loss_classfiy,k,k_index1
            else:
                return loss,loss_model,loss_set,loss_label,loss_classfiy,loss_human,k,k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            if self.opt.AA:
                return loss,out,k,k_index1
            else:
                return loss,out,k,k_label
