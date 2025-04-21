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
        # if opt.resum:
        #     state_dict = torch.load(opt.pth_path, map_location=self.device)
        #     self.model.load_state_dict(state_dict)
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
        # indices1 is model
        # label 1 is human, 0 is model
        # indices2 is model and human set 
        bsz = batch['input_ids'].size(0)
        q = self.model(batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index1 = self.fabric.all_gather(indices1).view(-1)
        k_index2 = self.fabric.all_gather(indices2).view(-1)
        #q:N
        #k:4N

        # --------- Contrastive Loss ---------
        if self.only_classifier:
            loss_label = torch.tensor(0,device=self.device)
        else:
            logits_label = self._compute_logits(q,indices1, indices2,label,k,k_index1,k_index2,k_label)
            gt = torch.zeros(bsz, dtype=torch.long,device=logits_label.device)
            loss_label = F.cross_entropy(logits_label, gt)
        
        # --------- Classification Loss (ID) ---------
        if self.training:
            machine_txt_idx = (label == 0).view(-1)
            q_machine = q[machine_txt_idx]
            classify_label = indices2[machine_txt_idx]
            out = self.classifier(q_machine)
            loss_classify = F.cross_entropy(out, classify_label)
        else:
            loss_classify = torch.tensor(0,device=self.device)


        # --------- Energy Loss (ID vs OOD) ---------
        logits_energy = self.classifier(q)
        energy = -torch.logsumexp(logits_energy, dim=1)
        
        m_in = -25.0
        m_out = -7.0
        is_id = 1 - label.view(-1) # 1 = ID, 0 = OOD

        loss_energy_in = F.relu(energy - m_in) ** 2
        loss_energy_out = F.relu(m_out - energy) ** 2
        loss_energy = torch.where(is_id == 1, loss_energy_in, loss_energy_out).mean()

        # --------- Final Loss ---------
        loss = self.a*loss_label+self.d*loss_classify + 0.01 * loss_energy
        if self.training:
            return loss,loss_classify,loss_energy,k, is_id
        else:
            energy = self.fabric.all_gather(energy).view(-1)
            neg_energy = -energy
            return loss,neg_energy,k, 1 - k_label
