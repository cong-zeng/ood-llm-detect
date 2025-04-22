import torch
import torch.nn as nn
import torch.nn.functional as F
from src.text_embedding import TextEmbeddingModel
from tqdm import tqdm
from lightning import Fabric

class ClassificationHead(nn.Module):

    def __init__(self, in_dim, out_dim=1):
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
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.out_proj(x)
        return x

class SimCLR_Classifier_SCL(nn.Module):
    def __init__(self, opt, num_model, fabric):
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
        self.classifiers = nn.ModuleList(
                ClassificationHead(opt.projection_size, opt.classifier_dim)  for _ in range(num_model)
            )
        
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
    
    def _calc_gradient_penalty(self, classifier, x, y):
        batch_size = x.shape[0]
        if x.dim() == 2:
            alpha = torch.rand(batch_size, 1)
        else:
            alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand(x.size())
        alpha = alpha.to(x.device)

        interp = alpha * x + (1 - alpha) * y
        interp = interp.to(x.device)
        interp = torch.autograd.Variable(interp, requires_grad=True)
    
        d_interp = classifier(interp)
        grad = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                   grad_outputs=torch.ones(d_interp.size(), device=x.device),
                                   create_graph=True, retain_graph=True,
                                   only_inputs=True)[0]
        gradient_penalty = ((grad.norm(2, dim=1) - 1) ** 12).mean()
        return gradient_penalty
    
    def forward(self, batch, model_idx, indices1, indices2, label, run_all=False):
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

        # Contrastive loss 
        if self.only_classifier:
            loss_label = torch.tensor(0,device=self.device)        
        else:
            logits_label = self._compute_logits(q,indices1, indices2,label,k,k_index1,k_index2,k_label)
            gt = torch.zeros(bsz, dtype=torch.long,device=logits_label.device)
            loss_label = F.cross_entropy(logits_label, gt)

        mainloss_p = 0.0
        loss_pen = 0.0

        # Classifier loss with HRN gradient penalty
        if self.training:
            machine_txt_idx = (label == 0).view(-1)
            q_machine = q[machine_txt_idx]
            out = self.classifiers[model_idx](q_machine).squeeze(-1)
            mainloss_p = torch.log(torch.sigmoid(out) + self.esp).mean()
            loss_pen = self._calc_gradient_penalty(self.classifiers[model_idx], q_machine, q_machine)
        loss_classify = -mainloss_p + 0.1 * loss_pen


        # Total loss
        loss = self.a*loss_label+self.d*loss_classify
        scores = 0.0
        if self.training:
            return loss,loss_label,loss_classify,k,k_label
        else:
            if not run_all:
                out = self.classifiers[model_idx](k).squeeze(-1)
                scores = torch.sigmoid(out)
            else:
                for classifier in self.classifiers:
                    score = classifier(k)
                    scores += score
                scores = scores / len(self.classifiers)

            return loss,scores,k, k_index2, k_label
