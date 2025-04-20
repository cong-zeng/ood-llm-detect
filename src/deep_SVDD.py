import torch
import torch.nn as nn
import torch.nn.functional as F
from src.text_embedding import TextEmbeddingModel
from tqdm import tqdm
from lightning import Fabric


class DeepSVDD(nn.Module):
    '''Deep SVDD model for anomaly detection.
    '''
    def __init__(self, objective, out_dim, R, c, nu: float, device):
        super(DeepSVDD, self).__init__()
        self.device = device
        self.R = torch.tensor(R, device=self.device)
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu # nu (0, 1]
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        self.out_dim = out_dim


    def forward(self, x, net):
        x = net(x)
        return x
    
    def compute_loss(self, outputs, machine_txt_idx, human_txt_idx):
        machine_outputs = outputs[machine_txt_idx]
        human_outputs = outputs[human_txt_idx]

        dist_machine = torch.sum((machine_outputs - self.c) ** 2, dim=1)
        dist_human = torch.sum((human_outputs - self.c) ** 2, dim=1)

        avg_dist_machine = dist_machine.mean()
        avg_dist_human = dist_human.mean()
        dist = avg_dist_machine - avg_dist_human


        if self.objective == 'soft-boundary':
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = dist
        return loss
    
class SimCLR_Classifier_SCL(nn.Module):
    """
    SimCLR_Classifier_SCL model combining contrastive learning and DeepSVDD 
    """
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_SCL, self).__init__()
        
        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric

        # Initialize the text embedding model.
        self.model = TextEmbeddingModel(opt.model_name)
        self.device=next(self.model.parameters()).device

        # Load a pretrained model if resume option is set.
        # if opt.resum:
        #     state_dict = torch.load(opt.pth_path, map_location=self.device)
        #     self.model.load_state_dict(state_dict)

        # self.model = self.fabric.setup_module(self.model)
        # print("Model is on:", next(self.model.parameters()).device)

        # Additional hyperparameters.
        self.esp=torch.tensor(1e-6,device=self.device)
        self.a=torch.tensor(opt.a,device=self.device)
        self.d=torch.tensor(opt.d,device=self.device)
        self.only_classifier=opt.only_classifier

        # Initialize DeepSVDD module.
        self.R = nn.Parameter(torch.tensor(opt.R))
        self.c = nn.Parameter(torch.zeros(self.opt.out_dim))
        self.nu = opt.nu # nu (0, 1]
        self.objective = opt.objective
        
        # No Need for this
        # self.DeepSVDD = DeepSVDD(
        #     objective=opt.objective, 
        #     out_dim=opt.out_dim, 
        #     R=opt.R, 
        #     c=None, 
        #     nu=opt.nu, 
        #     device=self.fabric.device,)
        
        # self.DeepSVDD = self.fabric.setup_module(self.DeepSVDD)
    
    def initialize_center_c(self,train_loader, eps=0.1):
        
        """Initialize hypersphere center c as the mean from an initial forward pass on the machine data."""
        n_samples = 0
        c = torch.zeros(self.opt.out_dim, device=self.fabric.device)
        # Compute the mean of the output of the encoder for all training samples.
        
        self.model = self.model.to(self.fabric.device)
        self.model.eval()
        print('Initializing center c, to device:{}',self.fabric.device)
        
        with torch.no_grad():
            for batch in tqdm(train_loader):
                encoded_batch,_,_,_ = batch
                encoded_batch = {k: v.to(self.fabric.device) for k, v in encoded_batch.items()}
                outputs = self.model(encoded_batch)
                c += outputs.sum(dim=0)
                n_samples += outputs.shape[0]
        if self.fabric.world_size > 1:
            c = self.fabric.all_reduce(c, reduce_op="sum")
            # torch.distributed.all_reduce(c, torch.distributed.ReduceOp.SUM)
        c /= n_samples
        # Normalize to the hypersphere surface.
        c = c / torch.norm(c)
        self.c.data = c
        # print('Center c initialized:',c)


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
        pos_logits_model = torch.sum( logits * same_label, dim=1) / torch.max(torch.sum(same_label,dim=1), self.esp) 
        neg_logits_model = logits * torch.logical_not(same_label) 
        logits_model = torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1) 

        return logits_model
    
    def compute_loss(self, outputs, machine_txt_idx, human_txt_idx):
        machine_outputs = outputs[machine_txt_idx]
        human_outputs = outputs[human_txt_idx]

        dist_machine = torch.sum((machine_outputs - self.c) ** 2, dim=1)
        dist_human = torch.sum((human_outputs - self.c) ** 2, dim=1)

        avg_dist_machine = dist_machine.mean()
        avg_dist_human = dist_human.mean()
        dist = avg_dist_machine - avg_dist_human


        if self.objective == 'soft-boundary':
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = dist
        return loss

    def forward(self, batch, indices1, indices2,label):
        """
        Forward pass of the model.

        Args:
            batch: Input data batch for TextEmbeddingModel.
            indices1, indices2: Auxiliary indices for contrastive learning.
            label: Ground truth labels for the input batch.

        Returns:
            loss: Combined loss (contrastive + DeepSVDD).
            Additional outputs based on the mode (training or evaluation).
        """
                
        bsz = batch['input_ids'].size(0)
        q = self.model(batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index1 = self.fabric.all_gather(indices1).view(-1)
        k_index2 = self.fabric.all_gather(indices2).view(-1)
        #q:N
        #k:4N
        # Compute contrastive logits.
        logits_label = self._compute_logits(q,indices1, indices2,label,k,k_index1,k_index2,k_label)
        
        # Calculate DeepSVDD loss only positive sample.
        # loss_DeepSVDD = self.DeepSVDD.compute_loss(q)

        # Calculate DeepSVDD loss all sample.
        machine_txt_idx = (label == 0).view(-1)
        human_txt_idx = (label == 1).view(-1)
        loss_DeepSVDD = self.compute_loss(q, machine_txt_idx, human_txt_idx)  
        
        # Compute contrastive loss.
        gt = torch.zeros(bsz, dtype=torch.long,device=logits_label.device)
        if self.only_classifier:
            loss_label = torch.tensor(0,device=self.device)
        else:
            loss_label = F.cross_entropy(logits_label, gt)

        # Combine both losses with their respective weights.
        loss = self.a * loss_label+ self.d * loss_DeepSVDD 
        if self.training:
            return loss,loss_label,loss_DeepSVDD ,k,k_label
        else:
            # Gather outputs across devices during evaluation.
            dist = torch.sum((k - self.c) ** 2, dim=1)
            return loss,dist,k,k_label
        