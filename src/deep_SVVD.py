import torch
import torch.nn as nn
import torch.nn.functional as F
from src.text_embedding import TextEmbeddingModel
from tqdm import tqdm
from lightning import Fabric

# 把classificationHead替换成一个DeepSVDD的网络结构
class DeepSVDD(nn.Module):
    '''Deep SVDD model for anomaly detection.
    '''
    def __init__(self, objective, out_dim, R, c, nu: float, device, base_net):
        super(DeepSVDD, self).__init__()
        self.device = device
        self.R = torch.tensor(R, device=self.device)
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu # nu (0, 1]
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        self.out_dim = out_dim

        # Define a simple feed-forward neural network as the representation network.
        self.net = base_net

    def forward(self, x):
        x = self.net(x)
        return x
    
    def compute_loss(self, outputs):
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        return loss
    
class SimCLR_Classifier_SCL(nn.Module):
    """
    SimCLR_Classifier_SCL model combining contrastive learning and DeepSVDD for anomaly detection and classification.
    """
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_SCL, self).__init__()
        
        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric

        # Initialize the text embedding model.
        self.model = TextEmbeddingModel(opt.model_name)
        

        # Load a pretrained model if resume option is set.
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.device=self.model.model.device
        # Additional hyperparameters.
        self.esp=torch.tensor(1e-6,device=self.device)
        self.a=torch.tensor(opt.a,device=self.device)
        self.d=torch.tensor(opt.d,device=self.device)
        self.only_classifier=opt.only_classifier

        # Initialize DeepSVDD module.
        self.DeepSVDD = DeepSVDD(objective=opt.objective, in_dim=opt.hidden_dim, out_dim=opt.out_dim, R=opt.R, c=None, nu=opt.nu)


    def get_encoder(self):
        return self.model

    def _compute_logits(self, q,q_index1, q_index2,q_label,k,k_index1,k_index2,k_label):
        def cosine_similarity_matrix(q, k):

            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        
        # logits是平滑后的cosine相似度，用于计算loss。温度参数用于控制对相似度分布的平滑程度，温度较低：相似度分布更加尖锐，温度较高：相似度分布更加平滑，增加负样本的影响
        logits=cosine_similarity_matrix(q,k)/self.temperature

        q_labels=q_label.view(-1, 1)# N,1 查询标签形状变为(N,1)
        k_labels=k_label.view(1, -1)# 1,N+K 候选键标签形状变为(1,N+K)

        same_label=(q_labels==k_labels)# N,N+K 布尔矩阵，标记查询和键是否属于同一类别。查询标签与候选键标签相同的位置为True，不同的位置为False

        #model:model set
        pos_logits_model = torch.sum( logits * same_label, dim=1) / torch.max(torch.sum(same_label,dim=1), self.esp) #只保留与查询属于同一类别的键的相似度。对每个查询累加其正样本的相似度。归一化，防止正样本数目为 0 时出现除零。
        neg_logits_model = logits * torch.logical_not(same_label) # torch.logical_not(same_label)：标记所有负样本的位置。
        logits_model = torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1) #将正样本 logits（形状为 (N, 1)）和负样本 logits（形状为 (N, N+K)）拼接成一个矩阵：形状为 (N, 1 + N+K)。

        return logits_model
    
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
        
        # DeepSVDD forward pass and loss computation.
        out = self.DeepSVDD(q)
        loss_DeepSVDD = self.DeepSVDD.compute_loss(out)  # Calculate DeepSVDD loss.
        
        # 计算对比学习的loss。 将 logits 传入交叉熵损失函数，通过交叉熵使正样本 logits 更大，负样本 logits 更小。
        gt = torch.zeros(bsz, dtype=torch.long,device=logits_label.device)
        if self.only_classifier:
            loss_label = torch.tensor(0,device=self.device)
        else:
            loss_label = F.cross_entropy(logits_label, gt)

        # Combine both losses with their respective weights.
        loss = self.a * loss_label+ self.d * loss_DeepSVDD # 混合两个loss
        if self.training:
            return loss,loss_label,loss_DeepSVDD ,k,k_label
        else:
            # Gather outputs across devices during evaluation.
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            return loss, out, k, k_label

class SimCLR_Classifier_test(nn.Module):
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_test, self).__init__()
        
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        self.DeepSVDD = DeepSVDD(objective=opt.objective, in_dim=opt.projection_size, out_dim=opt.out_dim, R=opt.R, c=None, nu=opt.nu)
        self.device=self.model.model.device
    
    def forward(self, batch):
        q = self.model(batch)
        out = self.DeepSVDD(q)
        return out

class SimCLR_Classifier(nn.Module):
    def __init__(self, opt, fabric):
        super(SimCLR_Classifier, self).__init__()

        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric

        self.model = TextEmbeddingModel(opt.model_name)

        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.model.device)
            self.model.load_state_dict(state_dict)
        
        self.model = self.fabric.setup_module(self.model)
        print("Model is on:", next(self.model.parameters()).device)
        self.device=self.model.device
       
        self.esp=self.fabric.to_device(torch.tensor(1e-6))
        self.a = self.fabric.to_device(torch.tensor(opt.a))
        self.b = self.fabric.to_device(torch.tensor(opt.b))
        self.c = self.fabric.to_device(torch.tensor(opt.c))
        self.d = self.fabric.to_device(torch.tensor(opt.d))
        self.only_classifier = opt.only_classifier

        self.DeepSVDD = DeepSVDD(
            objective=opt.objective, 
            out_dim=opt.out_dim, 
            R=opt.R, 
            c=None, 
            nu=opt.nu, 
            device=self.fabric.device,
            base_net=self.model)
        
        self.DeepSVDD = self.fabric.setup_module(self.DeepSVDD)

    def initialize_center_c(self,train_loader, eps=0.1):
        
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(self.opt.out_dim, device=self.fabric.device)
        # Compute the mean of the output of the encoder for all training samples.
        # move the model to the device before the loop
        
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
        c /= n_samples
        torch.distributed.all_reduce(c, torch.distributed.ReduceOp.SUM)
        # Normalize to the hypersphere surface.
        c = c / torch.norm(c)
        self.DeepSVDD.c = c
        print('Center c initialized:',c)

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
        # out = self.DeepSVDD(q)
        machine_txt_idx = (label == 0).view(-1)
        loss_DeepSVDD = self.DeepSVDD.compute_loss(q[machine_txt_idx])  # Calculate DeepSVDD loss.


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

        loss = self.a*loss_model + self.b*loss_set + self.c*loss_label+(self.a+self.b+self.c)*loss_human+self.d*loss_DeepSVDD
        if self.training:
            if self.opt.AA:
                return loss,loss_model,loss_set,loss_label,loss_human,loss_DeepSVDD,k,k_index1
            else:
                return loss,loss_model,loss_set,loss_label,loss_DeepSVDD,loss_human,k,k_label
        
        else:
            dist = torch.sum((k - self.c) ** 2, dim=1)
            # out = "placeholder"
            # print(out)
            if self.opt.AA:
                return loss,dist,k,k_index1
            else:
                return loss,dist,k,k_label
