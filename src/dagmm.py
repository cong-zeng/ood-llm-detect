import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.text_embedding import TextEmbeddingModel


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
    
class DaGMM(nn.Module):
    """Residual Block."""
    def __init__(self, n_gmm = 2, dim=768, latent_dim=18):
        super(DaGMM, self).__init__()

        layers = []
        layers += [nn.Linear(dim, int(dim/2))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(dim/2), int(dim/4))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(dim/4),int(dim/8))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(dim/8),16)]

        self.encoder = nn.Sequential(*layers)


        layers = []
        layers += [nn.Linear(16,int(dim/8))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(dim/8),int(dim/4))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(dim/4),int(dim/2))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(dim/2),dim)]

        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim,int(dim/8))]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]        
        layers += [nn.Linear(int(dim/8),n_gmm)]
        layers += [nn.Softmax(dim=1)]


        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):

        enc = self.encoder(x)

        dec = self.decoder(enc)

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        # z = torch.cat([enc, rec_cosine.unsqueeze(-1)], dim=1)


        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        device = z.device
        gamma = gamma.to(device)
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data
        self.phi = self.phi.to(device)

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        self.mu = self.mu.to(device)
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-4
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + torch.eye(D, device=cov[i].device)*eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            try:
                cov_ = (torch.linalg.cholesky(cov_k * (2*np.pi)).diag().prod()).unsqueeze(0)
            except:
                cov_ = (torch.linalg.cholesky( (cov_k * (2*np.pi)) + torch.eye(cov_k.shape[0], device=cov_k.device) * 1e-3 ).diag().prod()).unsqueeze(0)
            det_cov.append(cov_)
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov).cuda()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag

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
            print(f"Resume from the ckpt: {opt.pth_path}")
            state_dict = torch.load(opt.pth_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.esp=torch.tensor(1e-6,device=self.device)
        # self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        
        self.a=torch.tensor(opt.a,device=self.device)
        self.d=torch.tensor(opt.d,device=self.device)
        self.only_classifier=opt.only_classifier

        self.dagmm = DaGMM(n_gmm=2, dim=opt.projection_size)


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
        
        # out = self.classifier(q)

        # if self.opt.AA:
        #     loss_classfiy = F.cross_entropy(out, indices1)
        # else:
        #     loss_classfiy = F.cross_entropy(out, label.float().unsqueeze(-1))

        gt = torch.zeros(bsz, dtype=torch.long,device=logits_label.device)

        if self.only_classifier:
            loss_label = torch.tensor(0,device=self.device)
        else:
            loss_label = F.cross_entropy(logits_label, gt)

        # loss = self.a*loss_label+self.d*loss_classfiy
        # loss = self.a*loss_label
        loss = 0.0

        if self.training:
            enc, dec, z, gamma = self.dagmm(q)
            total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(q, dec, z, gamma, self.opt.lambda_energy, self.opt.lambda_cov_diag)
            loss += total_loss
            return loss,loss_label,sample_energy,k,k_label
        else:
            enc, dec, z, gamma = self.dagmm(k)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            # out = self.fabric.all_gather(out).view(-1, out.size(1))
            return loss,sample_energy,k,k_label


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
            state_dict = torch.load(opt.pth_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
  
        self.device=self.model.model.device
        self.esp=torch.tensor(1e-6,device=self.device)
        self.a = torch.tensor(opt.a,device=self.device)
        self.b = torch.tensor(opt.b,device=self.device)
        self.c = torch.tensor(opt.c,device=self.device)
        self.d = torch.tensor(opt.d,device=self.device)

        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        self.only_classifier = opt.only_classifier
        self.dagmm = DaGMM(n_gmm=3, dim=opt.projection_size)


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
            loss_classfiy = F.cross_entropy(out, label.float().unsqueeze(-1))

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

        loss = self.a*loss_model + self.b*loss_set + self.c*loss_label+(self.a+self.b+self.c)*loss_human + self.d * loss_classfiy
        if self.training:
            if self.opt.AA:
                return loss,loss_model,loss_set,loss_label,loss_human,None,k,k_index1
            else:
                q = q[(label == 0).view(-1)]
                enc, dec, z, gamma = self.dagmm(q)
                total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(q, dec, z, gamma, self.opt.lambda_energy, self.opt.lambda_cov_diag)
                loss += total_loss
                return loss,loss_model,loss_set,loss_label,None,loss_human,k,k_label
        else:
            # out = self.fabric.all_gather(out).view(-1, out.size(1))
            if self.opt.AA:
                return loss,out,k,k_index1
            else:
                enc, dec, z, gamma = self.dagmm(k)
                sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
                return loss,sample_energy,k,k_label
