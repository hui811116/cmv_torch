import numpy as np
import torch
import torch.nn as nn
import sys
import math

class Loss(nn.Module):
    def __init__(self,class_num):
        super(Loss,self).__init__()
        #self.batch_size = batch_size
        self.class_num = class_num
        self.ce_loss = nn.CrossEntropyLoss()
        self.similarity = nn.CosineSimilarity(dim=2)
    
    def masking(self,N):
        mask = torch.block_diag(torch.ones((N//2,N//2)),torch.ones((N//2,N//2)))
        for i in range(N//2):
            mask[N//2+i,i] = 1
            mask[i,N//2+i] = 1
        mask = mask.bool()
        mask = mask.logical_not()
        #print(mask) # correct
        return mask
        
    def forward_contrast(self,q1,q2):
        assert len(q1) == len(q2)
        batch_size = q1.shape[0]
        N = 2 * batch_size
        qlong = torch.cat([q1,q2],dim=0) # (2b,zd)
        hmat = qlong @ qlong.T  # (2b,2b)
        pos_h1 = hmat.diagonal(batch_size) # (b)
        pos_h2 = hmat.diagonal(-batch_size) # (b)
        pos_long = torch.cat([pos_h1,pos_h2],dim=0).reshape((N,1)) # (2b,1)
        # get all negative with cross views
        mask = self.masking(N)
        neg_all = hmat[mask].reshape((N,-1)) # 2b*(b-1)

        locations = torch.zeros(N,device=q1.device).long()
        merged_pos_neg = torch.cat([pos_long,neg_all],dim=1) # (2b,b)
        log_loss = self.ce_loss(merged_pos_neg,locations)
        log_loss/= N
        return log_loss
    def mask_correlated_samples(self,N):
        mask = torch.ones((N,N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 +i] = 0
            mask[N//2+i , i] = 0
        mask = mask.bool()
        return mask
    def forward_labels(self, q_i, q_j):
        assert np.all(q_i.shape == q_j.shape)
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j
        
        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / 1.0 # tmperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.ce_loss(logits, labels)
        loss /= N
        
        return loss + entropy
        #return loss
        #return entropy
    

class IMV_Loss(nn.Module):
    def __init__(self,class_num):
        super(IMV_Loss,self).__init__()
        self.class_num = class_num
        self.ce_loss = nn.CrossEntropyLoss()
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    def masking(self,N):
        #mask = torch.block_diag(torch.ones((N//2,N//2)),torch.ones((N//2,N//2)))
        # NOTE: alternative, self positive samples
        mask = torch.eye(N)
        for i in range(N//2):
            mask[i,N//2+i] = 1
            mask[N//2+i,i] = 1
        # COMMENT: negate to focus on the cross view items
        return mask.bool().logical_not()
    
    def forward_nentropy(self,q1,q2,m12):
        # m12: batch, (TF_1,TF_2)
        """
        ns = m12.sum(0)
        ne1 = (q1 * (q1.log()+math.log(self.class_num))).sum(1) # (batch,)
        ne2 = (q2 * (q2.log()+math.log(self.class_num))).sum(1) # (batch,)
        ne_merge = torch.cat([ne1.unsqueeze(1),ne2.unsqueeze(1)],dim=1)
        ne_merge = ne_merge.masked_fill_(~m12,0)
        entropy = (ne_merge.sum(0)/ns).sum()
        return entropy
        """
        #print(q1.shape)
        #print(m12.shape)
        #p_i = q1.sum(0).view(-1)
        #p_i = q1.masked_fill_(m12[:,0][:,None],0).sum(0).view(-1)
        p_i = (q1 * m12[:,0][:,None]).sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        #p_j = q2.sum(0).view(-1)
        #p_j = q2.masked_fill_(m12[:,1][:,None],0).sum(0).view(-1)
        p_j = (q2 * m12[:,1][:,None]).sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j
        return entropy
    def forward_feature(self,q1,q2,m12):
        pos_pairs = m12.all(1) # (b)
        assert pos_pairs.any() # must have at least one pairs

        s_size = pos_pairs.int().sum()
        mm_long = torch.cat([pos_pairs,pos_pairs],dim=0).float() # (2b,)
        N = 2 * s_size
        mmat = (mm_long.unsqueeze(-1) @ mm_long.unsqueeze(0)).bool() # (2b,2b)
        qlong = torch.cat([q1,q2],dim=0) # (2b,zd)
        hmat = qlong @ qlong.T  # (2b,2b) # FIXME: no non-linearity?
        mm_hmat = hmat[mmat].view(N,N) # (2s,2s)

        pos_h1 = mm_hmat.diagonal(s_size) # (b)
        pos_h2 = mm_hmat.diagonal(-s_size) # (b)
        pos_long = torch.cat([pos_h1,pos_h2],dim=0).reshape((N,1)) # (2b,1)
        # get all negative with cross views
        mask = self.masking(N)
        neg_all = mm_hmat[mask].reshape((N,-1)) # 2b*(b-1)

        locations = torch.zeros(N,device=q1.device).long()
        merged_pos_neg = torch.cat([pos_long,neg_all],dim=1) # (2b,b)
        log_loss = self.ce_loss(merged_pos_neg,locations)
        log_loss/= N
        return log_loss
    def forward_label(self,q1,q2,m12):
        paired_samp = torch.all(m12,dim=1) # (batch,) T/F
        q1_p = q1[paired_samp] # (p,c)
        q2_p = q2[paired_samp] # (p,c)
        # NOTE: follow the original implementation

        q = torch.cat([q1_p.t(),q2_p.t()],dim=0) # (2c,p)
        # get the prediction aligned for each view over the batchsize
        # focus on the one in pairs
        C = self.class_num * 2
        sim = self.similarity(q.unsqueeze(1),q.unsqueeze(0)) / 1.0 # temperature in original code
        sim_i_j = torch.diag(sim,self.class_num)
        sim_j_i = torch.diag(sim,-self.class_num)
        pos_cluster = torch.cat((sim_i_j,sim_j_i),dim=0).reshape(C,1)
        mask = self.masking(C)
        neg_cluster = sim[mask].reshape(C,-1)

        location = torch.zeros(C,device=q1.device).long()
        logits = torch.cat((pos_cluster,neg_cluster),dim=1)
        loss = self.ce_loss(logits,location)
        loss /= C
        return loss
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask
    def forward_label_ent(self,q_i, q_j,m12):
        
        #p_i = q_i.sum(0).view(-1)
        p_i = (q_i * m12[:,0][:,None]).sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        #p_j = q_j.sum(0).view(-1)
        p_j = (q_j*m12[:,1][:,None]).sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        paired_samp = torch.all(m12,dim=1)
        q_i = q_i[paired_samp,:] #
        q_i = q_i.t()
        q_j = q_j[paired_samp,:] #
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        #sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / 0.5
        #print(sim)
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy


#### BASELINES
class MFLVC_Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(MFLVC_Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        #print(sim)
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy,  sim 
    """
    ## separate negative entropy and similarity
    def forward_nentropy(self,q_i,q_j,mask_i,mask_j):
        p_i = (q_i*mask_i[:,None]).sum(0).view(-1)
        p_i /= mask_i.sum()
        #p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = (q_j*mask_j[:,None]).sum(0).view(-1)
        #p_j /= p_j.sum()
        p_j /= mask_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j
        return entropy
    def forward_matching(self,q_i,q_j,mask_i,mask_j):
        # common in pair samples
        # mask shape: (batch,T/F)
        mask_joint = mask_i.logical_and(mask_j) # (batch,TT)
        q_i = q_i[mask_joint]
        q_j = q_j[mask_joint]
        #
        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    ## following CVCL implementation
    def target_distribution(self,q):
        weight = (q **2.0) / torch.sum(q,0)
        return (weight.t() / torch.sum(weight,1)).t()
    """

class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = (p_i * torch.log(p_i)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = (p_j * torch.log(p_j)).sum()

        entropy = ne_i + ne_j

        return entropy

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):

        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

