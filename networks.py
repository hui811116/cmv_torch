import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
import numpy as np
import utils as uts
from torch.nn.functional import normalize

class Encoder(nn.Module):
    def __init__(self,input_dim,feat_dim):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,feat_dim),
        )
    def forward(self,x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self,input_dim,feat_dim):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,input_dim),
        )
    def forward(self,x):
        return self.decoder(x)

class MFLVC(nn.Module):
    def __init__(self,dims,feature_dim,high_feature_dim,class_num,device):
        super(MFLVC,self).__init__()
        nview = len(dims)
        encoder =[]
        for v in range(nview):
            encoder.append(Encoder(np.prod(dims[v]),feature_dim).to(device))
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for v in range(nview):
            decoder.append(Decoder(np.prod(dims[v]),feature_dim).to(device))
        self.decoder = nn.ModuleList(decoder)

        self.feature_model = nn.Sequential(
            nn.Linear(feature_dim,high_feature_dim)
        )
        self.label_model = nn.Sequential(
                nn.Linear(feature_dim,class_num),
                nn.Softmax(dim=1),
            )
    def forward(self,xs):
        zs = []
        qs =[]
        hs = []
        xrs = []
        nview = len(xs)
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            h = self.feature_model(z)
            q = self.label_model(z)
            xr = self.decoder[v](z)
            zs.append(z)
            qs.append(q)
            hs.append(h)
            xrs.append(xr)
        return zs, qs, hs, xrs,

class MFLVC_Q(nn.Module):
    def __init__(self,input_sizes,dims,feature_dim,class_num,device):
        super(MFLVC_Q,self).__init__()
        nview = len(input_sizes)
        encoder =[]
        for v in range(nview):
            #encoder.append(Encoder(np.prod(dims[v]),feature_dim).to(device))
            encoder.append(AutoEncoder(input_sizes[v],feature_dim,dims).to(device))
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for v in range(nview):
            #decoder.append(Decoder(np.prod(dims[v]),feature_dim).to(device))
            decoder.append(AutoDecoder(input_sizes[v],feature_dim,dims).to(device))
        self.decoder = nn.ModuleList(decoder)

        #self.feature_model = nn.Sequential(
        #    nn.Linear(feature_dim,high_feature_dim)
        #)
        self.label_model = nn.Sequential(
                nn.Linear(feature_dim,class_num),
                nn.Softmax(dim=1),
            )
    def forward(self,xs):
        zs = []
        qs =[]
        #hs = []
        xrs = []
        nview = len(xs)
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            #h = self.feature_model(z)
            q = self.label_model(z)
            xr = self.decoder[v](z)
            zs.append(z)
            qs.append(q)
            #hs.append(h)
            xrs.append(xr)
        return zs, qs, xrs,
"""
class CorrAE(nn.Module):
    def __init__(self,dims,feature_dim,class_num,device):
        super(CorrAE,self).__init__()
        # NOTE: trial for 2 views only
        nview = len(dims)
        encoder = []
        for v in range(nview):
            encoder.append(Encoder(np.prod(dims[v]),feature_dim).to(device))
        self.encoder = nn.ModuleList(encoder)

        mean_decoder = []
        for v in range(nview):
            mean_decoder.append(Decoder(np.prod(dims[v]),feature_dim).to(device))
        self.mean_decoder = nn.ModuleList(mean_decoder)

        corr_decoder = []
        for v in range(nview):
            for w in range(v+1,nview):
                corr_decoder.append(
                    nn.Sequential(
                        nn.Linear(feature_dim * feature_dim, 500),
                        nn.ReLU(),
                        nn.Linear(500,500),
                        nn.ReLU(),
                        nn.Linear(500,dims[v]*dims[w]),
                    ).to(device)
                )
        self.corr_decoder = nn.ModuleList(corr_decoder)
        '''
        self.corr_decoder = nn.Sequential(
            nn.Linear(feature_dim * feature_dim, 500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
        )
        '''

        self.label_module = nn.Sequential(
            nn.Linear(feature_dim*feature_dim,class_num),
            nn.Softmax(dim=1),
        )
    def forward(self,xs):
        zs = []
        qs = []
        xrs = []
        corrs = []
        nview = len(xs)
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            xr = self.mean_decoder[v](z)
            zs.append(z)
            xrs.append(xr)
            
        cidx = 0
        for v in range(nview):
            for w in range(v+1,nview):
                zcr = torch.bmm(zs[v].unsqueeze(2),zs[w].unsqueeze(1)).flatten(1)
                # Correlation matrix, 
                cr = self.corr_decoder[cidx](zcr).view(-1,xs[v].shape[1],xs[w].shape[1])
                cr = 0.7 * cr.tanh()
                cidx += 1
                q = self.label_module(zcr)
                corrs.append(cr)
                qs.append(q)
                
        return zs, qs, xrs, corrs,
"""

class CMV(nn.Module):
    def __init__(self,dims,feature_dim,class_num,device):
        super(CMV,self).__init__()
        nview = len(dims)
        encoder = []
        for v in range(nview):
            encoder.append(Encoder(np.prod(dims[v]),feature_dim).to(device))
        self.encoder = nn.ModuleList(encoder)

        direct_decoder = []
        for v in range(nview):
            direct_decoder.append(Decoder(np.prod(dims[v]),feature_dim*feature_dim).to(device))
        self.direct_decoder = nn.ModuleList(direct_decoder)
        self.label_module = nn.Sequential(
            nn.Linear(feature_dim*feature_dim,class_num),
            nn.Softmax(dim=1),
        )


    def forward(self,xs):
        zs = []
        qs = []
        dxrs = []
        nview = len(xs)
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            zs.append(z)
        zps = []
        for v in range(nview):
            for w in range(v+1,nview):
                ztmp = torch.bmm(zs[v].unsqueeze(2),zs[w].unsqueeze(1).detach()).flatten(1)
                q = self.label_module(ztmp)
                xr_v = self.direct_decoder[v](ztmp)
                xr_w = self.direct_decoder[w](ztmp)
                zps.append(ztmp)
                qs.append(q)
                dxrs.append([xr_v,xr_w])
        return zs, qs, dxrs, zps


class RecWyner(nn.Module):
    def __init__(self,dims,feature_dim,class_num,device):
        super(RecWyner,self).__init__()
        self.nview = len(dims)
        encoder = []
        for v in range(self.nview):
            encoder.append(Encoder(dims[v],feature_dim).to(device))
        self.encoder= nn.ModuleList(encoder)
        decoder =[]
        for v in range(self.nview):
            decoder.append(nn.ModuleList()) # each list for each class
            for c in range(class_num):
                decoder[v].append(Decoder(dims[v],feature_dim).to(device))
        self.decoder = nn.ModuleList(decoder)

        self.class_num = class_num
        self.feature_dim = feature_dim
    def forward(self,xs):
        nview = len(xs)
        zs = []
        xrs = []
        qs = []
        for v in range(nview):
            x = xs[v] # (batch, class * feature_dim)
            z = self.encoder[v](x)
            v_xr = [self.decoder[v][c](z) for c in range(self.class_num)] #[batch, xdim] * class_num
            v_diff_sq = torch.stack([(x-vx).square().mean(1) for vx in v_xr],-1) # (batch, class_num)
            q = F.softmax(-v_diff_sq,dim=1)
            zs.append(z)
            xrs.append(v_xr) # list of reconstruction
            qs.append(q) # error from the reconstructions

        return zs, qs, xrs
    
class IWyner(nn.Module):
    def __init__(self,dims,feature_dim,class_num,device):
        super(IWyner,self).__init__()
        encoder = []
        nview = len(dims)
        for v in range(nview):
            encoder.append(Encoder(dims[v],feature_dim).to(device))
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for v in range(nview):
            decoder.append(Decoder(dims[v],feature_dim).to(device))
        self.decoder = nn.ModuleList(decoder)

        self.label_module = nn.Sequential(
            nn.Linear(feature_dim,class_num),
            nn.Softmax(dim=1),
        )
        self.class_num = class_num
        self.feature_dim = feature_dim
    def forward(self,xs):
        zs = []
        xrs = []
        qs = []
        nview = len(xs)
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x) # (batch,class* feature)
            q = self.label_module(z)
            qs.append(q)
            zs.append(z)
            xrs.append([])
            for w in range(nview):
                # from x_v to x_w
                if v != w:
                    xrw = self.decoder[v](z)
                    xrs[v].append(xrw)
                else:
                    xrs[v].append(torch.zeros((1,),device=xs[0].device))
        return zs, qs, xrs
    
# dot product method?
# outer product method?
# NOTE: this implementation only applies to modality above three
class IVCWyner(nn.Module):
    def __init__(self,dims,feature_dim,class_num,device):
        super(IVCWyner,self).__init__()
        ##
        nview = len(dims)
        encoder =[]
        for v in range(nview):
            # for product method
            encoder.append(Encoder(dims[v],feature_dim).to(device))
        self.encoder = nn.ModuleList(encoder)

        self.label_module = nn.Sequential(
            nn.Linear(feature_dim,class_num),
            nn.Softmax(dim=1),
        )
        decoder = []
        for v in range(nview):
            decoder.append(Decoder(dims[v],feature_dim).to(device))
        self.decoder = nn.ModuleList(decoder)
        self.class_num = class_num
        self.feature_dim = feature_dim
    def forward(self,xs):
        zs = []
        qs =[]
        xps = []
        nview = len(xs)
        # FIXME: make the device flexible to modality
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            q = self.label_module(z)
            zs.append(z)
            qs.append(q)
        ## 
        qps = []
        
        for v in range(nview):
            cmpl_set = [k for k in range(nview) if k != v]
            zp = zs[cmpl_set[0]]
            for cidx in range(len(cmpl_set)-1):
                zp = zp.mul(zs[cmpl_set[cidx+1]])
            qpp = self.label_module(zp)
            xpp = self.decoder[v](zp)
            qps.append(qpp)
            xps.append(xpp)
        return zs,qs,qps,xps


class IncompleteWyner(nn.Module):
    def __init__(self,dims,feature_dim,class_num,device):
        super(IncompleteWyner,self).__init__()
        ##
        nview = len(dims)
        encoder =[]
        for v in range(nview):
            # for product method
            encoder.append(Encoder(dims[v],feature_dim).to(device))
        self.encoder = nn.ModuleList(encoder)

        ae_dec =[]
        for v in range(nview):
            ae_dec.append(Decoder(dims[v],feature_dim).to(device))
        self.ae_dec = nn.ModuleList(ae_dec)
        self.label_module = nn.Sequential(
            nn.Linear(feature_dim,class_num),
            nn.Softmax(dim=1),
        )
        """
        self.label_module = nn.Sequential(
            nn.Linear(feature_dim,500),
            nn.ReLU(),
            nn.Linear(500,class_num),
            nn.Softmax(dim=1),
        )
        """
        self.class_num = class_num
        self.feature_dim = feature_dim
    def forward(self,xs,mm):
        xds = []
        zs = []
        qs =[]
        nview = len(xs)
        # FIXME: make the device flexible to modality
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            xd = self.ae_dec[v](z)
            q = self.label_module(z)
            zs.append(z)
            xds.append(xd)
            qs.append(q)
        ## 
        qps = []
        zall = torch.stack(zs,dim=1) # (batch,nview,feature_dim)
        zall = zall.masked_fill_(~mm.unsqueeze(-1),1.0) # mm: (batch,nview) T/F (missing)
        for v in range(nview):
            ##qps.append([])
            """
            for w in range(v+1,nview):
                zp = zs[v].mul(zs[w])
                qpp = self.label_module(zp)
                qps.append(qpp)
            """
            cmpl_set = [k for k in range(nview) if k != v]
            zp = zall[:,cmpl_set[0],:]
            for cidx in range(len(cmpl_set)-1):
                zp = zp.mul(zall[:,cmpl_set[cidx+1],:])
            qpp = self.label_module(zp)
            qps.append(qpp)
        return zs,qs,xds,qps


### Gumbel wyner network
class GumbelWyner(nn.Module):
    def __init__(self,input_sizes,dims,feature_dim,class_num,temperature,device):
        super(GumbelWyner,self).__init__()
        ##
        nview = len(input_sizes)
        encoder =[]
        for v in range(nview):
            # for product method
            #encoder.append(Encoder(dims[v],feature_dim).to(device))
            encoder.append(AutoEncoder(input_sizes[v], feature_dim, dims).to(device))
            #encoder.append(AutoEncoder(input_sizes[v],class_num,dims).to(device))
        self.encoder = nn.ModuleList(encoder)

        ae_dec =[]
        decoder =[]
        for v in range(nview):
            #ae_dec.append(Decoder(dims[v],class_num).to(device))
            ae_dec.append(AutoDecoder(input_sizes[v], class_num, dims).to(device))
            decoder.append(AutoDecoder(input_sizes[v], feature_dim, dims).to(device))
        self.ae_dec = nn.ModuleList(ae_dec)
        self.decoder = nn.ModuleList(decoder)
        self.label_module = nn.Sequential(
            nn.Linear(feature_dim,class_num),
            nn.Softmax(dim=1),
        )
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.temperature = temperature
    def forward(self,xs,mv):
        xds = []
        xcs = []
        zs = []
        qps = []
        nview = len(xs)
        # FIXME: make the device flexible to modality
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            xc = self.decoder[v](z)
            q = self.label_module(z)
            #q = self.encoder[v](x)
            zs.append(z)
            qps.append(q)
            xcs.append(xc)
        ## 
        qall = torch.stack(qps,dim=1) #(batch,nbrance,class_num)
        if torch.is_tensor(mv):
            qmask = qall.masked_fill_(~mv.unsqueeze(-1),1.0/self.class_num) # changed to uniform
        else:
            qmask = qall
        # NOTE:
        # we need to get the equivalent category probability here... 
        # There will only be one case where all views are missing
        # we will replace all mixing of missing modality with its
        # reduced subsets...
        lq_eq = (qmask+1e-9).log().sum(1)
        # gumbel-softmax sampling
        eps = torch.rand_like(lq_eq) + 1e-9 # shape:(nb,nc)
        gs = -(-eps.log()).log()
        yobs = F.softmax((gs + lq_eq)/self.temperature,dim=1)
        for v in range(nview):
            xr = self.ae_dec[v](yobs)
            xds.append(xr)
        return zs,qps,xds,xcs,yobs
        #return qps,xds,yobs

class GumbelWynerZ(nn.Module):
    def __init__(self,input_sizes,dims,feature_dim,class_num,temperature,device):
        super(GumbelWynerZ,self).__init__()
        ##
        nview = len(input_sizes)
        encoder =[]
        for v in range(nview):
            # for product method
            #encoder.append(Encoder(dims[v],feature_dim).to(device))
            encoder.append(AutoEncoder(input_sizes[v], feature_dim, dims).to(device))
            #encoder.append(AutoEncoder(input_sizes[v],class_num,dims).to(device))
        self.encoder = nn.ModuleList(encoder)

        ae_dec =[]
        decoder =[]
        for v in range(nview):
            #ae_dec.append(Decoder(dims[v],class_num).to(device))
            ae_dec.append(AutoDecoder(feature_dim, class_num, dims).to(device))
            decoder.append(AutoDecoder(input_sizes[v], feature_dim, dims).to(device))
        self.ae_dec = nn.ModuleList(ae_dec)
        self.decoder = nn.ModuleList(decoder)
        self.label_module = nn.Sequential(
            nn.Linear(feature_dim,class_num),
            #nn.Softmax(dim=1),
        )
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.temperature = temperature
    def forward(self,xs,mv):
        zds = []
        xcs = []
        zs = []
        qps = []
        nview = len(xs)
        # FIXME: make the device flexible to modality
        for v in range(nview):
            x = xs[v]
            z = self.encoder[v](x)
            xc = self.decoder[v](z)
            lq = self.label_module(z)
            #q = self.encoder[v](x)
            zs.append(z)
            qps.append(F.softmax(lq+1e-9,dim=1))
            xcs.append(xc)
        ## 
        qall = torch.stack(qps,dim=1) #(batch,nbrance,class_num)
        if torch.is_tensor(mv):
            qmask = qall.masked_fill_(~mv.unsqueeze(-1),1.0/self.class_num) # changed to uniform
        else:
            qmask = qall
        # NOTE:
        # we need to get the equivalent category probability here... 
        # There will only be one case where all views are missing
        # we will replace all mixing of missing modality with its
        # reduced subsets...
        lq_eq = (qmask+1e-9).log().sum(1)
        # gumbel-softmax sampling
        eps = torch.rand_like(lq_eq) + 1e-9 # shape:(nb,nc)
        gs = -(-eps.log()).log()
        yobs = F.softmax((gs + lq_eq)/self.temperature,dim=1)
        for v in range(nview):
            zr = self.ae_dec[v](yobs)
            zds.append(zr)
        return zs,qps,zds,xcs,yobs

"""
# NOTE: from CVCL implementation
"""
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims)+1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i,  nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims)+1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i,  nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)