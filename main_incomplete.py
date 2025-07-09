import numpy as np
import os
import sys
import torch
#from dataloader import 
from cvcl_dataprocessing import IncompleteviewData
from networks import GumbelWyner, GumbelWynerZ
import utils as uts
import torch.nn as nn
from loss import IMV_Loss, MFLVC_Loss
from torch.nn import functional as F
import evaluate as ev
from data_mflvc import load_data
import argparse
import time
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--db",default="MSRCv1",choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP'],help="dataset names")
parser.add_argument("--datapath",default="./datasets/",type=str,help="path/to/dataset/folder")
parser.add_argument("--batch_size",type=int,default=128,help="batch size for training")
parser.add_argument("--missing_rate",type=float,default=0.1,help="missing rate in [0,1] scale")
parser.add_argument("--learning_rate",type=float,default=5e-4,help="Learning rate for training")
parser.add_argument("--seed",type=int,default=None,help="seed number for reproduction, default to None (time seeding)")
parser.add_argument("--save_dir",type=str,default="incomplete_model",help="model saving path")
parser.add_argument("--feature_dim",type=int,default=256,help="latent feature dimension for dimension reduction")
parser.add_argument("--mse_epochs",type=int,default=100,help="MSE dimension reduction training")
parser.add_argument("--con_epochs",type=int,default=100,help="contrastive training")
parser.add_argument("--cpu",action="store_true",default=False,help="force using cpu computing")

args = parser.parse_args()

print("Incomplete Wyner Solver")
print(vars(args))

if args.db == "MSRCv1":
    # db checked 97.62
    #args.learning_rate = 0.0005
    args.batch_size = 35
    #args.con_epochs = 400
    #args.seed = 10
    #args.normalized = False

    #dim_high_feature = 2000
    args.feature_dim = 2000
    #dim_low_feature = 1024
    dims = [256, 512]
    #lmd = 0.01
    #beta = 0.005

elif args.db == "MNIST-USPS":
    # db checked 99.7
    #args.learning_rate = 0.0001
    args.batch_size = 50
    #args.seed = 10
    #args.con_epochs = 200
    #args.normalized = False

    #dim_high_feature = 1500
    args.feature_dim = 1500
    #dim_low_feature = 1024
    dims = [256, 512, 1024]
    #lmd = 0.05
    #beta = 0.05

elif args.db == "COIL20":
    # db checked 84.65
    #args.learning_rate = 0.0005
    args.batch_size = 180
    #args.seed = 50
    #args.con_epochs = 400
    #args.normalized = False

    #dim_high_feature = 768
    args.feature_dim = 768
    #dim_low_feature = 200
    dims = [256, 512, 1024, 2048]
    #lmd = 0.01
    #beta = 0.01

elif args.db == "scene":
    # db checked 44.59
    #args.learning_rate = 0.0005
    #args.con_epochs = 100
    args.batch_size = 69
    #args.seed = 10
    #args.normalized = False

    #dim_high_feature = 1500
    args.feature_dim = 1500
    #dim_low_feature = 256
    dims = [256, 512, 1024, 2048]
    #lmd = 0.01
    #beta = 0.05

elif args.db == "hand":
    # db checked 96.85
    #args.learning_rate = 0.0001
    args.batch_size = 200
    #args.seed = 50
    #args.con_epochs = 200
    #args.normalized = True

    #dim_high_feature = 1024
    args.feature_dim = 1024
    #dim_low_feature = 1024
    dims = [256, 512, 1024]
    #lmd = 0.005
    #beta = 0.001

elif args.db == "Fashion":
    # db checked 99.31
    #args.learning_rate = 0.0005
    args.batch_size = 100
    #args.con_epochs = 100
    #args.seed = 20
    #args.normalized = True
    #args.temperature_l = 0.5

    #dim_high_feature = 2000
    args.feature_dim = 2000
    #dim_low_feature = 500
    dims = [256, 512]
    #lmd = 0.005
    #beta = 0.005

elif args.db == "BDGP":
    # db checked 99.2
    #args.learning_rate = 0.0001
    args.batch_size = 250
    #args.seed = 10
    #args.con_epochs = 100
    #args.normalized = True

    #dim_high_feature = 2000
    args.feature_dim = 2000
    #dim_low_feature = 1024
    dims = [256, 512]
    #lmd = 0.01
    #beta = 0.01

device = uts.getDevice(args.cpu)
 # NOTE: MISSING SEED is fixed different from system seed
train_dataset = IncompleteviewData(args.db,device,args.datapath,args.missing_rate,seed=1234)


train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        #num_workers=args.workers,
    )
"""
test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=d_batch_size,
        shuffle=False,
    )
"""
feat_dims =[]
for bi,data in enumerate(train_data_loader):
    xs, y,mm = data
    for xx in xs:
        feat_dims.append(np.prod(xx.shape[1:]))
    break
nview = len(feat_dims)
class_num = np.unique(train_dataset.labels).size

#
#
network = GumbelWyner(feat_dims,dims,args.feature_dim,class_num,0.5,device).to(device)
#network = GumbelWynerZ(feat_dims,dims,args.feature_dim,class_num,0.5,device).to(device)
loss_obj = IMV_Loss(class_num).to(device)
mfl_loss = MFLVC_Loss(args.batch_size,class_num,1.0,0.5,device).to(device)

uts.print_network(network)
optim = torch.optim.Adam(network.parameters(),lr=args.learning_rate)

def masked_mse(xr,xin,mask):
    num_has = mask.int().sum()
    assert num_has >0
    diff = (xr - xin).square().flatten(1)
    xdim = diff.shape[1]
    diff = diff.sum(1)
    diff = diff.masked_fill_(~mask,0)
    return diff.sum()/num_has/xdim

def train_mse(epoch):
    tot_loss = 0.
    #mes = torch.nn.MSELoss()
    #ce_l = torch.nn.CrossEntropyLoss()
    y_true =[]
    z_pred =[]
    for batch_idx, (xs, y, mm) in enumerate(train_data_loader):
        # NOTE: here if not present, cannot be counted for training
        for v in range(nview):
            xs[v] = xs[v].to(device)
        # Only for debugging (supervised settings)
        y = y.to(device)
        mm = mm.to(device)
        optim.zero_grad()
        zs, qs,zds,xcs,yobs = network(xs,mm)
        loss_list = []
        # using contrastive learning loss for label matching

        for v in range(nview):
            #loss_list.append(masked_mse(zds[v],zs[v].detach(),mm[:,v]))
            loss_list.append(masked_mse(xcs[v],xs[v],mm[:,v]))
        # total prediction
        qtol = yobs # softmax output
        # FIXME: Debugging only, supervised case
        #loss_list.append(ce_l(qtol.clip(min=1e-5).log(),y))
        loss = sum(loss_list)
        loss.backward()
        optim.step()
        tot_loss += loss.item()
        y_true.append(y.detach().cpu().numpy())
        z_pred.append(qtol.argmax(1).detach().cpu().numpy())
    y_true = np.concatenate(y_true,axis=0)
    z_pred = np.concatenate(z_pred,axis=0)
    acc,acc_cnt, tot_cnt = ev.clustering_accuracy(y_true,z_pred)
    nmi, ari, _, pur = ev.evaluate(y_true,z_pred)
    print('Epoch {:} (MSE) Loss:{:.6f}, Accuracy:{:.4f}({:}/{:})'.format(
        epoch,tot_loss/len(train_data_loader),acc,acc_cnt,tot_cnt))
    return {"acc":acc,"acc_cnt":acc_cnt,"total_cnt":tot_cnt,"loss":tot_loss/len(train_data_loader),
            "z_pred":z_pred,"nmi":nmi,"ari":ari,"pur":pur,}

def train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    #ce_l = torch.nn.CrossEntropyLoss()
    y_true =[]
    z_pred =[]
    for batch_idx, (xs, y, mm) in enumerate(train_data_loader):
        for v in range(nview):
            xs[v] = xs[v].to(device)
        # Only for debugging (supervised settings)
        y = y.to(device)
        mm = mm.to(device)
        optim.zero_grad()
        zs, qs,zds,xcs,yobs = network(xs,mm)
        loss_list = []
        # using contrastive learning loss for label matching

        for v in range(nview):
            #loss_list.append(masked_mse(zds[v],zs[v].detach(),mm[:,v]))
            loss_list.append(masked_mse(zds[v],xs[v],mm[:,v]))
            loss_list.append(masked_mse(xcs[v],xs[v],mm[:,v]))
            for w in range(v+1,nview):
                sub_mask = mm[:,[v,w]]
                """
                sub_mask = mm[:,[v,w]]
                # NOTE: call the loss, and send the mask along with it
                #con_fr_loss = loss_obj.forward_feature(qs[v],qs[w],sub_mask)
                con_lb_loss = loss_obj.forward_label(qs[v],qs[w],sub_mask)
                # FIXME: this part can be moved to the outer loop for less computation
                neg_ent_loss = loss_obj.forward_nentropy(qs[v],qs[w],sub_mask)
                #loss_list.append(con_fr_loss)
                loss_list.append(con_lb_loss)
                loss_list.append(neg_ent_loss)
                """
                #lab_loss, _ = mfl_loss.forward_label(qs[v],qs[w])
                #loss_list.append(lab_loss)
                loss_list.append(loss_obj.forward_label_ent(qs[v],qs[w],sub_mask))
        # total prediction
        qtol = yobs # softmax output
        # FIXME: Debugging only, supervised case
        #loss_list.append(ce_l(qtol.clip(min=1e-5).log(),y))
        loss = sum(loss_list)
        loss.backward()
        optim.step()
        if torch.isnan(loss):
            print("ERROR:")
            print(zs)
            print(qs)
            print(zds)
            print(xcs)
            print(yobs)
            print(loss_list)
            sys.exit()
        tot_loss += loss.item()
        y_true.append(y.detach().cpu().numpy())
        z_pred.append(qtol.argmax(1).detach().cpu().numpy())
    y_true = np.concatenate(y_true,axis=0)
    z_pred = np.concatenate(z_pred,axis=0)
    acc,acc_cnt, tot_cnt = ev.clustering_accuracy(y_true,z_pred)
    nmi,ari,_,pur = ev.evaluate(y_true,z_pred)
    print('Epoch {:} (CON) Loss:{:.6f}, Accuracy:{:.4f}({:}/{:})'.format(
        epoch,tot_loss/len(train_data_loader),acc,acc_cnt,tot_cnt))
    return {"acc":acc,"acc_cnt":acc_cnt,"total_cnt":tot_cnt,"loss":tot_loss/len(train_data_loader),
            "z_pred":z_pred,"nmi":nmi,"ari":ari,"pur":pur,}

tr_record = {"acc":[],"acc_cnt":[],"total_cnt":[],"loss":[],
             "nmi":[],"ari":[],"pur":[],"ep_type":[],'rt':[]}
dt_s = time.time()
for ep in range(args.mse_epochs):
    tr_dict = train_mse(ep)
    #if (ep+1)%d_ev_freq == 0:
    #    test()
    for k in tr_record.keys():
        if k in tr_dict.keys():
            tr_record[k].append(tr_dict[k])
        elif k == "ep_type":
            tr_record["ep_type"].append("mse")
    tr_record["rt"].append(time.time() - dt_s)
for ep in range(args.con_epochs):
    tr_dict = train(ep+args.mse_epochs)
    for k in tr_record.keys():
        if k in tr_dict.keys():
            tr_record[k].append(tr_dict[k])
        elif k == "ep_type":
            tr_record["ep_type"].append("con")
    tr_record["rt"].append(time.time() - dt_s)
dt_e = time.time() - dt_s
print("training complete! Time elasped:{:.6f} secs".format(dt_e))
state_dict = network.state_dict()
os.makedirs(args.save_dir,exist_ok=True)
mdl_name = "IMV_{:}_m{:.2f}_fd{:}_bs{:}_lr{:.3e}_em{:}_ec{:}".format(
    args.db,args.missing_rate,args.feature_dim,args.batch_size,
    args.learning_rate,args.mse_epochs,args.con_epochs,
)
if args.seed:
    mdl_name += "_sd{:}".format(args.seed)
torch.save(state_dict,os.path.join(args.save_dir,mdl_name+".pth"))
df = pd.DataFrame.from_dict(tr_record)
df.to_csv(os.path.join(args.save_dir,mdl_name+".csv"),index=False)
with open(os.path.join(args.save_dir,mdl_name+".pkl"),'wb') as fid:
    pickle.dump(vars(args),fid)
print("Saving models, logs, arguments to:{:}".format(args.save_dir))
print("File prefix:{:}".format(mdl_name))
