import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset
import scipy.io as sio

class GMM3v(Dataset):
    def __init__(self,path,train=False,label_map=None):
        self.is_train = train
        if train:
            x_data = sio.loadmat(os.path.join(path,"gmm3v_x.mat"))['x_train'].astype("float32")
            y_label = sio.loadmat(os.path.join(path,"gmm3v_label.mat"))['y_train'].astype("int")
        else:
            x_data = sio.loadmat(os.path.join(path,"gmm3v_x_test.mat"))['x_test'].astype("float32")
            y_label = sio.loadmat(os.path.join(path,"gmm3v_y_test.mat"))['y_test'].astype("int")
        self.x_data = torch.from_numpy(x_data)
        self.y_label = torch.from_numpy(np.squeeze(y_label))
        if not label_map:
            print("LOG:dataloader.py GMM3v--- no label map provided, building a new one")
            self.label_map = {yy:idx for idx,yy in enumerate(np.unique(y_label))}
        else:
            self.label_map = label_map
        self.y_mapped = torch.from_numpy(np.array([self.label_map[it] for it in np.squeeze(y_label)]))
        self.nview = x_data.shape[1]
    def __len__(self):
        return len(self.y_mapped)
    def __getitem__(self,idx):
        xv = self.x_data[idx]
        xs = []
        for v in range(self.nview):
            xs.append(xv[v].view(-1))
        y = self.y_mapped[idx]
        return xs,y,0

class GMM3vIncomplete(Dataset):
    def __init__(self,path,train=False,label_map=None,missing_rate=0.8,seed=None):
        self.is_train = train
        if train:
            x_data = sio.loadmat(os.path.join(path,"gmm3v_x.mat"))['x_train'].astype("float32")
            y_label = sio.loadmat(os.path.join(path,"gmm3v_label.mat"))['y_train'].astype("int")
        else:
            x_data = sio.loadmat(os.path.join(path,"gmm3v_x_test.mat"))['x_test'].astype("float32")
            y_label = sio.loadmat(os.path.join(path,"gmm3v_y_test.mat"))['y_test'].astype("int")
        self.x_data = torch.from_numpy(x_data)
        self.y_label = torch.from_numpy(np.squeeze(y_label))
        if not label_map:
            print("LOG:dataloader.py GMM3vIncomplete--- no label map provided, building a new one")
            self.label_map = {yy:idx for idx,yy in enumerate(np.unique(y_label))}
        else:
            self.label_map = label_map
        self.y_mapped = torch.from_numpy(np.array([self.label_map[it] for it in np.squeeze(y_label)]))
        self.nview = x_data.shape[1]
        # 
        rng = np.random.default_rng(seed=seed)
        miss_idx = rng.integers(self.nview,size=(len(self.y_mapped,)))
        # random permutation
        rperm = rng.permutation(np.arange(len(self.y_mapped)))
        # 
        nmiss = int(missing_rate * len(self.y_mapped))
        sel_mis = rperm[:nmiss]
        all_miss = np.array([-1]*len(self.y_mapped))
        for item in sel_mis:
            all_miss[item] = miss_idx[item]
        self.miss_state = all_miss
        
    def __len__(self):
        return len(self.y_mapped)
    def __getitem__(self,idx):
        xv = self.x_data[idx]
        xs = []
        for v in range(self.nview):
            xs.append(xv[v].view(-1))
        y = self.y_mapped[idx]
        miss_idx = self.miss_state[idx]
        mask = np.array([True]*self.nview)
        if miss_idx >=0:
            mask[miss_idx] = False
        return xs,y, mask


class Means3vIncomplete(Dataset):
    def __init__(self,path,train=False,label_map=None,missing_rate=0.8,seed=None):
        self.is_train = train
        if train:
            x_data = sio.loadmat(os.path.join(path,"means3v_x.mat"))['x_train'].astype("float32")
            y_label = sio.loadmat(os.path.join(path,"means3v_label.mat"))['y_train'].astype("int")
        else:
            x_data = sio.loadmat(os.path.join(path,"means3v_x_test.mat"))['x_test'].astype("float32")
            y_label = sio.loadmat(os.path.join(path,"means3v_y_test.mat"))['y_test'].astype("int")
        self.x_data = torch.from_numpy(x_data)
        self.y_label = torch.from_numpy(np.squeeze(y_label))
        if not label_map:
            print("LOG:dataloader.py Means3vIncomplete--- no label map provided, building a new one")
            self.label_map = {yy:idx for idx,yy in enumerate(np.unique(y_label))}
        else:
            self.label_map = label_map
        self.y_mapped = torch.from_numpy(np.array([self.label_map[it] for it in np.squeeze(y_label)]))
        self.nview = x_data.shape[1]
        # 
        rng = np.random.default_rng(seed=seed)
        miss_idx = rng.integers(self.nview,size=(len(self.y_mapped,)))
        # random permutation
        rperm = rng.permutation(np.arange(len(self.y_mapped)))
        # 
        nmiss = int(missing_rate * len(self.y_mapped))
        sel_mis = rperm[:nmiss]
        all_miss = np.array([-1]*len(self.y_mapped))
        for item in sel_mis:
            all_miss[item] = miss_idx[item]
        self.miss_state = all_miss
        
    def __len__(self):
        return len(self.y_mapped)
    def __getitem__(self,idx):
        xv = self.x_data[idx]
        xs = []
        for v in range(self.nview):
            xs.append(xv[v].view(-1))
        y = self.y_mapped[idx]
        miss_idx = self.miss_state[idx]
        mask = np.array([True]*self.nview)
        if miss_idx >=0:
            mask[miss_idx] = False
        return xs,y, mask
