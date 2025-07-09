import os, random, sys

import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset
# from torch.nn.functional import normalize
from utils import *


class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/"):
        self.data_views = list()

        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "BDGP":
            mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "hand":
            mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])+1).astype(np.int32)

        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        return sub_data_views, self.labels[index]


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


## modification to incomplete view settings
class IncompleteviewData(Dataset):
    def __init__(self, db, device, path="datasets/", missing_rate=0.2, seed=1234):
        self.data_views = list()

        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "BDGP":
            mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "hand":
            mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])+1).astype(np.int32)

        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)
        # FIXME: current version: only one view will be missing
        data_means = []
        for idx in range(self.num_views):
            dmean = self.data_views[idx].mean(0)
            data_means.append(dmean)
        self.data_means = data_means
        # FIXME: should avoid the case where only one view is there
        
        code_map, rev_code_map = missIdxMaps(self.num_views,1)
        # For sample indices
        rng = np.random.default_rng(seed=seed)
        #miss_idx = rng.integers(self.num_views,size=(len(self.labels),))
        miss_idx = rng.integers(len(code_map.keys()),size=(len(self.labels),))
        rperm = rng.permutation(np.arange(len(self.labels)))
        nmiss = int(missing_rate * len(self.labels))
        sel_mis = rperm[:nmiss]
        # For missing views
        all_masks = np.ones((len(self.labels),self.num_views)).astype("bool")
        case_code_map = {idx:list(code_map.keys())[idx] for idx in range(len(code_map.keys()))}
        for item in sel_mis:
            case_idx = miss_idx[item]
            miss_case = code_map[case_code_map[case_idx]]
            all_masks[item,:] = False
            all_masks[item,miss_case] = True # NOTE: miss case is for present views 
        #for ai in range(len(all_masks)):
        #    ait = all_masks[ai]
        #    #print(ait)
        #    b_mask = translate_mask(ait)
        #    #print(b_mask)
        #sys.exit()
        self.miss_state = all_masks
        #else:
        #    print("Warning: Only 2 Views available, No missing configuration")
        #    print("Missing Rate in this case will be set to 0 by default")
        #    if missing_rate != 0:
        #        raise RuntimeError("ERROR: for two views, only missing_rate=0 allowed")
        #    # two view case, no missing cases
        #    #self.miss_state = np.array([-1] * len(self.labels)) # all full view
        self.miss_state = np.ones((len(self.labels),self.num_views)).astype("bool")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        m_mask = self.miss_state[index] # True: present, False: missing view
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            if m_mask[view_idx]:
                sub_data_views.append(data_view[index])
            else:
                sub_data_views.append(self.data_means[view_idx])
        
        # FIXME: if the missing modality changes, revise below
        return sub_data_views, self.labels[index], m_mask