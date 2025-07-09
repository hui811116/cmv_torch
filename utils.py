import numpy as np
import sys
import os
import torch
import random
import itertools
import math 
from sklearn.metrics import normalized_mutual_info_score, v_measure_score, adjusted_rand_score, accuracy_score
from sklearn import cluster
from sklearn.preprocessing import Normalizer
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

def latinSquareTrans(nview):
    if nview == 2:
        return np.array([]) # NOT defined for this case
    if nview == 3:
        return np.array([
            [0,2,1],
            [2,1,0],
            [1,0,2]
        ])
    if nview == 4:
        return np.array([
            [0,2,3,1],
            [3,1,0,2],
            [1,3,2,0],
            [2,0,1,3],
        ])
    if nview == 5:
        return np.array([
            [0,2,3,4,1],
            [3,1,4,0,2],
            [4,0,2,1,3],
            [2,4,1,3,0],
            [1,3,0,2,4],
        ])
    else:
        raise NotImplementedError("Unsupported number of views:{:}".format(nview))

def getDevice(force_cpu):
	try:
		if force_cpu:
			device= torch.device("cpu")
			print("force using CPU")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
			print("using Apple MX chipset")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
	except:
		print("MPS is not supported for this version of PyTorch")
		if torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def getSetDict(nview):
	set_dict = {"uniset":set(np.arange(nview)),"tuple_list":[]}
	for widx in range(0,int(np.floor(nview/2))):
		for item in itertools.combinations(set_dict['uniset'],widx+1):
			cmpl_tuple = tuple(set_dict['uniset']-set(item)) 
			if (item,cmpl_tuple) in set_dict['tuple_list'] or (cmpl_tuple,item) in set_dict['tuple_list']:
				pass
			else:
				set_dict['tuple_list'].append((item,cmpl_tuple)) # from tuple to set
	return set_dict

##### CVCL utils
def normalize_multiview_data(data_views, row_normalized=True):
    '''The rows or columns of a matrix normalized '''
    norm2 = Normalizer(norm='l2')
    num_views = len(data_views)
    for idx in range(num_views):
        if row_normalized:
            data_views[idx] = norm2.fit_transform(data_views[idx])
        else:
            data_views[idx] = norm2.fit_transform(data_views[idx].T).T

    return data_views


def spectral_clustering(W, num_clusters):
    """
    Apply spectral clustering on W.
    # Arguments
    :param W: an affinity matrix
    :param num_clusters: the number of clusters
    :return: cluster labels.
    """
    # spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed',
    #                                       assign_labels='discretize')

    assign_labels='kmeans'
    spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(W)
    labels = spectral.fit_predict(W)

    return labels


def cal_spectral_embedding(W, num_clusters):

    D = np.diag(1 / np.sqrt(np.sum(W, axis=1) + math.e))
    # D1 = np.diag(np.power((np.sum(W, axis=1) + math.e), -0.5))
    Z = np.dot(np.dot(D, W), D)
    U, _, _ = np.linalg.svd(Z)
    eigenvectors = U[:, 0 : num_clusters]

    return eigenvectors


def cal_spectral_embedding_1(W, num_clusters):
    D = np.diag(np.power((np.sum(W, axis=1) + math.e), -0.5))
    L = np.eye(len(W)) - np.dot(np.dot(D, W), D)
    eigvals, eigvecs = np.linalg.eig(L)
    x_val = []
    x_vec = np.zeros((len(eigvecs[:, 0]), len(eigvecs[0])))
    for i in range(len(eigvecs[:, 0])):
        for j in range(len(eigvecs[0])):
            x_vec[i][j] = eigvecs[i][j].real
    for i in range(len(eigvals)):
        x_val.append(eigvals[i].real)
    # 选择前n个最小的特征向量
    indices = np.argsort(x_val)[: num_clusters]
    eigenvectors = x_vec[:, indices[: num_clusters]]

    return eigenvectors


def cal_l2_distances(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sqrt(np.sum(np.square(data_view - data_view[i]), axis=1)).T
    return dists


def cal_l2_distances_1(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sqrt(np.sum(np.square(data_view[i]-data_view[j])))

    return dists


def cal_squared_l2_distances(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sum(np.square(data_view - data_view[i]), axis=1).T
    return dists


def cal_squared_l2_distances_1(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sum(np.square(data_view[i]-data_view[j]))

    return dists


def cal_similiarity_matrix(data_view, k):
    '''
    calculate similiarity matrix
    '''
    num_samples = data_view.shape[0]
    dist = cal_squared_l2_distances(data_view)

    W = np.zeros((num_samples,num_samples), dtype=float)

    idx_set = dist.argsort()[::1]
    for i in range(num_samples):
        idx_sub_set = idx_set[i, 1:(k+2)]
        di = dist[i, idx_sub_set]
        W[i, idx_sub_set] = (di[k]-di) / (di[k] - np.mean(di[0:(k-1)]) + math.e)

    W = (W + W.T) / 2

    return W

def missIdxMaps(num_views,min_views=1):
     #assert num_views > min_views
     cand = [item for item in range(min_views,num_views)] # possible views
     #print(cand)
     cmap = {}
     for vv in cand:
        for item in itertools.combinations(list(range(num_views)),vv):
             dec_code = sum([2**tt for tt in item])
             #print(item,dec_code)
             cmap[dec_code] = item 
     rev_map = {v:k for k,v in cmap.items()}
     return cmap, rev_map

def bipartiteMasks(num_views):
     m_map = {}
     for vv in range(1,num_views):
        for item in itertools.combinations(list(range(num_views)),vv):
             #print(item)
             tmp_mask = np.array([False]*num_views)
             tmp_mask[list(item)] = True
             #print(tmp_mask)
             dec_con = [int(item) * 2**idx for idx, item in enumerate(tmp_mask)]
             dec_con = sum(dec_con)
             #print(dec_con)
             m_map[dec_con] = tmp_mask
     ## NOTE: append the last case, all present
     #last_all = np.array([True]*num_views)
     #dec_last = [int(item) * 2 ** idx for idx, item in enumerate(last_all)]
     #dec_con = sum(dec_last)
     #m_map[dec_con] = last_all
     #rev_map ={v:k for k,v in m_map.items()} 
     return m_map
def translate_mask(v_mask):
     nview = len(v_mask)
     pidx = [idx for idx,it in enumerate(v_mask) if it == True]
     translate_mask = np.array([False]*(2**nview-1))
     for vv in range(1,min(len(pidx)+1,nview)):
          for item in itertools.combinations(pidx,vv):
               print(item)
               di = sum([2**kk for kk in list(item)])
               translate_mask[di] = True
     return translate_mask[1:]