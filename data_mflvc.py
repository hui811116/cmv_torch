from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

train_test_split = 0.9

class BDGP(Dataset):
    def __init__(self, path, train=False):
        self.is_train=train
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        ntrain = int(data1.shape[0]*train_test_split)
        if train:
            self.x1 = data1[:ntrain]
            self.x2 = data2[:ntrain]
            self.y = labels[:ntrain]
        else:
            self.x1 = data1[ntrain:]
            self.x2 = data2[ntrain:]
            self.y = labels[ntrain:]
        #print(np.unique(self.y))
        #print([np.sum(ic == self.y) for ic in range(len(np.unique(self.y)))])

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

def uniformTrainIndex(labels,train_test_split):
    unif_dict = {}
    #labels = np.squeeze(labels)
    for idx, item in enumerate(labels):
        if not item in unif_dict.keys():
            unif_dict[item] = []
        unif_dict[item].append(idx)
    train_idx = []
    #train_labels = []
    test_idx = []
    #test_labels =[]
    for k,v in unif_dict.items():
        ntrain = int(len(v)*train_test_split)
        #train_idx = train_idx + v[:ntrain]
        train_idx.extend(v[:ntrain])
        #train_labels.extend([k]*ntrain)
        #test_idx = test_idx + v[ntrain:]
        test_idx.extend(v[ntrain:])
        #test_labels.extend([k]*(len(v)-ntrain))
    return train_idx,test_idx

class CCV(Dataset):
    def __init__(self, path, train=False):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.squeeze(np.load(path+'label.npy'))
        unif_dict = {}
        list_label = self.labels.tolist()
        for idx,item in enumerate(list_label):
            if not item in unif_dict.keys():
                unif_dict[item] =[]
            unif_dict[item].append(idx)
        train_idx, test_idx = uniformTrainIndex(list_label,train_test_split)
        if train:
            self.data1 = self.data1[train_idx]
            self.data2 = self.data2[train_idx]
            self.data3 = self.data3[train_idx]
            self.labels = self.labels[train_idx]
        else:
            self.data1 = self.data1[test_idx]
            self.data2 = self.data2[test_idx]
            self.data3 = self.data3[test_idx]
            self.labels = self.labels[test_idx]
        #print(np.unique(self.labels))
        #print([np.sum(ic==self.labels) for ic in np.unique(self.labels)])
    def __len__(self):
        #return 6773
        return self.data1.shape[0]

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(np.asarray(self.labels[idx])), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path,train):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)
        ntrain = int(self.V1.shape[0]*train_test_split)
        if train:
            self.Y = self.Y[:ntrain]
            self.V1 = self.V1[:ntrain]
            self.V2 = self.V2[:ntrain]
        else:
            self.Y = self.Y[ntrain:]
            self.V1 = self.V1[ntrain:]
            self.V2 = self.V2[ntrain:]
        #print(np.unique(self.Y))
        #print([np.sum(self.Y==ic) for ic in np.unique(self.Y)])

    def __len__(self):
        return self.V1.shape[0]
        #return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path, train):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)
        ntrain = int(self.V1.shape[0] * train_test_split)
        if train:
            self.Y = self.Y[:ntrain]
            self.V1 = self.V1[:ntrain]
            self.V2 = self.V2[:ntrain]
            self.V3 = self.V3[:ntrain]
        else:
            self.Y = self.Y[ntrain:]
            self.V1 = self.V1[ntrain:]
            self.V2 = self.V2[ntrain:]
            self.V3 = self.V3[ntrain:]
        #print(np.unique(self.Y))
        #print([np.sum(ic == self.Y) for ic in np.unique(self.Y)])
    def __len__(self):
        #return 10000
        return self.V1.shape[0]

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view, train):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        #print(self.labels.shape)
        self.view = view
        ntrain = int(self.labels.shape[0] * train_test_split)
        if train:
            self.view1 = self.view1[:ntrain]
            self.view2 = self.view2[:ntrain]
            self.view3 = self.view3[:ntrain]
            self.view4 = self.view4[:ntrain]
            self.view5 = self.view5[:ntrain]
            self.labels = self.labels[:ntrain]
        else:
            self.view1 = self.view1[ntrain:]
            self.view2 = self.view2[ntrain:]
            self.view3 = self.view3[ntrain:]
            self.view4 = self.view4[ntrain:]
            self.view5 = self.view5[ntrain:]
            self.labels = self.labels[ntrain:]
        #print(np.unique(self.labels))
        #print([np.sum(ic == self.labels) for ic in np.unique(self.labels)])
    def __len__(self):
        #return 1400
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset,datapath,train):
    if dataset == "BDGP":
        #dataset = BDGP('./data/')
        dataset = BDGP(datapath,train)
        dims = [1750, 79]
        view = 2
        #data_size = 2500
        data_size = len(dataset)
        class_num = 5
    elif dataset == "MNIST-USPS":
        #dataset = MNIST_USPS('./data/')
        dataset = MNIST_USPS(datapath,train)
        dims = [784, 784]
        view = 2
        class_num = 10
        #data_size = 5000
        data_size = len(dataset)
    elif dataset == "CCV":
        #dataset = CCV('./data/')
        dataset = CCV(datapath,train)
        dims = [5000, 5000, 4000]
        view = 3
        #data_size = 6773
        data_size = len(dataset)
        class_num = 20
    elif dataset == "Fashion":
        #dataset = Fashion('./data/')
        dataset = Fashion(datapath,train)
        dims = [784, 784, 784]
        view = 3
        #data_size = 10000
        data_size = len(dataset)
        class_num = 10
    elif dataset == "Caltech-2V":
        #dataset = Caltech('data/Caltech-5V.mat', view=2)
        dataset = Caltech(datapath+'Caltech-5V.mat', view=2,train=train)
        dims = [40, 254]
        view = 2
        #data_size = 1400
        data_size = len(dataset)
        class_num = 7
    elif dataset == "Caltech-3V":
        #dataset = Caltech('data/Caltech-5V.mat', view=3)
        dataset = Caltech(datapath+'Caltech-5V.mat', view=3,train=train)
        dims = [40, 254, 928]
        view = 3
        #data_size = 1400
        data_size = len(dataset)
        class_num = 7
    elif dataset == "Caltech-4V":
        #dataset = Caltech('data/Caltech-5V.mat', view=4)
        dataset = Caltech(datapath+'Caltech-5V.mat', view=4,train=train)
        dims = [40, 254, 928, 512]
        view = 4
        #data_size = 1400
        data_size = len(dataset)
        class_num = 7
    elif dataset == "Caltech-5V":
        #dataset = Caltech('data/Caltech-5V.mat', view=5)
        dataset = Caltech(datapath+'Caltech-5V.mat', view=5,train=train)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        #data_size = 1400
        data_size = len(dataset)
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
