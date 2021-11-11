import os
import pickle

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class MinMaxScaler_torch:

    def __init__(self, min=None, max=None, device='cuda:0'):
        self.min = min
        self.max = max
        self.device = device

    def fit(self, data):
        self.min = torch.min(data)
        self.max = torch.max(data)

    def transform(self, data):
        _data = data.clone()
        return (_data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return (data * (self.max - self.min + 1e-8)) + self.min


class StandardScaler_torch:

    def __init__(self):
        self.means = 0
        self.stds = 0

    def fit(self, data):
        self.means = torch.mean(data, dim=0)
        self.stds = torch.std(data, dim=0)

    def transform(self, data):
        _data = data.clone()
        data_size = data.size()

        if len(data_size) > 2:
            _data = _data.reshape(-1, data_size[-1])

        _data = (_data - self.means) / (self.stds + 1e-8)

        if len(data_size) > 2:
            _data = _data.reshape(data.size())

        return _data

    def inverse_transform(self, data):
        data_size = data.size()
        if len(data_size) > 2:
            data = data.reshape(-1, data_size[-1])

        data = (data * (self.stds + 1e-8)) + self.means

        if len(data_size) > 2:
            data = data.reshape(data_size)

        return data


def granularity(data, k):
    if k == 1:
        return np.copy(data)
    else:
        newdata = [np.mean(data[i:i + k], axis=0) for i in range(0, data.shape[0], k)]
        newdata = np.asarray(newdata)
        print('new data: ', newdata.shape)
        return newdata


class TrafficDataset(Dataset):

    def __init__(self, dataset, args):
        # save parameters
        self.args = args

        self.type = args.type
        self.out_seq_len = args.out_seq_len
        self.X = dataset['X']
        self.Y = dataset['Y']
        self.Xgt = dataset['Xgt']
        self.Ygt = dataset['Ygt']
        self.scaler = dataset['Scaler']

        self.nsample, self.len_x, self.nflows, self.nfeatures = self.X.shape

        # get valid start indices for sub-series
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.np2torch(self.X[t])
        y = self.np2torch(self.Y[t])
        xgt = self.np2torch(self.Xgt[t])
        ygt = self.np2torch(self.Ygt[t])
        sample = {'x': x, 'y': y, 'x_gt': xgt, 'y_gt': ygt}
        return sample

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        indices = np.arange(self.nsample)
        return indices


def load_raw(args):
    # load ground truth
    path = args.datapath

    data_path = os.path.join(path, 'data/{}.mat'.format(args.dataset))
    X = loadmat(data_path)['X']
    if len(X.shape) > 2:
        X = np.reshape(X, newshape=(X.shape[0], -1))

    return X


def np2torch(X, device):
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        X = X.to(device)
    return X


def data_preprocessing(data, args, gen_times=5, scaler=None, is_train=False):
    n_timesteps, n_series = data.shape

    # original dataset with granularity k = 1
    oX = np.copy(data)
    oX = np2torch(oX, args.device)

    # Obtain data with different granularity k
    X = granularity(data, args.k)
    X = np2torch(X, args.device)

    # scaling data
    if scaler is None:
        scaler = StandardScaler_torch()
        scaler.fit(X)
    else:
        scaler = scaler

    X_scaled = scaler.transform(X)

    len_x = args.seq_len_x
    len_y = args.seq_len_y

    dataset = {'X': [], 'Y': [], 'Xgt': [], 'Ygt': [], 'Scaler': scaler}

    skip = 4
    start_idx = 0
    for _ in range(gen_times):
        for t in range(start_idx, n_timesteps - len_x - len_y, len_x):
            x = X_scaled[t:t + len_x]
            x = x.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]

            y = torch.max(X[t + len_x:t + len_x + len_y], dim=0)[0]
            y = y.reshape(1, -1)

            # Data for doing traffic engineering
            x_gt = oX[t * args.k:(t + len_x) * args.k]
            y_gt = oX[(t + len_x) * args.k: (t + len_x + len_y) * args.k]
            if (torch.max(x_gt) <= 1.0 or torch.max(y_gt) <= 1.0) and is_train:
                continue

            dataset['X'].append(x)  # [sample, len_x, k, 1]
            dataset['Y'].append(y)  # [sample, 1, k]
            dataset['Xgt'].append(x_gt)
            dataset['Ygt'].append(y_gt)

        start_idx = start_idx + skip

    dataset['X'] = torch.stack(dataset['X'], dim=0)
    dataset['Y'] = torch.stack(dataset['Y'], dim=0)
    dataset['Xgt'] = torch.stack(dataset['Xgt'], dim=0)
    dataset['Ygt'] = torch.stack(dataset['Ygt'], dim=0)

    dataset['X'] = dataset['X'].cpu().data.numpy()
    dataset['Y'] = dataset['Y'].cpu().data.numpy()
    dataset['Xgt'] = dataset['Xgt'].cpu().data.numpy()
    dataset['Ygt'] = dataset['Ygt'].cpu().data.numpy()

    print('   X: ', dataset['X'].shape)
    print('   Y: ', dataset['Y'].shape)
    print('   Xgt: ', dataset['Xgt'].shape)
    print('   Ygt: ', dataset['Ygt'].shape)

    return dataset


def remove_outliers(data):
    q25, q75 = np.percentile(data, 25, axis=0), np.percentile(data, 75, axis=0)
    iqr = q75 - q25
    cut_off = iqr * 3
    lower, upper = q25 - cut_off, q75 + cut_off
    for i in range(data.shape[1]):
        flow = data[:, i]
        flow[flow > upper[i]] = upper[i]
        # flow[flow < lower[i]] = lower[i]
        data[:, i] = flow

    return data


def train_test_split(X, dataset):
    if 'abilene' in dataset:
        train_size = 4 * 7 * 288  # 4 weeks
        val_size = 288 * 7  # 1 week
        test_size = 288 * 7 * 2  # 2 weeks

    elif 'geant' in dataset:
        train_size = 96 * 7 * 4 * 2  # 2 months
        val_size = 96 * 7 * 2  # 2 weeks
        test_size = 96 * 7 * 4  # 1 month

    elif 'brain' in dataset:
        train_size = 1440 * 3  # 3 days
        val_size = 1440  # 1 day
        test_size = 1440 * 2  # 2 days
    elif 'uninett' in dataset:  # granularity: 1 hour
        train_size = 4 * 7 * 288  # 4 weeks
        val_size = 288 * 7  # 1 week
        test_size = 288 * 7 * 2  # 2 weeks
    elif 'renater_tm' in dataset:  # granularity: 5 min
        train_size = 4 * 7 * 288  # 4 weeks
        val_size = 288 * 7  # 1 week
        test_size = 288 * 7 * 2  # 2 weeks
    else:
        raise NotImplementedError

    X_train = X[:train_size]

    X_val = X[train_size:val_size + train_size]

    X_test = X[val_size + train_size: val_size + train_size + test_size]

    if 'abilene' in dataset or 'geant' in dataset or 'brain' in dataset:
        X_train = remove_outliers(X_train)
        X_val = remove_outliers(X_val)

    return X_train, X_val, X_test


def get_dataloader(args):
    # loading data
    X = load_raw(args)
    total_timesteps, total_series = X.shape
    stored_path = os.path.join(args.datapath, 'data/gwn_{}_{}_{}/'.format(args.dataset, args.seq_len_x,
                                                                          args.seq_len_y))
    if not os.path.exists(stored_path):
        os.makedirs(stored_path)

    saved_train_path = os.path.join(stored_path, 'train.pkl')
    saved_val_path = os.path.join(stored_path, 'val.pkl')
    saved_test_path = os.path.join(stored_path, 'test.pkl')

    if not os.path.exists(saved_train_path) \
            or not os.path.exists(saved_val_path) or not os.path.exists(saved_test_path):
        train, val, test = train_test_split(X, args.dataset)

        trainset = data_preprocessing(data=train, args=args, gen_times=10, scaler=None, is_train=True)
        train_scaler = trainset['Scaler']
        with open(saved_train_path, 'wb') as fp:
            pickle.dump(trainset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        print('Data preprocessing: VALSET')
        valset = data_preprocessing(data=val, args=args, gen_times=10, scaler=train_scaler, is_train=True)
        with open(saved_val_path, 'wb') as fp:
            pickle.dump(valset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        print('Data preprocessing: TESTSET')
        testset = data_preprocessing(data=test, args=args, gen_times=1, scaler=train_scaler, is_train=False)

        with open(saved_test_path, 'wb') as fp:
            pickle.dump(testset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
    else:
        print('Load saved dataset from {}'.format(stored_path))
        if args.test:
            trainset, valset = None, None
        else:
            with open(saved_train_path, 'rb') as fp:
                trainset = pickle.load(fp)
                fp.close()
            with open(saved_val_path, 'rb') as fp:
                valset = pickle.load(fp)
                fp.close()

        with open(saved_test_path, 'rb') as fp:
            testset = pickle.load(fp)
            fp.close()

    if args.test:  # Only load testing set
        train_loader = None
        val_loader = None
    else:
        # Training set
        train_set = TrafficDataset(trainset, args=args)
        train_loader = DataLoader(train_set,
                                  batch_size=args.train_batch_size,
                                  shuffle=True)

        # validation set
        val_set = TrafficDataset(valset, args=args)
        val_loader = DataLoader(val_set,
                                batch_size=args.val_batch_size,
                                shuffle=False)

    test_set = TrafficDataset(testset, args=args)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, total_timesteps, total_series
