import os

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
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


class ArimaDatasetP1(Dataset):

    def __init__(self, X, args, scaler=None):
        # save parameters
        self.args = args

        self.type = args.type
        self.out_seq_len = args.out_seq_len

        self.X = X
        self.n_timeslots, self.n_series = self.X.shape

        # learn scaler
        if args.use_scaler:
            if scaler is None:
                self.scaler = StandardScaler(copy=True)
                self.scaler.fit(self.X)
            else:
                self.scaler = scaler
            # transform if needed and convert to torch
            self.X_scaled = self.scaler.transform(self.X)
        else:
            self.scaler = None
            self.X_scaled = self.X

        # get valid start indices for sub-series
        self.indices = self.get_indices()

        if np.isnan(self.X).any():
            raise ValueError('Data has Nan')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.X_scaled[t:t + self.args.seq_len_x]  # step: t-> t + seq_x
        y = self.X[t + self.args.seq_len_x: t + self.args.seq_len_x + self.args.seq_len_y]

        x_gt = self.X[t:t + self.args.seq_len_x]  # step: t-> t + seq_x
        y_gt = self.X[t + self.args.seq_len_x: t + self.args.seq_len_x + self.args.seq_len_y]

        sample = {'x': x, 'y': y, 'x_gt': x_gt, 'y_gt': y_gt}
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
        T, D = self.X.shape
        indices = np.arange(T - self.args.seq_len_x - self.args.seq_len_y)
        return indices


class ArimaDatasetP2(Dataset):

    def __init__(self, X, args, scaler=None):
        # save parameters
        self.args = args

        self.type = args.type
        assert self.type == 'p2'

        # X: aggregated max data granularity T = seq_len_y. gt_data_set: actual data with granularity t = 1
        self.X, self.gt_data_set, in_seq_len = self.prepare_P2_data(X_full=X, args=args)

        self.seq_len_x = args.seq_len_x
        self.seq_len_y = args.seq_len_y
        self.in_seq_len = in_seq_len
        self.out_seq_len = args.out_seq_len

        self.n_timeslots, self.n_series = self.X.shape

        # learn scaler
        if args.use_scaler:
            if scaler is None:
                self.scaler = StandardScaler(copy=True)
                self.scaler.fit(self.X)
            else:
                self.scaler = scaler
            # transform if needed and convert to torch
            self.X_scaled = self.scaler.transform(self.X)
        else:
            self.scaler = None
            self.X_scaled = self.X

        # get valid start indices for sub-series
        self.indices = self.get_indices()

        if np.isnan(self.X).any():
            raise ValueError('Data has Nan')

    def prepare_P2_data(self, X_full, args):
        """
        Prepare data for P2 problem. (max tms in seq_len_y steps)
        :param X: Full data (10000 steps)
        :param args:
        :return:
        """

        train_size = int(X_full.shape[0] * 0.7)
        val_size = int(X_full.shape[0] * 0.1)

        in_seq_len_x = int(args.seq_len_x / args.seq_len_y)

        padding = in_seq_len_x * args.seq_len_y - args.dl_seq_len_x

        # as arima use more historial steps than DL models. The actual test sets of both approaches are the same.
        X = X_full[val_size + train_size - padding:]  # add more historical data to test set

        X_history = X[:args.seq_len_x]
        X_actual_test = X[args.seq_len_x:]

        max_X_history = [np.max(X_history[i:i + args.seq_len_y], axis=0)
                         for i in range(0, X_history.shape[0], args.seq_len_y)]
        max_X_actual_test = [np.max(X_actual_test[i:i + args.seq_len_y], axis=0)
                             for i in range(0, X_actual_test.shape[0], args.seq_len_y)]

        test_data = np.concatenate([max_X_history, max_X_actual_test], axis=0)  # actual test set for P2 (max tms)

        return test_data, X, in_seq_len_x

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.X_scaled[t:t + self.in_seq_len]  # step: t-> t + seq_x

        y = self.X[t + self.in_seq_len: t + self.in_seq_len + self.out_seq_len]

        y = y.reshape(1, -1)

        x_gt = self.gt_data_set[t * self.seq_len_y: t * self.seq_len_y + self.in_seq_len * self.seq_len_y]
        y_gt = self.gt_data_set[t * self.seq_len_y + self.in_seq_len * self.seq_len_y:
                                t * self.seq_len_y + self.in_seq_len * self.seq_len_y + self.seq_len_y]

        sample = {'x': x, 'y': y, 'x_gt': x_gt, 'y_gt': y_gt}
        return sample

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def get_indices(self):
        T, D = self.X_scaled.shape
        indices = np.arange(T - self.in_seq_len - self.out_seq_len)
        return indices


def load_raw(args):
    # load ground truth
    path = args.datapath

    data_path = os.path.join(path, 'data/{}.mat'.format(args.dataset))
    X = loadmat(data_path)['X']
    if len(X.shape) > 2:
        X = np.reshape(X, newshape=(X.shape[0], -1))

    return X


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
        train_size = 3 * 7 * 288  # 3 weeks
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

    train, val, test = train_test_split(X, args.dataset)

    if args.type == 'p1':
        test_set = ArimaDatasetP1(test, args=args)
    elif args.type == 'p2':
        test_set = ArimaDatasetP2(test, args=args)
    else:
        raise NotImplementedError('Dataset for {} is not implemented!'.format(args.type))

    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    return test_loader
