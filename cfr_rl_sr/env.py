from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import pickle

import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from scipy.io import loadmat
from scipy.io import savemat


# import matplotlib.pyplot as plt


def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')


def get_path(graph, i, j, k):
    """
    get a path for flow (i, j) with middle point k
    return:
        - list of edges on path, list of nodes in path or (None, None) in case of duplicated path or non-simple path
    """
    p_ik = shortest_path(graph, i, k)
    p_kj = shortest_path(graph, k, j)

    edges_ik, edges_kj = [], []
    # compute edges from path p_ik, p_kj (which is 2 lists of nodes)
    for u, v in zip(p_ik[:-1], p_ik[1:]):
        edges_ik.append((u, v))
    for u, v in zip(p_kj[:-1], p_kj[1:]):
        edges_kj.append((u, v))
    return edges_ik, edges_kj


def get_paths(graph, i, j):
    """
    get all simple path for flow (i, j) on graph G
    return:
        - flows: list of paths
        - path: list of links on path (u, v)
    """
    if i != j:
        N = graph.number_of_nodes()

        path_edges = []
        for k in range(N):
            try:
                edges = get_path(graph, i, j, k)
                path_edges.append(edges)
            except nx.NetworkXNoPath:
                pass
        return path_edges
    else:
        return []


def get_segments(graph):
    n = graph.number_of_nodes()
    segments = {}
    segments_edges = Parallel(n_jobs=os.cpu_count() * 2)(delayed(get_paths)(graph, i, j)
                                                         for i, j in itertools.product(range(n), range(n)))
    for i, j in itertools.product(range(n), range(n)):
        segments[i, j] = segments_edges[i * n + j]

    return segments


def save(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


class Topology(object):
    def __init__(self, config, data_dir='./data/'):
        self.topology_file = data_dir + 'topo/' + config.topology_file
        self.shortest_paths_file = self.topology_file + '_shortest_paths'
        self.DG = nx.DiGraph()

        self.load_topology()
        self.calculate_paths()
        self.segment = self.compute_path(dataset=config.topology_file, datapath=data_dir)

    def compute_path(self, dataset, datapath):
        def load(path):
            with open(path, 'rb') as fp:
                obj = pickle.load(fp)
            return obj

        folder = os.path.join(datapath, 'topo/segments/2sr/')
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, '{}_segments_digraph.pkl'.format(dataset))
        if os.path.exists(path):
            print('|--- Load precomputed segment from {}'.format(path))
            data = load(path)
            segments = data['segments']
        else:
            segments = get_segments(self.DG)
            data = {
                'segments': segments,
            }
            save(path, data)

        return segments

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        header = header.split('\t')
        self.num_nodes = int(header[1])
        self.num_links = int(header[3])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty(self.num_links)
        self.link_weights = np.empty(self.num_links)
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = int(w)
            # self.DG.add_weighted_edges_from([(int(s), int(d), int(w))])
            self.DG.add_edge(int(s), int(d), weight=int(w),
                             capacity=float(c))

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        # print('nodes: %d, links: %d\n'%(self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG)
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.DG, s, d, weight='weight')))
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class Traffic(object):
    def __init__(self, config, num_nodes, data_dir='../data/', is_training=False):
        self.num_nodes = num_nodes

        splitted_data_fname = os.path.join(data_dir, 'splitted_data/{}.mat'.format(config.dataset))
        if not os.path.isfile(splitted_data_fname):
            self.split_data_from_mat(data_dir=data_dir, dataset=config.dataset)

        print('Load data from ', splitted_data_fname)
        data = loadmat(splitted_data_fname)
        if is_training:
            traffic_matrices = data['train']
        else:
            traffic_matrices = data['test']

        tms_shape = traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        self.traffic_matrices = np.reshape(traffic_matrices, newshape=(self.tm_cnt, num_nodes, num_nodes))
        self.traffic_file = splitted_data_fname
        print('Traffic matrices dims: [%d, %d, %d]\n' % (self.traffic_matrices.shape[0],
                                                         self.traffic_matrices.shape[1],
                                                         self.traffic_matrices.shape[2]))

    @staticmethod
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

    def train_test_split(self, X, dataset):
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
        elif 'uninett' in dataset:  # granularity: 5 min
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
            X_train = self.remove_outliers(X_train)
            X_val = self.remove_outliers(X_val)

        print('Raw data:')
        print('X_train: ', X_train.shape)
        print('X_val: ', X_val.shape)
        print('X_test: ', X_test.shape)

        return X_train, X_val, X_test

    def split_data_from_mat(self, data_dir, dataset):
        X = self.load_raw(data_dir=data_dir, dataset=dataset)
        train, val, test = self.train_test_split(X, dataset)
        savepath = os.path.join(data_dir, 'splitted_data/')
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        savepathfile = os.path.join(savepath, '{}.mat'.format(dataset))
        savemat(savepathfile, {'train': train,
                               'val': val,
                               'test': test})

    @staticmethod
    def load_raw(data_dir, dataset):
        # load raw data
        data_path = os.path.join(data_dir, 'data/{}.mat'.format(dataset))
        X = loadmat(data_path)['X']
        X = np.reshape(X, newshape=(X.shape[0], -1))
        return X

    def load_traffic(self, config):
        assert os.path.exists(self.traffic_file)
        print('[*] Loading traffic matrices...', self.traffic_file)
        f = open(self.traffic_file, 'r')
        traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            assert total_volume_cnt == self.num_nodes * self.num_nodes
            matrix = np.zeros((self.num_nodes, self.num_nodes))
            for v in range(total_volume_cnt):
                i = int(v / self.num_nodes)
                j = v % self.num_nodes
                if i != j:
                    matrix[i][j] = float(volumes[v])
            # print(matrix + '\n')
            traffic_matrices.append(matrix)

        f.close()
        self.traffic_matrices = np.array(traffic_matrices)

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        print('Traffic matrices dims: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class Environment(object):
    def __init__(self, config, is_training=False):
        self.data_dir = config.data_dir
        self.topology = Topology(config, self.data_dir)
        self.traffic = Traffic(config, self.topology.num_nodes, self.data_dir, is_training=is_training)
        self.traffic_matrices = self.traffic.traffic_matrices  # kbps
        # self.traffic_matrices = self.traffic.traffic_matrices * 100 * 8 / 300 / 1000  # kbps
        self.tm_cnt = self.traffic.tm_cnt
        self.traffic_file = self.traffic.traffic_file
        self.num_pairs = self.topology.num_pairs
        self.pair_idx_to_sd = self.topology.pair_idx_to_sd
        self.pair_sd_to_idx = self.topology.pair_sd_to_idx
        self.num_nodes = self.topology.num_nodes
        self.num_links = self.topology.num_links
        self.link_idx_to_sd = self.topology.link_idx_to_sd
        self.link_sd_to_idx = self.topology.link_sd_to_idx
        self.link_capacities = self.topology.link_capacities
        self.link_weights = self.topology.link_weights
        self.shortest_paths_node = self.topology.shortest_paths  # paths consist of nodes
        self.shortest_paths_link = self.convert_to_edge_path(self.shortest_paths_node)  # paths consist of links

    def convert_to_edge_path(self, node_paths):
        edge_paths = []
        num_pairs = len(node_paths)
        for i in range(num_pairs):
            edge_paths.append([])
            num_paths = len(node_paths[i])
            for j in range(num_paths):
                edge_paths[i].append([])
                path_len = len(node_paths[i][j])
                for n in range(path_len - 1):
                    e = self.link_sd_to_idx[(node_paths[i][j][n], node_paths[i][j][n + 1])]
                    assert e >= 0 and e < self.num_links
                    edge_paths[i][j].append(e)
                # print(i, j, edge_paths[i][j])

        return edge_paths
