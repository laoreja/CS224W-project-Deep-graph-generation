import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from utils.data_helper import *


class GANData(object):

  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name
    self.max_num_nodes = config.model.max_num_nodes

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = config.model.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.dataset.num_fwd_pass
    self.is_overwrite_precompute = config.dataset.is_overwrite_precompute


    self.save_path = os.path.join(
        self.data_path, '{}_{}_{}_{}_{}_precompute'.format(
            config.model.name, config.dataset.name, tag, self.num_canonical_order, self.node_order))

    if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
      self.file_names = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)

      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        G = self.graphs[index]
        data = self._get_graph_data(G)
        tmp_path = os.path.join(self.save_path, '{}_{}.p'.format(tag, index))
        pickle.dump(data, open(tmp_path, 'wb'))
        self.file_names += [tmp_path]
    else:
      self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))

  def _get_graph_data(self, G):
    node_degree_list = [(n, d) for n, d in G.degree()]

    adj_0 = np.array(nx.to_numpy_matrix(G))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### Degree ascent ranking
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    adj_2 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### BFS & DFS from largest-degree node
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
      node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)

      bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
      dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

      node_list_bfs += list(bfs_tree.nodes())
      node_list_dfs += list(dfs_tree.nodes())

    adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
    adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))

    ### k-core
    num_core = nx.core_number(G)
    core_order_list = sorted(list(set(num_core.values())), reverse=True)
    degree_dict = dict(G.degree())
    core_to_node = defaultdict(list)
    for nn, kk in num_core.items():
      core_to_node[kk] += [nn]

    node_list = []
    for kk in core_order_list:
      sort_node_tuple = sorted(
          [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
          key=lambda tt: tt[1],
          reverse=True)
      node_list += [nn for nn, dd in sort_node_tuple]

    adj_5 = np.array(nx.to_numpy_matrix(G, nodelist=node_list))

    if self.num_canonical_order == 5:
      adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]
    else:
      if self.node_order == 'degree_decent':
        adj_list = [adj_1]
      elif self.node_order == 'degree_accent':
        adj_list = [adj_2]
      elif self.node_order == 'BFS':
        adj_list = [adj_3]
      elif self.node_order == 'DFS':
        adj_list = [adj_4]
      elif self.node_order == 'k_core':
        adj_list = [adj_5]
      elif self.node_order == 'DFS+BFS':
        adj_list = [adj_4, adj_3]
      elif self.node_order == 'DFS+BFS+k_core':
        adj_list = [adj_4, adj_3, adj_5]
      elif self.node_order == 'DFS+BFS+k_core+degree_decent':
        adj_list = [adj_4, adj_3, adj_5, adj_1]
      elif self.node_order == 'all':
        adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
      elif self.node_order == 'default':
        adj_list = [adj_0]

    # print('number of nodes = {}'.format(adj_0.shape[0]))

    return adj_list, G.graph['i_nodes'], G.graph['j_nodes']

  def __getitem__(self, index):
    N = self.max_num_nodes

    # load graph
    adj_list, m, n  = pickle.load(open(self.file_names[index], 'rb'))
    num_nodes = adj_list[0].shape[0]

    data_batch = []
    for ff in range(self.num_fwd_pass):  # 1 for grid
      ### pack tensors
      data = {}
      data['adj'] = np.stack(adj_list, axis=0)  # do tril for each adj
      data['m'] = m
      data['n'] = n
      data['num_nodes'] = num_nodes
      data['num_edges'] = np.sum(adj_list[0]) // 2
      data_batch += [data]
    return data_batch

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]

      data['num_nodes'] = torch.from_numpy(
        np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1)
      data['num_edges'] = torch.from_numpy(
        np.array([bb['num_edges'] for bb in batch_pass])).long().view(-1)
      data['m'] = torch.from_numpy(
        np.array([bb['m'] for bb in batch_pass])).long().view(-1)
      data['n'] = torch.from_numpy(
        np.array([bb['n'] for bb in batch_pass])).long().view(-1)

      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()  # B X C X N X N
      batch_data += [data]
    return batch_data
