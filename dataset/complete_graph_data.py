import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
from collections import defaultdict
import torch.nn.functional as F
from utils.data_helper import *


class CompleteGraphData(object):

  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name
    self.max_num_nodes = config.model.max_num_nodes

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = 1
    # default order to make finding incomplete graph easier
    # order doesn't matter in GNN
    #config.model.num_canonical_order
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

    # adj_0 = np.array(nx.to_numpy_matrix(G))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    # degree_sequence = sorted(
    #     node_degree_list, key=lambda tt: tt[1], reverse=True)
    # adj_1 = np.array(
    #     nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### Degree ascent ranking
    # degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    # adj_2 = np.array(
    #     nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### BFS & DFS from largest-degree node
    CGs = [G]  #[G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    # node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
      node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)

      # bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
      dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

      # node_list_bfs += list(bfs_tree.nodes())
      node_list_dfs += list(dfs_tree.nodes())

    # adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
    adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))

    # ### k-core
    # num_core = nx.core_number(G)
    # core_order_list = sorted(list(set(num_core.values())), reverse=True)
    # degree_dict = dict(G.degree())
    # core_to_node = defaultdict(list)
    # for nn, kk in num_core.items():
    #   core_to_node[kk] += [nn]
    #
    # node_list = []
    # for kk in core_order_list:
    #   sort_node_tuple = sorted(
    #       [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
    #       key=lambda tt: tt[1],
    #       reverse=True)
    #   node_list += [nn for nn, dd in sort_node_tuple]
    #
    # adj_5 = np.array(nx.to_numpy_matrix(G, nodelist=node_list))


    adj_list = [adj_4]

    # print('number of nodes = {}'.format(adj_0.shape[0]))

    return adj_list, G.graph['i_nodes'], G.graph['j_nodes']

  def __getitem__(self, index):
    N = self.max_num_nodes

    real_index = index // 2
    # load graph
    loaded_data = pickle.load(open(self.file_names[real_index], 'rb'))
    adj_list, m, n = loaded_data
    num_nodes = adj_list[0].shape[0]

    if index % 2 == 0:
      # complete
      sample_incomplete = False
      num_subgraph_nodes = num_nodes
      # print('Full graph', 'num_nodes', num_nodes, 'num_subgraph_nodes', num_subgraph_nodes)
    else:
      sample_incomplete = True
      # possible incomplete graphs:
      all_possible_num_nodes = list(range( (n-1)*m+1, n*m )) + list(range(4*m+1, (n-1)*m+1, m))
      num_subgraph_nodes = random.choice(all_possible_num_nodes)
      # print('Incomplete graph', 'num_nodes', num_nodes, 'num_subgraph_nodes', num_subgraph_nodes)

    num_subgraphs = 1

    data_batch = []
    adj_block_list = []
    for ff in range(self.num_fwd_pass):  # 1 for grid
      edges = []
      subgraph_size = []
      subgraph_idx = []
      complete_graph_label = []
      subgraph_count = 0

      for ii in range(len(adj_list)):
        # loop over different orderings
        adj_full = adj_list[ii]
        # adj_tril = np.tril(adj_full, k=-1)

        idx = -1
        for jj in range(num_subgraph_nodes, num_subgraph_nodes+1):
          # loop over different subgraphs
          idx += 1

          if sample_incomplete:
            complete_graph_label += [0]
          else:
            complete_graph_label += [1]

          ### get graph for GNN propagation
          adj_block = adj_full[:jj, :jj]
          adj_block = np.tril(adj_block, k=-1)  # lower triangle, diagonal = 0
          adj_block = adj_block + adj_block.transpose()
          adj_block_list.append(adj_block)

          adj_block = torch.from_numpy(adj_block).to_sparse()
          edges += [adj_block.coalesce().indices().long()]

          subgraph_size += [num_subgraph_nodes]
          subgraph_idx += [
              np.ones((num_subgraph_nodes)).astype(np.int64) * subgraph_count
          ]  # TODO: do not understand this
          subgraph_count += 1

      ### adjust index basis for the selected subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for ii in range(len(edges)):
        edges[ii] += cum_size[ii]

      ### pack tensors
      data = {}
      data['adj'] = np.stack(adj_block_list, axis=0)  # do tril for each adj
      data['edges'] = torch.cat(edges, dim=1).t()  # return size [E, 2] after t()
      data['complete_graph_label'] = np.array(complete_graph_label)
      data['subgraph_idx'] = np.concatenate(subgraph_idx)
      data['subgraph_count'] = subgraph_count
      data['num_nodes'] = num_subgraph_nodes
      data['subgraph_size'] = subgraph_size
      data['num_count'] = sum(subgraph_size)
      data_batch += [data]

    return data_batch

  def __len__(self):
    return self.num_graphs * 2

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    batch_size = len(batch)
    # assert batch_size == 1
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      data['adj'] = torch.from_numpy(
          np.stack(
              [
                np.pad(
                  bb['adj'], ((0, 0), (0, 0), (0, pad_size[ii])),
                  'constant',
                  constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()  # B X C X N X N

      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)

      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0)

      if len(batch_pass) != 1:
        data['complete_graph_label'] = torch.from_numpy(
          np.concatenate(bb['complete_graph_label'] for bb in batch_pass).float())
      else:
        data['complete_graph_label'] = torch.from_numpy(batch_pass[0]['complete_graph_label']).float()

      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long()

      batch_data += [data]


    return batch_data
