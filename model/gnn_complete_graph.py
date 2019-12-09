import time
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import softmax


EPS = np.finfo(np.float32).eps  # not used

__all__ = ['GNNStack']


class GNNStack(torch.nn.Module):
  def __init__(self, config):
    super(GNNStack, self).__init__()
    self.task = 'graph'
    self.max_num_nodes = config.model.max_num_nodes
    self.model_type = config.model.model_type
    self.num_head = config.model.num_head
    if self.model_type != 'GAT':
      self.num_head = 1
    self.num_layers = config.model.num_GraphGNN_layers
    self.embedding_dim = config.model.embedding_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = 1 #config.model.output_dim
    self.dropout = config.model.dropout
    self.dimension_reduce = config.model.dimension_reduce
    self.use_one_feature = hasattr(config.model, 'use_one_feature') and config.model.use_one_feature

    self.decoder_input = nn.Sequential(
      nn.Linear(self.max_num_nodes, self.embedding_dim))

    conv_model = self.build_conv_model(self.model_type, self.num_head)
    self.convs = nn.ModuleList()
    self.convs.append(conv_model(self.embedding_dim, self.hidden_dim))
    assert (self.num_layers >= 1), 'Number of layers is not >=1'
    for l in range(self.num_layers - 1):
      self.convs.append(conv_model(self.hidden_dim * self.num_head, self.hidden_dim))

    # post-message-passing
    self.post_mp = nn.Sequential(
      nn.Linear(self.hidden_dim * self.num_head, self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(self.dropout),
      nn.Linear(self.hidden_dim, self.output_dim))

    if not (self.task == 'node' or self.task == 'graph'):
      raise RuntimeError('Unknown task.')

  def build_conv_model(self, model_type, num_heads):
    if model_type == 'GCN':
      return pyg_nn.GCNConv
    elif model_type == 'GraphSage':
      return pyg_nn.SAGEConv
    elif model_type == 'GAT':
      # When applying GAT with num heads > 1, one needs to modify the
      # input and output dimension of the conv layers (self.convs),
      # to ensure that the input dim of the next layer is num heads
      # multiplied by the output dim of the previous layer.
      # HINT: In case you want to play with multiheads, you need to change the for-loop when builds up self.convs to be
      # self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim)),
      # and also the first nn.Linear(hidden_dim * num_heads, hidden_dim) in post-message-passing.
      return partial(pyg_nn.GATConv, heads=num_heads)

  def forward(self, data):
    # x, edge_index, batch = data.x, data.edge_index, data.batch
    edge_index = data['edges'].t()
    batch = data['subgraph_idx']


    if self.use_one_feature:
      x = torch.from_numpy(np.ones(batch.size(0), self.max_num_nodes))
      x.requires_grad_(True)
      x.to(batch.device)
    else:
      A_pad = data['adj']
      B, C, _, N_max = A_pad.shape
      A_pad = A_pad.view(-1, N_max)

      if self.dimension_reduce:
        x = self.decoder_input(A_pad)  # BCN X H
      else:
        x = A_pad  # BCN X N_max, N: num of nodes in current graph, != N_max


    ############################################################################
    # TODO: Your code here!
    # Each layer in GNN should consist of a convolution (specified in model_type),
    # a non-linearity (use RELU), and dropout.
    # HINT: the __init__ function contains parameters you will need. For whole
    # graph classification (as specified in self.task) apply max pooling over
    # all of the nodes with pyg_nn.global_max_pool as the final layer.
    # Our implementation is ~6 lines, but don't worry if you deviate from this.

    # TODO
    for conv in self.convs[:-1]:
      x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
    if self.task == 'graph':
      x = pyg_nn.global_max_pool(x, batch)

    ############################################################################

    x = self.post_mp(x)
    return x  # raw logit
