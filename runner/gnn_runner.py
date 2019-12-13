from __future__ import (division, print_function)
import os
import time
import networkx as nx
import numpy as np
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
from decimal import Decimal

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.vis_helper import draw_graph_list, draw_graph_list_separate
from utils.data_parallel import DataParallel


try:
  ###
  # workaround for solving the issue of multi-worker
  # https://github.com/pytorch/pytorch/issues/973
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
  ###
except:
  pass

logger = get_logger('exp_logger')
__all__ = ['GNNRunner']

NPR = np.random.RandomState(seed=1234)

def get_graph(adj):
  """ get a graph from zero-padded adj """
  # remove all zeros rows and columns
  adj = adj[~np.all(adj == 0, axis=1)]
  adj = adj[:, ~np.all(adj == 0, axis=0)]
  adj = np.asmatrix(adj)
  G = nx.from_numpy_matrix(adj)
  return G


class GNNRunner(object):

  def __init__(self, config):
    self.config = config
    self.seed = config.seed
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device
    self.writer = SummaryWriter(config.save_dir)
    self.is_vis = config.test.is_vis
    self.better_vis = config.test.better_vis
    self.num_vis = config.test.num_vis
    self.vis_num_row = config.test.vis_num_row
    self.is_single_plot = config.test.is_single_plot
    self.num_gpus = len(self.gpus)
    self.is_shuffle = True  # TODO: grid graph not shuffled?

    assert self.use_gpu == True

    if self.train_conf.is_resume:
      self.config.save_dir = self.train_conf.resume_dir

    ### load graphs
    self.graphs = create_graphs(config.dataset.name, data_dir=config.dataset.data_path)
    
    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio

    self.num_graphs = len(self.graphs)
    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train

    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                   self.num_test_gt))

    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.graphs)

    self.graphs_train = self.graphs[:self.num_train]
    self.graphs_dev = self.graphs[:self.num_dev]
    self.graphs_test = self.graphs[self.num_train:]
    

    ### save split for benchmarking
    if config.dataset.is_save_split:      
      base_path = os.path.join(config.dataset.data_path, 'save_split')
      if not os.path.exists(base_path):
        os.makedirs(base_path)
      
      save_graph_list(
          self.graphs_train,
          os.path.join(base_path, '{}_train.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_dev,
          os.path.join(base_path, '{}_dev.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_test,
          os.path.join(base_path, '{}_test.p'.format(config.dataset.name)))

  def train(self):
    torch.autograd.set_detect_anomaly(True)

    ### create data loader
    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, tag='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,  # true for grid
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)
    criterion = nn.BCEWithLogitsLoss()

    if self.use_gpu:
      model = DataParallel(model, device_ids=self.gpus).to(self.device)
      criterion = criterion.cuda()
    model.train()

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    # TODO: not used?
    early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=self.train_conf.lr_decay_epoch,
        gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    best_acc = 0.
    # resume training
    # TODO: record resume_epoch to the saved file
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file = os.path.join(self.train_conf.resume_dir,
                                self.train_conf.resume_model)
      load_model(
          model.module if self.use_gpu else model,
          model_file,
          self.device,
          optimizer=optimizer,
          scheduler=lr_scheduler)
      resume_epoch = self.train_conf.resume_epoch

    # Training Loop
    iter_count = 0    
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      model.train()
      train_iterator = train_loader.__iter__()

      avg_acc_whole_epoch = 0.
      cnt = 0.

      for inner_iter in range(len(train_loader) // self.num_gpus):
        optimizer.zero_grad()

        batch_data = []
        if self.use_gpu:
          for _ in self.gpus:
            data = train_iterator.next()
            batch_data.append(data)
            iter_count += 1
        
        
        avg_train_loss = .0
        avg_acc = 0.
        for ff in range(self.dataset_conf.num_fwd_pass):
          batch_fwd = []
          
          if self.use_gpu:
            for dd, gpu_id in enumerate(self.gpus):
              data = {}
              data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
              data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
              # data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              # data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              # data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              # data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['complete_graph_label'] = batch_data[dd][ff]['complete_graph_label'].pin_memory().to(gpu_id, non_blocking=True)
              batch_fwd.append((data,))


          pred = model(*batch_fwd)
          label = data['complete_graph_label'][:, None]
          train_loss = criterion(pred, label).mean()
          train_loss.backward()

          pred = (torch.sigmoid(pred) > 0.5).type_as(label)
          avg_acc += (pred.eq(label)).float().mean().item()

          avg_train_loss += train_loss.item()

          # assign gradient

        
        # clip_grad_norm_(model.parameters(), 5.0e-0)
        optimizer.step()
        lr_scheduler.step()
        avg_train_loss /= self.dataset_conf.num_fwd_pass  # num_fwd_pass always 1
        avg_acc /= self.dataset_conf.num_fwd_pass

        avg_acc_whole_epoch += avg_acc
        cnt += len(data['complete_graph_label'])
        
        # reduce
        self.writer.add_scalar('train_loss', avg_train_loss, iter_count)
        self.writer.add_scalar('train_acc', avg_acc, iter_count)
        results['train_loss'] += [avg_train_loss]
        results['train_acc'] += [avg_acc]
        results['train_step'] += [iter_count]

        # if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
        #   logger.info("NLL Loss @ epoch {:04d} iteration {:08d} = {}\tAcc = {}".format(epoch + 1, iter_count, train_loss, avg_acc))

      avg_acc_whole_epoch /= cnt
      is_new_best = avg_acc_whole_epoch > best_acc
      if is_new_best:
        logger.info('!!! New best')
        best_acc = avg_acc_whole_epoch
      logger.info("Avg acc = {} @ epoch {:04d}".format(avg_acc_whole_epoch, epoch + 1))

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0 or is_new_best:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
    
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    
    return 1

  def test(self):
    with torch.no_grad():
      ### create data loader
      test_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_test, tag='test')
      test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=False,  # true for grid
        num_workers=self.train_conf.num_workers,
        collate_fn=test_dataset.collate_fn,
        drop_last=False)

      self.config.save_dir_train = self.test_conf.test_model_dir

      ### load model
      model = eval(self.model_conf.name)(self.config)
      criterion = nn.BCEWithLogitsLoss()
      model_file = os.path.join(self.config.save_dir_train, self.test_conf.test_model_name)
      load_model(model, model_file, self.device)

      if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)
        criterion = criterion.cuda()

      model.eval()

      test_iterator = test_loader.__iter__()

      iter_count = 0

      total_count = 0.
      total_avg_test_loss = 0.
      total_avg_test_acc = 0.

      for inner_iter in range(len(test_loader) // self.num_gpus):
        batch_data = []
        if self.use_gpu:
          for _ in self.gpus:
            data = test_iterator.next()
            batch_data.append(data)
            iter_count += 1

        for ff in range(self.dataset_conf.num_fwd_pass):
          batch_fwd = []

          if self.use_gpu:
            for dd, gpu_id in enumerate(self.gpus):
              data = {}
              data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
              data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
              # data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              # data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              # data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              # data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['complete_graph_label'] = batch_data[dd][ff]['complete_graph_label'].pin_memory().to(gpu_id,
                                                                                                        non_blocking=True)
              batch_fwd.append(data)


          pred = model(*batch_fwd)
          label = data['complete_graph_label'][:, None]

          pred = (torch.sigmoid(pred) > 0.5).type_as(label)
          test_acc = (pred.eq(label)).float().mean().item()
          test_loss = criterion(pred, label).mean().item()

          total_count += pred.size(0)
          total_avg_test_loss += test_loss
          total_avg_test_acc += test_acc


        # reduce
        self.writer.add_scalar('test_loss', test_loss, iter_count)
        self.writer.add_scalar('train_acc', test_acc, iter_count)

        if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
          logger.info(
            "Test NLL Loss @ iteration {:08d} = {}\tAcc = {}".format(iter_count,
                                                                     test_loss,
                                                                     test_acc))

      logger.info(
        "Test final avg NLL Loss = {} Acc = {}".format(total_avg_test_loss / total_count,
                                                       total_avg_test_acc / total_count))
      self.writer.close()






