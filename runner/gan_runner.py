from __future__ import (division, print_function)
import os
import os.path as osp
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
import torch.nn.functional as F
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

from runner import compute_edge_ratio, get_graph, evaluate

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
__all__ = ['GANRunner']

NPR = np.random.RandomState(seed=1234)

torch.autograd.set_detect_anomaly(True)


class GANRunner(object):

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

    assert self.use_gpu

    if self.train_conf.is_resume:
      self.config.save_dir = self.train_conf.resume_dir

    ### load graphs
    self.graphs = create_graphs(config.dataset.name, data_dir=config.dataset.data_path,
                                max_m=self.dataset_conf.max_m,
                                max_n=self.dataset_conf.max_n)
    
    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio
    self.num_graphs = len(self.graphs)
    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train
    self.num_test_gen = config.test.num_test_gen

    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                   self.num_test_gt))

    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.graphs)

    self.graphs_train = self.graphs[:self.num_train]
    self.graphs_dev = self.graphs[:self.num_dev]
    self.graphs_test = self.graphs[self.num_train:]
    
    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
    logger.info('No Edges vs. Edges in training set = {}'.format(
        self.config.dataset.sparse_ratio))

    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
    # Note this is diffenrent from the one in the config, only used in Erdos-Renyi exp
    self.max_num_nodes = len(self.num_nodes_pmf_train)
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()
    
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
    # model = eval(self.model_conf.name)(self.config)
    args = self.config.model
    n_labels = self.dataset_conf.max_m + self.dataset_conf.max_n
    G = define_G(args.nz, args.ngf, args.netG, args.final_activation, args.norm_G)
    D = define_D(args.ndf, args.netD, norm=args.norm_D)

    ### define losses
    criterionGAN = GANLoss(args.gan_mode)
    rote_loss = nn.L1Loss(reduction='none')
    if args.sparsity > 0.:
      sparse_loss = nn.L1Loss()

    if self.use_gpu:
      # G = DataParallel(G).to(self.device)
      # D = DataParallel(D).to(self.device)
      G = G.cuda()
      D = D.cuda()
      criterionGAN = criterionGAN.to(self.device)
      rote_loss = rote_loss.cuda()
      if args.sparsity > 0.:
        sparse_loss = sparse_loss.cuda()

    G.train()
    D.train()

    # create optimizer
    G_params = filter(lambda p: p.requires_grad, G.parameters())
    D_params = filter(lambda p: p.requires_grad, D.parameters())
    optimizer_G = optim.Adam(G_params, lr=self.train_conf.lr, betas=(self.train_conf.beta1, 0.999))
    optimizer_D = optim.Adam(D_params, lr=self.train_conf.lr, betas=(self.train_conf.beta1, 0.999))
    fake_pool = ImagePool(args.pool_size)


    # resume training
    # TODO: record resume_epoch to the saved file
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file_G = os.path.join(self.train_conf.resume_dir,
                                'G_' + self.train_conf.resume_model)
      model_file_D = os.path.join(self.train_conf.resume_dir,
                                  'D_' + self.train_conf.resume_model)
      load_model(
          G,
        model_file_G,
          self.device,
          optimizer=optimizer_G)
      load_model(
          D,
        model_file_D,
          self.device,
          optimizer=optimizer_D)
      resume_epoch = int(osp.splitext(self.train_conf.resume_model)[0].split('_')[-1])
      #original: self.train_conf.resume_epoch

    # Training Loop
    iter_count = 0  # iter idx thoughout the whole training
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      train_iterator = train_loader.__iter__()

      for batch_data in train_iterator:
        set_requires_grad(D, False)
        # set_requires_grad(G, True)
        optimizer_G.zero_grad()

        iter_count += 1
        # assert in arg helper
        ff = 0
        data = {}
        data['adj'] = batch_data[ff]['adj'].pin_memory().to(self.config.device, non_blocking=True)
        data['m'] = batch_data[ff]['m'].to(self.config.device, non_blocking=True)
        data['n'] = batch_data[ff]['n'].to(self.config.device, non_blocking=True)

        batch_size = data['adj'].size(0)

        i_onehot = torch.zeros((batch_size, self.dataset_conf.max_m), requires_grad=True).pin_memory().to(self.config.device, non_blocking=True)
        i_onehot.scatter_(1, data['m'][:, None]-1, 1)
        j_onehot = torch.zeros((batch_size, self.dataset_conf.max_n), requires_grad=True).pin_memory().to(self.config.device, non_blocking=True)
        j_onehot.scatter_(1, data['n'][:, None]-1, 1)
        y_onehot = torch.cat((i_onehot, j_onehot), dim=1)

        if args.nz > n_labels:
          noise = torch.randn((batch_size, args.nz - n_labels, 1, 1), requires_grad=True).to(self.config.device, non_blocking=True)
          z_input = torch.cat((y_onehot.view(batch_size, n_labels, 1, 1), noise), dim=1)
        else:
          z_input = y_onehot.view(batch_size, n_labels, 1, 1)

        output = G(z_input)  # (B, 1, n, n)
        if self.model_conf.is_sym:
          output = torch.tril(output, diagonal=-1)
          output = output + output.transpose(2, 3)

        loss_G = 0.
        if args.sparsity > 0:
          loss_G_sparse =  sparse_loss(output, torch.tensor(0.).expand_as(output).cuda())
          loss_G += args.sparsity * loss_G_sparse
        if args.lambda_rote > 0:
          if args.final_activation == 'tanh':
            tmp_obj = (data['adj'] - 0.5) * 2
          else:
            tmp_obj = data['adj']
          loss_G_rote = rote_loss(output, tmp_obj)
          rote_mask = (loss_G_rote > 0.2).type_as(loss_G_rote)
          loss_G_rote = (loss_G_rote * rote_mask).mean()
          loss_G += args.lambda_rote * loss_G_rote



        # backward G

        loss_G_GAN = criterionGAN(D(output), True)
        loss_G += loss_G_GAN
        loss_G.backward()
        optimizer_G.step()

        # backward D
        set_requires_grad(D, True)
        # set_requires_grad(G, False)
        optimizer_D.zero_grad()
        real = data['adj']

        if args.final_activation == 'sigmoid':
            ones_soft = torch.rand_like(real) * 0.1 + 0.9
            zeros_soft = torch.rand_like(real) * 0.1
        elif args.final_activation == 'tanh':
            ones_soft = torch.rand_like(real) * 0.2 + 0.8
            zeros_soft = -(torch.rand_like(real) * 0.2 + 0.8)
        ones_mask = (real == 1.)
        zeros_mask = (real == 0.)
        real[ones_mask] = ones_soft[ones_mask]
        real[zeros_mask] = zeros_soft[zeros_mask]
        if self.model_conf.is_sym:
          real = torch.tril(real, diagonal=-1)
          real = real + real.transpose(2, 3)
        pred_real = D(real)
        loss_D_real = criterionGAN(pred_real, True)
        # Fake
        if args.pool_size:
            queried_fake = fake_pool.query(output.detach())
        else:
            queried_fake = output.detach()
        pred_fake = D(queried_fake)
        loss_D_fake = criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # reduce
        self.writer.add_scalar('train_loss_G', loss_G.item(), iter_count)
        self.writer.add_scalar('train_loss_D', loss_D.item(), iter_count)
        results['train_loss_G'] += [loss_G]
        results['train_loss_D'] += [loss_D]
        results['train_step'] += [iter_count]

        if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
          logger.info("@ epoch {:04d} iter {:08d} loss_G: {:.5f}, loss_G_GAN: {:.5f}, loss_D: {:.5f}, loss_D_real: {:.5f}, loss_D_fake: {:.5f}".format(epoch + 1, iter_count, loss_G.item(), loss_G_GAN.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()))
          if args.lambda_rote > 0:
            logger.info(
              "@ epoch {:04d} iter {:08d} loss_rote: {:.5f}".format(
                epoch + 1, iter_count, loss_G_rote.item()))

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(G, optimizer_G, self.config, epoch + 1, fname_prefix='G_')
        snapshot(D, optimizer_G, self.config, epoch + 1, fname_prefix='D_')

    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    return 1


  def test(self):
    self.config.save_dir_train = self.test_conf.test_model_dir

    ### test dataset
    test_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_test, tag='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=self.test_conf.batch_size,
        shuffle=False,
        num_workers=self.train_conf.num_workers,
        collate_fn=test_dataset.collate_fn,
        drop_last=False)

    ### load model
    args = self.config.model
    n_labels = self.dataset_conf.max_m + self.dataset_conf.max_n
    G = define_G(args.nz, args.ngf, args.netG, args.final_activation, args.norm_G)
    model_file_G = os.path.join(self.config.save_dir_train,
                                self.test_conf.test_model_name)

    load_model(G, model_file_G, self.device)
    if self.use_gpu:
      G = G.cuda() #nn.DataParallel(G).to(self.device)
    G.train()

    if not hasattr(self.config.test, 'hard_multi') or not self.config.test.hard_multi:
      hard_thre_list = [None]
    else:
      hard_thre_list = np.arange(0.5, 1, 0.1)

    for test_hard_idx, hard_thre in enumerate(hard_thre_list):
      logger.info('Test pass {}. Hard threshold {}'.format(test_hard_idx, hard_thre))
      ### Generate Graphs
      A_pred = []
      gen_run_time = []

      for batch_data in test_loader:
        # asserted in arg helper
        ff = 0

        with torch.no_grad():
          data = {}
          data['adj'] = batch_data[ff]['adj'].pin_memory().to(self.config.device, non_blocking=True)
          data['m'] = batch_data[ff]['m'].to(self.config.device, non_blocking=True)
          data['n'] = batch_data[ff]['n'].to(self.config.device, non_blocking=True)

          batch_size = data['adj'].size(0)

          i_onehot = torch.zeros((batch_size, self.dataset_conf.max_m), requires_grad=True).pin_memory().to(self.config.device,
                                                                                                      non_blocking=True)
          i_onehot.scatter_(1, data['m'][:, None]-1, 1)
          j_onehot = torch.zeros((batch_size, self.dataset_conf.max_n), requires_grad=True).pin_memory().to(self.config.device,
                                                                                                      non_blocking=True)
          j_onehot.scatter_(1, data['n'][:, None]-1, 1)
          y_onehot = torch.cat((i_onehot, j_onehot), dim=1)

          if args.nz > n_labels:
            noise = torch.randn((batch_size, args.nz - n_labels, 1, 1), requires_grad=True).to(
              self.config.device, non_blocking=True)
            z_input = torch.cat((y_onehot.view(batch_size, n_labels, 1, 1), noise), dim=1)
          else:
            z_input = y_onehot.view(batch_size, n_labels, 1, 1)

          start_time = time.time()
          output = G(z_input).squeeze(1)  # (B, 1, n, n)
          if self.model_conf.final_activation == 'tanh':
            output = (output + 1) / 2
          if self.model_conf.is_sym:
            output = torch.tril(output, diagonal=-1)
            output = output + output.transpose(1, 2)
          gen_run_time += [time.time() - start_time]

          if hard_thre is not None:
            A_pred += [(output[batch_idx, ...] > hard_thre).long().cpu().numpy() for batch_idx in range(batch_size)]
          else:
            A_pred += [torch.bernoulli(output[batch_idx, ...]).long().cpu().numpy() for batch_idx in range(batch_size)]


      logger.info('Average test time per mini-batch = {}'.format(
        np.mean(gen_run_time)))

      graphs_gen = [get_graph(aa) for aa in A_pred]

      ### Visualize Generated Graphs
      if self.is_vis:
        num_col = self.vis_num_row
        num_row = self.num_vis // num_col
        test_epoch = self.test_conf.test_model_name
        test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
        if hard_thre is not None:
          save_name = os.path.join(self.config.save_dir_train, '{}_gen_graphs_epoch_{}_hard_{}.png'.format(
            self.config.test.test_model_name[:-4], test_epoch,
            int(round(hard_thre*10))))
          save_name2 = os.path.join(self.config.save_dir,
                                   '{}_gen_graphs_epoch_{}_hard_{}.png'.format(
                                     self.config.test.test_model_name[:-4], test_epoch,
                                     int(round(hard_thre * 10))))
        else:
          save_name = os.path.join(self.config.save_dir_train,
                                   '{}_gen_graphs_epoch_{}.png'.format(
                                     self.config.test.test_model_name[:-4], test_epoch))
          save_name2 = os.path.join(self.config.save_dir,
                                    '{}_gen_graphs_epoch_{}.png'.format(
                                      self.config.test.test_model_name[:-4], test_epoch))

        # remove isolated nodes for better visulization
        graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[:self.num_vis]]

        if self.better_vis:
          # actually not necessary with the following largest connected component selection
          for gg in graphs_pred_vis:
            gg.remove_nodes_from(list(nx.isolates(gg)))

        # display the largest connected component for better visualization
        vis_graphs = []
        for gg in graphs_pred_vis:
          if self.better_vis:
            CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            vis_graphs += [CGs[0]]
          else:
            vis_graphs += [gg]
        print('number of nodes after better vis', [tmp_g.number_of_nodes() for tmp_g in vis_graphs])

        if self.is_single_plot:
          # draw_graph_list(vis_graphs, num_row, num_col, fname=save_name, layout='spring')
          draw_graph_list(vis_graphs, num_row, num_col, fname=save_name2, layout='spring')
        else:
          # draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='spring')
          draw_graph_list_separate(vis_graphs, fname=save_name2[:-4], is_single=True, layout='spring')

        if test_hard_idx == 0:
          save_name = os.path.join(self.config.save_dir_train, 'train_graphs.png')

          if self.is_single_plot:
            draw_graph_list(
              self.graphs_train[:self.num_vis],
              num_row,
              num_col,
              fname=save_name,
              layout='spring')
          else:
            draw_graph_list_separate(
              self.graphs_train[:self.num_vis],
              fname=save_name[:-4],
              is_single=True,
              layout='spring')

      ### Evaluation
      if self.config.dataset.name in ['lobster']:
        acc = eval_acc_lobster_graph(graphs_gen)
        logger.info('Validity accuracy of generated graphs = {}'.format(acc))

      num_nodes_gen = [len(aa) for aa in graphs_gen]

      # Compared with Validation Set
      num_nodes_dev = [len(gg.nodes) for gg in self.graphs_dev]  # shape B X 1
      mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(self.graphs_dev, graphs_gen,
                                                                                       degree_only=False)
      mmd_num_nodes_dev = compute_mmd([np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

      # Compared with Test Set
      num_nodes_test = [len(gg.nodes) for gg in self.graphs_test]  # shape B X 1
      mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(self.graphs_test, graphs_gen,
                                                                                           degree_only=False)
      mmd_num_nodes_test = compute_mmd([np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

      logger.info(
        "Validation MMD scores of #nodes/degree/clustering/4orbits/spectral are = {:.4E}/{:.4E}/{:.4E}/{:.4E}/{:.4E}".format(Decimal(mmd_num_nodes_dev),
                                                                                                   Decimal(mmd_degree_dev),
                                                                                                   Decimal(mmd_clustering_dev),
                                                                                                   Decimal(mmd_4orbits_dev),
                                                                                                   Decimal(mmd_spectral_dev)))
      logger.info(
        "Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {:.4E}/{:.4E}/{:.4E}/{:.4E}/{:.4E}".format(Decimal(mmd_num_nodes_test),
                                                                                                   Decimal(mmd_degree_test),
                                                                                                   Decimal(mmd_clustering_test),
                                                                                                   Decimal(mmd_4orbits_test),
                                                                                                   Decimal(mmd_spectral_test)))

