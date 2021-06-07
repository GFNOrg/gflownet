import argparse
from copy import copy, deepcopy
from collections import defaultdict
from datetime import timedelta
import concurrent.futures
import gc
import gzip
import os
import os.path as osp
import pickle
import psutil
import pdb
import subprocess
import sys
import threading
import time
import traceback
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn


from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from gflownet import Dataset, make_model, Proxy
import model_atom, model_block, model_fingerprint

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.99, type=float)
parser.add_argument("--nemb", default=256, help="#hidden", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=8, type=int)
parser.add_argument("--num_iterations", default=4000, type=int)
parser.add_argument("--num_conv_steps", default=6, type=int)
parser.add_argument("--log_reg_c", default=1e-2, type=float)
parser.add_argument("--reward_exp", default=4, type=float)
parser.add_argument("--reward_norm", default=10, type=float)
parser.add_argument("--sample_prob", default=1, type=float)
parser.add_argument("--clip_grad", default=0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
parser.add_argument("--replay_mode", default='online', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--array", default='array_may_18')
parser.add_argument("--repr_type", default='block_graph')
parser.add_argument("--model_version", default='v4')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--include_nblocks", default=False)
parser.add_argument("--save_path", default='results/ppo/')
parser.add_argument("--proxy_path", default='data/pretrained_proxy/')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--progress", default='yes')
parser.add_argument("--floatX", default='float64')


parser.add_argument("--ppo_clip", default=0.2, type=float)
parser.add_argument("--ppo_entropy_coef", default=1e-4, type=float)
parser.add_argument("--ppo_num_samples_per_step", default=256, type=float)
parser.add_argument("--ppo_num_epochs_per_step", default=32, type=float)




class PPODataset(Dataset):

    def __init__(self, args, bpath, device):
        super().__init__(args, bpath, device)
        self.current_dataset = []

    def _get_sample_model(self):
        m = BlockMoleculeDataExtended()
        traj = []
        for t in range(self.max_blocks):
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            with torch.no_grad():
                s_o, m_o = self.sampling_model(s)

            v = m_o[0, 1]
            logits = torch.cat([m_o[0,0].reshape(1), s_o.reshape(-1)])
            cat = torch.distributions.Categorical(
                logits=logits)
            action = cat.sample()
            lp = cat.log_prob(action)
            action = action.item()
            if t >= self.min_blocks and action == 0:
                r = self._get_reward(m)
                traj.append([m, (-1, 0), r, BlockMoleculeDataExtended(), 1, lp, v])
                break
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                m_new = self.mdp.add_block_to(m, *action)
                if len(m_new.blocks) and not len(m_new.stems) or t == self.max_blocks - 1:
                    r = self._get_reward(m_new)
                    traj.append([m, action, r, m_new, 1, lp, v])
                    m = m_new
                    break
                else:
                    traj.append([m, action, 0, m_new, 0, lp, v])
            m = m_new
        for i in range(len(traj)):
            traj[i].append(r) # The return is the terminal reward
            # the advantage, r + vsp * (1-done) - vs
            traj[i].append(traj[i][2] + (traj[i+1][6] if i < len(traj)-2 else 0) - traj[i][6])
        self.sampled_mols.append((r, m))
        return traj

    def sample2batch(self, mb):
        s, a, r, sp, d, lp, v, G, A = mb
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        a = torch.tensor(a, device=self._device).long()
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        lp = torch.tensor(lp, device=self._device).to(self.floatX)
        G = torch.tensor(G, device=self._device).to(self.floatX)
        A = torch.tensor(A, device=self._device).to(self.floatX)
        return s, a, r, d, lp, v, G, A

    def r2r(self, dockscore=None, normscore=None):
        if dockscore is not None:
            normscore = 4-(min(0, dockscore)-self.target_norm[0])/self.target_norm[1]
        normscore = max(self.R_min, normscore)
        return (normscore/self.reward_norm) ** self.reward_exp


    def start_samplers(self, n, mbsize):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n
        def f(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(self.sample(mbsize))
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()
        self.sampler_threads = [threading.Thread(target=f, args=(i,)) for i in range(n)]
        [setattr(i, 'failed', False) for i in self.sampler_threads]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
          while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]


_stop = [None]


def train_model_with_proxy(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = False
    device = torch.device('cuda')

    if num_steps is None:
        num_steps = args.num_iterations + 1

    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(model)

    if do_save:
        exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
        os.makedirs(exp_dir, exist_ok=True)


    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

    def save_stuff():
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols,
                    gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        pickle.dump(train_infos,
                    gzip.open(f'{exp_dir}/train_info.pkl.gz', 'wb'))


    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2))
    #opt = torch.optim.SGD(model.parameters(), args.learning_rate)

    #tf = lambda x: torch.tensor(x, device=device).float()
    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()

    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    last_losses = []

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    _stop[0] = stop_everything

    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    loginf = 1000 # to prevent nans
    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])
    clip_param = args.ppo_clip
    entropy_coef = args.ppo_entropy_coef

    for i in range(num_steps):
        samples = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(dataset._get_sample_model)
                       for i in range(args.ppo_num_samples_per_step)]
            for future in tqdm(concurrent.futures.as_completed(futures), leave=False):
                samples += future.result()
        for j in range(args.ppo_num_epochs_per_step):
            idxs = dataset.train_rng.randint(0, len(samples), args.mbsize)
            mb = [samples[i] for i in idxs]
            s, a, r, d, lp, v, G, A = dataset.sample2batch(zip(*mb))

            s_o, m_o = model(s)
            new_logprob = -model.action_negloglikelihood(s, a, 0, s_o, m_o)
            values = m_o[:, 1]
            ratio = torch.exp(new_logprob - lp)

            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                1.0 + clip_param) * A
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (G - values).pow(2).mean()
            m_p, s_p = model.out_to_policy(s, s_o, m_o)
            p = torch.zeros_like(m_p).index_add_(0, s.stems_batch, (s_p * torch.log(s_p)).sum(1))
            p = p + m_p * torch.log(m_p)
            entropy = -p.mean()
            loss = action_loss + value_loss - entropy * entropy_coef
            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                                                args.clip_grad)
            opt.step()

            last_losses.append((loss.item(), value_loss.item(), entropy.item()))
            train_losses.append((loss.item(), value_loss.item(), entropy.item()))

        if not i % 10:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses, G.mean().item())
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

        if not i % 25 and do_save:
            save_stuff()

    stop_everything()
    if do_save:
        save_stuff()
    return model


def main(args):
    bpath = "data/blocks_PDB_105.json"
    device = torch.device('cuda')

    if args.floatX == 'float32':
        args.floatX = torch.float
    else:
        args.floatX = torch.double

    dataset = PPODataset(args, bpath, device)
    print(args)


    mdp = dataset.mdp

    model = make_model(args, mdp, out_per_mol=2)
    model.to(torch.double)
    model.to(device)

    proxy = Proxy(args, bpath, device)

    train_model_with_proxy(args, model, proxy, dataset, do_save=True)
    print('Done.')



def array_may_18(args):
    base = {'nemb': 256,
    }

    all_hps = [
        {**base,},
    ]
    return all_hps

if __name__ == '__main__':
  args = parser.parse_args()
  if args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
    else:
      hps = all_hps[args.run]
      print(hps)
      for k,v in hps.items():
        setattr(args, k, v)
      main(args)
  else:
      try:
          main(args)
      except KeyboardInterrupt as e:
          print("stopping for", e)
          _stop[0]()
          raise e
      except Exception as e:
          print("exception", e)
          _stop[0]()
          raise e
