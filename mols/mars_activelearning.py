import argparse
from copy import copy, deepcopy
from collections import defaultdict
from datetime import timedelta
import gc
import gzip
import os
import os.path as osp
import pickle
from types import prepare_class
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
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import QED
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
from torch.distributions.categorical import Categorical


from utils import chem
import ray
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended

import model_atom, model_block, model_fingerprint
from train_proxy import Dataset as _ProxyDataset
from mars import Dataset as GenModelDataset
from mars import SplitCategorical
tmp_dir = '/tmp/molexp/'
os.makedirs(tmp_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--proxy_learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--proxy_dropout", default=0.1, help="MC Dropout in Proxy", type=float)
parser.add_argument("--proxy_weight_decay", default=1e-6, help="Weight Decay in Proxy", type=float)
parser.add_argument("--proxy_mbsize", default=64, help="Minibatch size", type=int)
parser.add_argument("--proxy_opt_beta", default=0.9, type=float)
parser.add_argument("--proxy_nemb", default=64, help="#hidden", type=int)
parser.add_argument("--proxy_num_iterations", default=10000, type=int)
parser.add_argument("--num_init_examples", default=2000, type=int)
parser.add_argument("--num_outer_loop_iters", default=25, type=int)
parser.add_argument("--num_samples", default=200, type=int)
parser.add_argument("--proxy_num_conv_steps", default=12, type=int)
parser.add_argument("--proxy_repr_type", default='atom_graph')
parser.add_argument("--proxy_model_version", default='v2')
parser.add_argument("--save_path", default='results/')
parser.add_argument("--cpu_req", default=8)
parser.add_argument("--progress", action='store_true')
parser.add_argument("--include_nblocks", action='store_true')

# gen_model
parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.999, type=float)
parser.add_argument("--opt_epsilon", default=1e-8, type=float)
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--nemb", default=32, help="#hidden", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=7, type=int)
parser.add_argument("--num_iterations", default=150, type=int)
parser.add_argument("--num_conv_steps", default=12, type=int)
parser.add_argument("--log_reg_c", default=1e-1, type=float)
parser.add_argument("--reward_exp", default=4, type=float)
parser.add_argument("--reward_norm", default=1, type=float)
parser.add_argument("--R_min", default=0.1, type=float)
parser.add_argument("--sample_prob", default=1, type=float)
parser.add_argument("--clip_grad", default=0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
parser.add_argument("--buffer_size", default=4000, type=int)
parser.add_argument("--num_sgd_steps", default=25, type=int)
parser.add_argument("--random_action_prob", default=0.05, type=float)
parser.add_argument("--leaf_coef", default=10, type=float)
parser.add_argument("--replay_mode", default='online', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='atom_graph')
parser.add_argument("--model_version", default='v5')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--balanced_loss", default=True)
parser.add_argument("--floatX", default='float64')


class ProxyDataset(_ProxyDataset):
    def add_samples(self, samples):
        for m in samples:
            self.train_mols.append(m)


class Docker:
    def __init__(self, tmp_dir, cpu_req=2):
        self.target_norm = [-8.6, 1.10]
        self.dock = chem.DockVina_smi(tmp_dir) #, cpu_req=cpu_req) #, cpu_req=cpu_req)

    def eval(self, mol, norm=False):
        s = "None"
        try:
            s = Chem.MolToSmiles(mol.mol)
            print("docking {}".format(s))
            _, r, _ = self.dock.dock(s)
        except Exception as e: # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        if not norm:
            return r
        reward = -(r-self.target_norm[0])/self.target_norm[1]
        return reward

    def __call__(self, m):
        return self.eval(m)


class Proxy:
    def __init__(self, args, bpath, device):
        self.args = args
        # eargs = pickle.load(gzip.open(f'{args.proxy_path}/info.pkl.gz'))['args']
        # params = pickle.load(gzip.open(f'{args.proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.proxy_repr_type)
        self.mdp.floatX = torch.double
        self.proxy = make_model(args, self.mdp, is_proxy=True)
        # for a,b in zip(self.proxy.parameters(), params):
        #     a.data = torch.tensor(b, dtype=self.mdp.floatX)
        self.proxy.to(device)
        self.device = device

    def reset(self):
        for layer in self.proxy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train(self, dataset):
        self.reset()
        stop_event = threading.Event()
        best_model = self.proxy
        best_test_loss = 1000
        opt = torch.optim.Adam(self.proxy.parameters(), self.args.proxy_learning_rate, betas=(self.args.proxy_opt_beta, 0.999),
                                weight_decay=self.args.proxy_weight_decay)
        debug_no_threads = False
        mbsize = self.args.mbsize

        if not debug_no_threads:
            sampler = dataset.start_samplers(8, mbsize)

        last_losses = []

        def stop_everything():
            stop_event.set()
            print('joining')
            dataset.stop_samplers_and_join()

        train_losses = []
        test_losses = []
        time_start = time.time()
        time_last_check = time.time()

        max_early_stop_tolerance = 5
        early_stop_tol = max_early_stop_tolerance

        for i in range(self.args.proxy_num_iterations+1):
            if not debug_no_threads:
                r = sampler()
                for thread in dataset.sampler_threads:
                    if thread.failed:
                        stop_event.set()
                        stop_everything()
                        pdb.post_mortem(thread.exception.__traceback__)
                s, r = r
            else:
                p, pb, a, r, s, d = dataset.sample2batch(dataset.sample(mbsize))

            # state outputs
            stem_out_s, mol_out_s = self.proxy(s, None, do_stems=False)
            loss = (mol_out_s[:, 0] - r).pow(2).mean()
            loss.backward()
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            opt.step()
            opt.zero_grad()
            self.proxy.training_steps = i + 1

            if not i % 1000:
                last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
                print(i, last_losses)
                print('time:', time.time() - time_last_check)
                time_last_check = time.time()
                last_losses = []

                if i % 5000:
                    continue

                if 0:
                    # save_stuff()
                    continue

                t0 = time.time()
                total_test_loss = 0
                total_test_n = 0

                for s, r in dataset.itertest(max(mbsize, 128)):
                    with torch.no_grad():
                        stem_o, mol_o = self.proxy(s, None, do_stems=False)
                        loss = (mol_o[:, 0] - r).pow(2)
                        total_test_loss += loss.sum().item()
                        total_test_n += loss.shape[0]
                test_loss = total_test_loss / total_test_n
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model = deepcopy(self.proxy)
                    best_model.to('cpu')
                    early_stop_tol = max_early_stop_tolerance
                else:
                    early_stop_tol -= 1
                    print('test loss:', test_loss)
                    print('test took:', time.time() - t0)
                    test_losses.append(test_loss)

        stop_everything()
        self.proxy = deepcopy(best_model)
        self.proxy.to(self.device)
        print('Done.')

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()


def make_model(args, mdp, is_proxy=False):
    repr_type = args.proxy_repr_type if is_proxy else args.repr_type
    nemb = args.proxy_nemb if is_proxy else args.nemb
    num_conv_steps = args.proxy_num_conv_steps if is_proxy else args.num_conv_steps
    model_version = args.proxy_model_version if is_proxy else args.model_version
    if repr_type == 'block_graph':
        model = model_block.GraphAgent(nemb=nemb,
                                       nvec=0,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=1,
                                       num_conv_steps=num_conv_steps,
                                       mdp_cfg=mdp,
                                       version='v4')
    elif repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=num_conv_steps,
                                     version=model_version,
                                     dropout_rate=args.proxy_dropout)
    elif repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
    return model


_stop = [None]
def train_generative_model(args, model, proxy, dataset, num_steps=None, do_save=True):
    device = torch.device('cuda')
    debug_no_threads = False
    mdp = dataset.mdp
    model = model.double()
    proxy.proxy = proxy.proxy.double()
    stop_event = threading.Event()

    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

    opt = torch.optim.Adam(model.parameters(), args.learning_rate, #weight_decay=1e-4,
                           betas=(args.opt_beta, 0.99))

    tf = lambda x: torch.tensor(x, device=device).double()
    tint = lambda x: torch.tensor(x, device=device).long()

    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    num_threads = 8 if not debug_no_threads else 1
    last_losses = []

    def stop_everything():
        stop_event.set()
        print('joining')
        # dataset.stop_samplers_and_join()

    _stop[0] = stop_everything

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

    train_losses = []
    test_losses = []
    test_infos = []
    time_start = time.time()
    time_last_check = time.time()

    max_early_stop_tolerance = 5
    early_stop_tol = max_early_stop_tolerance
    loginf = 1000 # to prevent nans
    log_reg_c = 1e-1 # run 214
    log_reg_c = args.log_reg_c # run 217
    clip_loss = tf([args.clip_loss])

    for i in range(args.num_iterations+1):
        dataset.step_all(num_threads)
        for _ in tqdm(range(args.num_sgd_steps), leave=False):
            s, a = dataset.sample2batch(dataset.sample(mbsize))
            stem_out, mol_out, bond_out = model(s, None, do_bonds=True)
            bs = torch.tensor(s.__slices__['bonds'])
            ss = torch.tensor(s.__slices__['stems'])
            loss = 0
            for j in range(mbsize):
                sj = stem_out[ss[j]:ss[j+1]]
                bj = bond_out[bs[j]:bs[j+1]]
                cat = SplitCategorical(np.prod(sj.shape),
                                       logits=torch.cat([sj.flatten(), bj.flatten()]))
                lp = cat.log_prob(a[j])
                loss = (loss - lp) / mbsize

            opt.zero_grad()
            loss.backward()
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                                               args.clip_grad)
            opt.step()
        model.training_steps = i + 1
        if not i % 10:

            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses)
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []
            # save_stuff()

    stop_everything()
    # save_stuff()
    print('Done.')
    return model, dataset, {'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time()}

@ray.remote
class _SimDockLet:
    def __init__(self, tmp_dir):
        self.dock = chem.DockVina_smi(tmp_dir)
        self.target_norm = [-8.6, 1.10]
        # self.attribute = attribute

    def eval(self, mol, norm=False):
        s = "None"
        try:
            s = Chem.MolToSmiles(mol.mol)
            print("docking {}".format(s))
            _, r, _ = self.dock.dock(s)
        except Exception as e: # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        if not norm:
            return r
        reward = -(r-self.target_norm[0])/self.target_norm[1]
        return reward



def sample_and_update_dataset(args, model, proxy_dataset, generator_dataset, dock_pool):
    # generator_dataset.set_sampling_model(model, docker, sample_prob=args.sample_prob)
    # sampler = generator_dataset.start_samplers(8, args.num_samples)
    print("Sampling")
    # sampled_mols = sampler()
    # generator_dataset.stop_samplers_and_join()
    # import pdb; pdb.set_trace()
    mdp = generator_dataset.mdp
    nblocks = mdp.num_blocks
    sampled_mols = []
    rews = []
    smis = []
    t=0
    while len(sampled_mols) < args.num_samples:
        mol = BlockMoleculeDataExtended()
        for i in range(args.max_blocks):
            s = mdp.mols2batch([mdp.mol2repr(mol)])
            stem_o, mol_o = model(s)
            logits = torch.cat([stem_o.flatten(), mol_o.flatten()])
            if i < args.min_blocks:
                logits[-1] = -20
            cat = Categorical(logits=logits)
            act = cat.sample().item()
            if act == logits.shape[0] - 1:
                break
            else:
                act = (act % nblocks, act // nblocks)
                mol = mdp.add_block_to(mol, block_idx=act[0], stem_idx=act[1])
            if not len(mol.stems):
                break
        if mol.mol is None:
            # print('skip', mol.blockidxs, mol.jbonds)
            continue
        # print('here')
        # t0=time.time()
        # score = docker.eval(mol, norm=False)
        # t += (time.time() - t0)
        # mol.reward = proxy_dataset.r2r(score)
        smis.append(mol.smiles)
        # rews.append(mol.reward)
        # print(mol.smiles, mol.reward)
        sampled_mols.append(mol)
    # print('Docking sim seq done in {}'.format(t))

    t0 = time.time()
    rews = list(dock_pool.map(lambda a, m: a.eval.remote(m), sampled_mols))
    t1 = time.time()
    print('Docking sim done in {}'.format(t1-t0))
    for i in range(len(sampled_mols)):
        sampled_mols[i].reward = rews[i]

    print("Computing distances")
    dists =[]
    for m1, m2 in zip(sampled_mols, sampled_mols[1:] + sampled_mols[:1]):
        dist = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m1.mol), Chem.RDKFingerprint(m2.mol))
        dists.append(dist)
    print("Get batch rewards")
    rewards = []
    for m in sampled_mols:
        rewards.append(m.reward)
    print("Add to dataset")
    proxy_dataset.add_samples(sampled_mols)
    return proxy_dataset, rews, smis, {
        'dists': dists, 'rewards': rewards, 'reward_mean': np.mean(rewards), 'reward_max': np.max(rewards),
        'dists_mean': np.mean(dists), 'dists_sum': np.sum(dists)
    }


def main(args):
    ray.init()
    bpath = "data/blocks_PDB_105.json"
    device = torch.device('cuda')
    proxy_repr_type = args.proxy_repr_type
    repr_type = args.repr_type
    reward_exp = args.reward_exp
    reward_norm = args.reward_norm
    rews = []
    smis = []

    actors = [_SimDockLet.remote(tmp_dir)
                    for i in range(10)]
    pool = ray.util.ActorPool(actors)
    args.repr_type = proxy_repr_type
    args.replay_mode = "dataset"
    args.reward_exp = 1
    args.reward_norm = 1
    proxy_dataset = ProxyDataset(args, bpath, device, floatX=torch.float)
    proxy_dataset.load_h5("data/docked_mols.h5", args, num_examples=args.num_init_examples)
    rew_max = np.max(proxy_dataset.rews)
    rews.append(proxy_dataset.rews)
    smis.append([mol.smiles for mol in proxy_dataset.train_mols])

    print(np.max(proxy_dataset.rews))
    exp_dir = f'{args.save_path}/mars_{args.num_init_examples}_{args.array}_{args.run}/'
    os.makedirs(exp_dir, exist_ok=True)

    print(len(proxy_dataset.train_mols), 'train mols')
    print(len(proxy_dataset.test_mols), 'test mols')
    print(args)

    proxy = Proxy(args, bpath, device)
    mdp = proxy_dataset.mdp
    train_metrics = []
    metrics = []
    proxy.train(proxy_dataset)

    for i in range(args.num_outer_loop_iters):
        print(f"Starting step: {i}")
        # Initialize model and dataset for training generator
        args.sample_prob = 1
        args.repr_type = repr_type
        args.reward_exp = reward_exp
        args.reward_norm = reward_norm
        args.replay_mode = "online"
        gen_model_dataset = GenModelDataset(args, bpath, device, args.repr_type)
        model = make_model(args, gen_model_dataset.mdp)

        if args.floatX == 'float64':
            model = model.double()
        model.to(device)
        # train model with with proxy
        print(f"Training model: {i}")
        model, gen_model_dataset, training_metrics = train_generative_model(args, model, proxy, gen_model_dataset, do_save=False)

        print(f"Sampling mols: {i}")
        # sample molecule batch for generator and update dataset with docking scores for sampled batch
        _proxy_dataset, r, s, batch_metrics = sample_and_update_dataset(args, model, proxy_dataset, gen_model_dataset, pool)
        print(f"Batch Metrics: dists_mean: {batch_metrics['dists_mean']}, dists_sum: {batch_metrics['dists_sum']}, reward_mean: {batch_metrics['reward_mean']}, reward_max: {batch_metrics['reward_max']}")
        rews.append(r)
        smis.append(s)
        args.sample_prob = 0
        args.repr_type = proxy_repr_type
        args.replay_mode = "dataset"
        args.reward_exp = 1
        args.reward_norm = 1

        train_metrics.append(training_metrics)
        metrics.append(batch_metrics)

        proxy_dataset = ProxyDataset(args, bpath, device, floatX=torch.float)
        proxy_dataset.train_mols.extend(_proxy_dataset.train_mols)
        proxy_dataset.test_mols.extend(_proxy_dataset.test_mols)

        proxy = Proxy(args, bpath, device)
        mdp = proxy_dataset.mdp

        pickle.dump({'train_metrics': train_metrics,
                     'batch_metrics': metrics,
                     'rews': rews,
                     'smis': smis,
                     'rew_max': rew_max,
                     'args': args},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        print(f"Updating proxy: {i}")
        # update proxy with new data
        proxy.train(proxy_dataset)
    ray.shutdown()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
