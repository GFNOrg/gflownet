import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.stats import norm
from scipy.spatial import distance_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import gpytorch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from torch.utils.data import TensorDataset, DataLoader

from toy_grid_dag import GridEnv, func_cos_N, func_corners_floor_A, func_corners_floor_B, func_corners
from toy_grid_dag import make_mlp, make_opt, SplitCategorical, compute_empirical_distribution_error, set_device
from toy_grid_dag import ReplayBuffer, FlowNetAgent, MARSAgent, MHAgent, RandomTrajAgent, PPOAgent



parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/e2e', type=str)
parser.add_argument("--init_data_path", default='results/e2e', type=str)
parser.add_argument("--learning_rate", default=2.5e-5, help="Learning rate", type=float)
parser.add_argument("--method", default='ppo', type=str)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--bootstrap_tau", default=0.1, type=float)
parser.add_argument("--kappa", default=0, type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--bufsize", default=16, help="MCMC buffer size", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--ndim", default=4, type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=300, type=int)
parser.add_argument("--num_empirical_loss", default=200000, type=int,
                    help="Number of samples used to compute the empirical distribution loss")
parser.add_argument('--func', default='corners_floor_B')
parser.add_argument("--num_val_iters", default=500, type=int)
parser.add_argument("--reward_topk", default=5, type=int)
parser.add_argument("--reward_lambda", default=0, type=float)
parser.add_argument("--inf_batch_size", default=32, type=int)
parser.add_argument("--num_samples", default=8, type=int)
parser.add_argument("--num_init_points", default=10, type=int)
parser.add_argument("--num_val_points", default=128, type=int)
parser.add_argument("--num_iter", default=10, type=int)
parser.add_argument("--use_model", action='store_true')

parser.add_argument("--replay_strategy", default='none', type=str) # top_k none
parser.add_argument("--replay_sample_size", default=2, type=int)
parser.add_argument("--replay_buf_size", default=100, type=float)

parser.add_argument("--ppo_num_epochs", default=32, type=int) # number of SGD steps per epoch
parser.add_argument("--ppo_epoch_size", default=16, type=int) # number of sampled minibatches per epoch
parser.add_argument("--ppo_clip", default=0.2, type=float)
parser.add_argument("--ppo_entropy_coef", default=4e-1, type=float)
parser.add_argument("--clip_grad_norm", default=0., type=float)


# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')
dev = torch.device('cpu')
_dev = [torch.device('cpu')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


class UCB:
    def __init__(self, model, kappa):
        self.model = model
        self.kappa = kappa

    def __call__(self, x):
        t_x = tf(np.array([[x]]))
        with torch.no_grad():
            output = self.model(t_x)
            mean, std = output.mean, torch.sqrt(output.variance)
        return torch.clamp(mean + self.kappa * std, min=0).item()

    def many(self, x):
        with torch.no_grad():
            output = self.model(tf(x))
            mean, std = output.mean, torch.sqrt(output.variance)
        return torch.clamp(mean + self.kappa * std, min=0)


def get_init_data(args, func):
    # Generate initial data to train proxy
    # import pdb; pdb.set_trace();
    env = GridEnv(args.horizon, args.ndim, func=func)
    td, end_states, true_r = env.true_density()
    idx = np.random.choice(len(end_states), args.num_init_points, replace=False)
    end_states = np.array(end_states)
    true_r = np.array(true_r)
    states, y = end_states[idx], true_r[idx]
    print(states[0])
    x = np.array([env.s2x(s) for s in states])
    init_data = x, y
    # data = np.dstack((end_states, true_r))[0]
    # np.random.shuffle(data)
    # init_data = data[:args.num_init_points]
    return init_data, td, end_states, true_r, env


def get_network_output(args, network, inputs, mean_std=False):
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, args.inf_batch_size, num_workers=0, shuffle=False)
    if not mean_std:
        outputs = []
        for batch in dataloader:
            outputs.append(network(batch[0].to(dev)))
        return torch.cat(outputs, dim=0)
    else:
        mean = []
        std = []
        for batch in dataloader:
            out = network(batch[0].to(dev))
            mean.append(out.mean.cpu())
            std.append(torch.sqrt(out.variance).cpu())
        return torch.cat(mean, dim=0), torch.cat(std, dim=0)


def generate_batch(args, agent, dataset, env):
    # Sample data from trained policy, given dataset.
    # Currently only naively samples data and adds to dataset, but should ideally also
    # have a diversity constraint based on existing dataset
    batch_s, sampled_x, sampled_y = [], [], []
    agent.sample_many(args.num_samples, batch_s)
    # import pdb; pdb.set_trace();
    # batch_x = np.array(batch_x)
    for s in batch_s:
        sampled_x.append(env.s2x(s))
        sampled_y.append(env.func(sampled_x[-1]))
    sampled_x = np.array(sampled_x)
    sampled_y = np.array(sampled_y)
    x, y = dataset

    x = torch.cat([x, tf(np.array(sampled_x))])
    y = torch.cat([y, tf(sampled_y)])
    return (x, y)


def update_proxy(args, data):
    # Train proxy(GP) on collected data
    train_x, train_y = data
    model = SingleTaskGP(train_x.to(dev), train_y.unsqueeze(-1).to(dev),
                         covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5), lengthscale_prior=gpytorch.priors.GammaPrior(0.5, 2.5)))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def diverse_topk_mean_reward(args, d_prev, d):
    topk_new, new_indices = torch.topk(d[1], k=args.reward_topk)
    topk_old, old_indices = torch.topk(d_prev[1], k=args.reward_topk)
    new_reward = topk_new.mean() + args.reward_lambda * get_pairwise_distances(d[0][new_indices].cpu().numpy())
    old_reward = topk_old.mean() + args.reward_lambda * get_pairwise_distances(d_prev[0][old_indices].cpu().numpy())
    return (new_reward - old_reward).item()


def get_pairwise_distances(arr):
    return np.mean(np.tril(distance_matrix(arr, arr))) * 2 / (arr.shape[0] * (arr.shape[0] - 1))


def main(args):
    args.dev = torch.device(args.device)
    set_device(args.dev)
    f = {'default': None,
         'cos_N': func_cos_N,
         'corners': func_corners,
         'corners_floor_A': func_corners_floor_A,
         'corners_floor_B': func_corners_floor_B,
    }[args.func]

    # Main Loop
    init_data, td, end_states, true_r, env = get_init_data(args, f)
    all_x, all_y = tf(end_states), tf(true_r)
    init_x, init_y = tf(init_data[0]), tf(init_data[1])
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    reward = []
    base_path = os.path.join(args.save_path, args.method)
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    # dataset = (init_x, init_y)
    dataset = torch.load(args.init_data_path)
    model = update_proxy(args, dataset)
    metrics = []
    for i in range(args.num_iter):
        model.eval()
        torch.save(dataset, os.path.join(base_path, f"dataset-aq-{i}.pth"))
        func = UCB(model, args.kappa) if args.use_model else f
        agent, _metrics = train_generative_model(args, func)
        metrics.append(_metrics)
        new_dataset = generate_batch(args, agent, dataset, env)
        reward.append(diverse_topk_mean_reward(args, dataset, new_dataset))
        print(reward)
        dataset = new_dataset
        # distrib_distances.append(metrics)
        model = update_proxy(args, dataset)
        pickle.dump({
            'metrics': metrics,
            'rewards': reward,
            'args': args
        }, gzip.open(os.path.join(base_path, 'result.pkl.gz'), 'wb'))


def train_generative_model(args, f):
    args.is_mcmc = args.method in ['mars', 'mcmc']

    env = GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
    envs = [GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
            for i in range(args.bufsize)]
    ndim = args.ndim

    if args.method == 'flownet':
        agent = FlowNetAgent(args, envs)
    elif args.method == 'mars':
        agent = MARSAgent(args, envs)
    elif args.method == 'mcmc':
        agent = MHAgent(args, envs)
    elif args.method == 'ppo':
        agent = PPOAgent(args, envs)
    elif args.method == 'random_traj':
        agent = RandomTrajAgent(args, envs)

    opt = make_opt(agent.parameters(), args)

    # metrics
    all_losses = []
    all_visited = []
    empirical_distrib_losses = []
    ttsr = max(int(args.train_to_sample_ratio), 1)
    sttr = max(int(1/args.train_to_sample_ratio), 1) # sample to train ratio

    if args.method == 'ppo':
        ttsr = args.ppo_num_epochs
        sttr = args.ppo_epoch_size

    for i in tqdm(range(args.n_train_steps+1), disable=not args.progress):
        data = []
        for j in range(sttr):
            data += agent.sample_many(args.mbsize, all_visited)
        for j in range(ttsr):
            losses = agent.learn_from(i * ttsr + j, data) # returns (opt loss, *metrics)
            if losses is not None:
                losses[0].backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                                   args.clip_grad_norm)
                opt.step()
                opt.zero_grad()
                all_losses.append([i.item() for i in losses])

        if not i % 100:
            empirical_distrib_losses.append(
                compute_empirical_distribution_error(env, all_visited[-args.num_empirical_loss:]))
            if args.progress:
                k1, kl = empirical_distrib_losses[-1]
                print('empirical L1 distance', k1, 'KL', kl)
                if len(all_losses):
                    print(*[f'{np.mean([i[j] for i in all_losses[-100:]]):.3f}'
                            for j in range(len(all_losses[0]))])

    # root = os.path.split(args.save_path)[0]
    # os.makedirs(root, exist_ok=True)
    # pickle.dump(
    metrics = {'losses': np.float32(all_losses),
         'model': agent.model.to('cpu') if agent.model else None,
         'visited': np.int8(all_visited),
         'emp_dist_loss': empirical_distrib_losses}# ,
        #  'true_d': env.true_density()[0],
        #  'args':args} #,
        # gzip.open(args.save_path, 'wb'))
    return agent, metrics

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
