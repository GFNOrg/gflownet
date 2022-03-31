import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count, chain

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/example_branincurrin.pkl.gz', type=str)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true') # Shows a tqdm bar

# GFN
parser.add_argument("--method", default='flownet_tb', type=str)
parser.add_argument("--learning_rate", default=1e-2, help="Learning rate", type=float)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--mbsize", default=128, help="Minibatch size", type=int)
parser.add_argument("--n_hid", default=64, type=int)
parser.add_argument("--n_layers", default=3, type=int)
parser.add_argument("--n_train_steps", default=5000, type=int)

# Measurement
parser.add_argument("--n_distr_measurements", default=50, type=int)

# Training
parser.add_argument("--n_mp_procs", default=4, type=int)

# Env
parser.add_argument('--func', default='BraninCurrin')
parser.add_argument("--horizon", default=32, type=int)




_dev = [torch.device('cpu')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


def currin(x):
    x_0 = x[..., 0] / 2 + 0.5
    x_1 = x[..., 1] / 2 + 0.5
    factor1 = 1 - np.exp(- 1 / (2 * x_1 + 1e-10))
    numer = 2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
    denom = 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20
    return factor1 * numer / denom / 13.77 # Dividing by the max to help normalize

def branin(x):
    x_0 = 15 * (x[..., 0] / 2 + 0.5) - 5
    x_1 = 15 * (x[..., 1] / 2 + 0.5)
    t1 = (x_1 - 5.1 / (4 * np.pi ** 2) * x_0 ** 2
          + 5 / np.pi * x_0 - 6)
    t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x_0)
    return 1 - (t1 ** 2 + t2 + 10) / 308.13 # Dividing by the max to help normalize

class GridEnv:

    def __init__(self, horizon, ndim=2, xrange=[-1, 1], funcs=None,
                 obs_type='one-hot'):
        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.width = xrange[1] - xrange[0]
        self.funcs = (
            [lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01]
            if funcs is None else funcs)
        self.num_cond_dim = len(self.funcs) + 1
        self.xspace = np.linspace(*xrange, horizon)
        self._true_density = None
        self.obs_type = obs_type
        if obs_type == 'one-hot':
            self.num_obs_dim = self.horizon * self.ndim
        elif obs_type == 'scalar':
            self.num_obs_dim = self.ndim
        elif obs_type == 'tab':
            self.num_obs_dim = self.horizon ** self.ndim
        

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros(self.num_obs_dim + self.num_cond_dim)
        if self.obs_type == 'one-hot':
            z = np.zeros((self.horizon * self.ndim + self.num_cond_dim), dtype=np.float32)
            z[np.arange(len(s)) * self.horizon + s] = 1
        elif self.obs_type == 'scalar':
            z[:self.ndim] = self.s2x(s)
        elif self.obs_type == 'tab':
            idx = (s * (self.horizon ** np.arange(self.ndim))).sum()
            z[idx] = 1
        z[-self.num_cond_dim:] = self.cond_obs
        return z

    def s2x(self, s):
        return s / (self.horizon-1) * self.width + self.start

    def s2r(self, s):
        x = self.s2x(s)
        return (self.coefficients * np.array([i(x) for i in self.funcs])).sum() ** self.temperature

    def reset(self, coefs=None, temp=None):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        self.coefficients = np.random.dirichlet([1.5]*len(self.funcs)) if coefs is None else coefs
        self.temperature = np.random.gamma(2,1) if temp is None else temp
        self.cond_obs = np.concatenate([self.coefficients, [self.temperature]])
        return self.obs(), self.s2r(self._state), self._state

    def parent_transitions(self, s, used_stop_action):
        if used_stop_action:
            return [self.obs(s)], [self.ndim]
        parents = []
        actions = []
        for i in range(self.ndim):
            if s[i] > 0:
                sp = s + 0
                sp[i] -= 1
                if sp.max() == self.horizon-1: # can't have a terminal parent
                    continue
                parents += [self.obs(sp)]
                actions += [i]
        return parents, actions


    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.s2r(s), done, s


    def state_info(self):
        all_int_states = np.float32(list(itertools.product(*[list(range(self.horizon))]*self.ndim)))
        state_mask = (all_int_states == self.horizon-1).sum(1) <= 1
        pos = all_int_states[state_mask].astype('float')
        s = pos / (self.horizon-1) * (self.xspace[-1] - self.xspace[0]) + self.xspace[0]
        r = np.stack([f(s) for f in self.funcs]).T
        return s, r, pos

    
    
def make_mlp(l, act=nn.LeakyReLU, tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act()] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))



class FlowNet_TBAgent:
    def __init__(self, args, envs):
        self.model = make_mlp([envs[0].num_obs_dim + envs[0].num_cond_dim] +
                              [args.n_hid] * args.n_layers +
                              [args.ndim+1])
        self.Z = make_mlp([envs[0].num_cond_dim] + [args.n_hid // 2] * args.n_layers + [1])
        self.model.to(args.dev)
        self.n_forward_logits = args.ndim+1
        self.envs = envs
        self.ndim = args.ndim
        
    def forward_logits(self, x):
        return self.model(x)[:, :self.n_forward_logits]

    def parameters(self):
        return chain(self.model.parameters(), self.Z.parameters())

    def sample_many(self, mbsize):
        s = tf(np.float32([i.reset()[0] for i in self.envs]))
        done = [False] * mbsize
        
        Z = self.Z(torch.tensor([i.cond_obs for i in self.envs]).float())[:, 0]
        self._Z = Z.detach().numpy().reshape(-1)
        #traj_mass = list(traj_mass) # allows x[i] += y
        fwd_prob = [[i] for i in Z]
        bck_prob = [[] for i in range(mbsize)]
        # We will progressively add log P_F(s|), subtract log P_B(|s) and R(s)
        while not all(done):
            cat = Categorical(logits=self.model(s))
            acts = cat.sample()
            ridx = torch.tensor((np.random.uniform(0,1,acts.shape[0]) < 0.01).nonzero()[0])
            if len(ridx):
                racts = np.random.randint(0, cat.logits.shape[1], len(ridx))
                acts[ridx] = torch.tensor(racts)
            logp = cat.log_prob(acts)
            step = [i.step(a) for i,a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            p_a = [self.envs[0].parent_transitions(sp_state, a == self.ndim)
                   for a, (sp, r, done, sp_state) in zip(acts, step)]
            for i, (bi, lp, (_, r, d, sp)) in enumerate(zip(np.nonzero(np.logical_not(done))[0], logp, step)):
                fwd_prob[bi].append(logp[i])
                bck_prob[bi].append(torch.tensor(np.log(1/len(p_a[i][0]))).float())
                if d: bck_prob[bi].append(torch.tensor(np.log(r)).float())
                #traj_mass[bi] = traj_mass[bi] + (
                #    logp[i] - np.log(1/len(p_a[i][0])) - (np.log(r) if d else 0))
            c = count(0)
            m = {j:next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf(np.float32([i[0] for i in step if not i[2]]))

        numerator = torch.stack([sum(i) for i in fwd_prob])
        denominator = torch.stack([sum(i) for i in bck_prob])
        log_ratio = numerator - denominator
        return log_ratio

    def learn_from(self, it, batch):
        if type(batch) is list:
            log_ratio = torch.stack(batch, 0)
        else:
            log_ratio = batch
        loss = log_ratio.pow(2).mean()
        return loss, self._Z[0]

def make_opt(params, args):
    params = list(params)
    if not len(params):
        return None
    if args.opt == 'adam':
        opt = torch.optim.Adam(params, args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2),
                               weight_decay=1e-4)
    elif args.opt == 'msgd':
        opt = torch.optim.SGD(params, args.learning_rate, momentum=args.momentum)
    return opt


def compute_exact_dag_distribution(envs, agent, args):
    env = envs[0]
    stack = [np.zeros(env.ndim, dtype=np.int32)]
    state_prob = defaultdict(lambda: np.zeros(len(envs)))
    state_prob[tuple(stack[0])] += 1
    end_prob = {}
    opened = {}
    softmax = nn.Softmax(1)
    asd = tqdm(total=env.horizon ** env.ndim, disable=not args.progress or 1, leave=False)
    while len(stack):
        asd.update(1)
        s = stack.pop(0)
        p = state_prob[tuple(s)]
        if s.max() >= env.horizon - 1:
            end_prob[tuple(s)] = p
            continue
        policy = softmax(agent.forward_logits(
            torch.tensor(np.float32([i.obs(s) for i in envs])))).detach().numpy()
        end_prob[tuple(s)] = p * policy[:, -1]
        for i in range(env.ndim):
            sp = s + 0
            sp[i] += 1
            state_prob[tuple(sp)] += policy[:, i] * p
            if tuple(sp) not in opened:
                opened[tuple(sp)] = 1
                stack.append(sp)
    asd.close()
    all_int_states = np.int32(list(itertools.product(*[list(range(env.horizon))]*env.ndim)))
    state_mask = (all_int_states == env.horizon-1).sum(1) <= 1
    distribution = np.float32([
        end_prob[i] for i in map(tuple,all_int_states[state_mask])])
    return distribution

def worker(args, agent, events, outq):
    stop_event, backprop_barrier = events
    torch.set_num_threads(1)
    torch.manual_seed(os.getpid())
    np.random.seed(os.getpid())
    mbs = args.mbsize // args.n_mp_procs
    
    agent.envs = [GridEnv(args.horizon, args.ndim, funcs=agent.envs[0].funcs)
                  for i in range(mbs)]

    while not stop_event.is_set():
        data = agent.sample_many(mbs)
        losses = agent.learn_from(-1, data) # returns (opt loss, *metrics)
        losses[0].backward()
        outq.put([losses[0].item()] + list(losses[1:]))
        backprop_barrier.wait()

        
def main(args):
    args.dev = torch.device(args.device)
    args.ndim = 2 # Force this for Branin-Currin
    fs = [branin, currin]
    envs = [GridEnv(args.horizon, args.ndim, funcs=fs)
            for i in range(args.mbsize)]
    
    agent = FlowNet_TBAgent(args, envs)
    for i in agent.parameters():
        i.grad = torch.zeros_like(i)
    agent.model.share_memory()
    agent.Z.share_memory()

    assert args.mbsize % args.n_mp_procs == 0
    
    opt = make_opt(agent.model.parameters(), args)
    optZ = make_opt(agent.Z.parameters(), args)
              
    # We want to test our model on a series of conditional configurations
    cond_confs = [
        ([a,1-a], temp)
        for a in np.linspace(0,1,11)
        for temp in [1,2,4,8,16]]
    test_envs = [GridEnv(args.horizon, args.ndim, funcs=fs)
                 for i in range(len(cond_confs))]

    stop_event, backprop_barrier = mp.Event(), mp.Barrier(args.n_mp_procs + 1)
    losses_q = mp.Queue()

    processes = [
        mp.Process(target=worker, args=(args, agent, (stop_event, backprop_barrier), losses_q))
        for i in range(args.n_mp_procs)]
    [i.start() for i in processes]
    
    all_losses = []
    distributions = []
    progress_bar = tqdm(range(args.n_train_steps + 1), disable=not args.progress)
    for t in progress_bar:
        while backprop_barrier.n_waiting < args.n_mp_procs:
            pass
        for i in processes:
            all_losses.append(losses_q.get())
        if len(all_losses):
            progress_bar.set_description_str(' '.join([
                f'{np.mean([i[j] for i in all_losses[-100:]]):.5f}'
                for j in range(len(all_losses[0]))]))

        if t % (args.n_train_steps // args.n_distr_measurements) == 0:
            for cfg, env in zip(cond_confs, test_envs):
                env.reset(*cfg)
            distributions.append(compute_exact_dag_distribution(test_envs, agent, args))
        # Workers add to the .grad even if they take the mean of the
        # loss, so let's divide here
        [i.grad.mul_(1 / args.n_mp_procs) for i in agent.parameters()]
        opt.step()
        opt.zero_grad()
        optZ.step()
        optZ.zero_grad()
        if t == args.n_train_steps:
            stop_event.set()
        backprop_barrier.wait() # Trigger barrier passing
        
    [i.join() for i in processes]
    
    for cfg, env in zip(cond_confs, test_envs):
        env.reset(*cfg)
    final_distribution = compute_exact_dag_distribution(test_envs, agent, args)
    
    results = {'losses': np.float32(all_losses),
               'params': [i.data.to('cpu').numpy() for i in agent.parameters()],
               'distributions': distributions,
               'final_distribution': final_distribution,
               'cond_confs': cond_confs,
               'state_info': envs[0].state_info(),
               'args':args}
    if args.save_path is not None:
        root = os.path.split(args.save_path)[0]
        if len(root):
            os.makedirs(root, exist_ok=True)
        pickle.dump(results, gzip.open(args.save_path, 'wb'))
    else:
        return results
    

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(4)
    main(args)
