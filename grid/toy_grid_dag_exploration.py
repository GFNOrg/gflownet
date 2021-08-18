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
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from explo_util.scmsgd import SCMTDProp
from explo_util.sumtree import SumTree
from backpack import extend, backpack
from backpack.extensions import BatchGrad

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/12.pkl.gz', type=str)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')
parser.add_argument("--seed", default=0, type=int)

#
parser.add_argument("--method", default='flownet', type=str)
# Opt
parser.add_argument("--learning_rate", default=5e-4, help="Learning rate", type=float)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--clip_grad_norm", default=0., type=float)
# MSGD
parser.add_argument("--momentum", default=0.9, type=float)
# Adam
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
# SCMSGD -- really bad :3
parser.add_argument("--scm_beta1", default=0.9, type=float)
parser.add_argument("--scm_beta2", default=0.999, type=float)
parser.add_argument("--scm_diagonal", default=0, type=int)
parser.add_argument("--scm_block_diagonal", default=32, type=int)
parser.add_argument("--scm_true_batched", default=1, type=int)

# training loop params
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--n_train_steps", default=5000, type=int)
parser.add_argument("--num_empirical_loss", default=200000, type=int,
                    help="Number of samples used to compute the empirical distribution loss")
parser.add_argument("--do_tracking", default=1, type=int)
parser.add_argument("--do_empirical", default=0, type=int)
# Model
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)

# Env
parser.add_argument('--func', default='nmodes')
parser.add_argument("--horizon", default=16, type=int)
parser.add_argument("--ndim", default=5, type=int)

#parser.add_argument('--func', default='corners')
#parser.add_argument("--horizon", default=8, type=int)
#parser.add_argument("--ndim", default=4, type=int)

# Flownet
parser.add_argument("--bootstrap_style", default='none', type=str) # none ema frozen double
parser.add_argument("--bootstrap_tau", default=0., type=float)
parser.add_argument("--bootstrap_update_steps", default=0, type=int)
parser.add_argument("--objective", default='v_pi_trans', type=str) # q_full, v_pi_trans
parser.add_argument("--balanced_loss", default=1, type=int)
parser.add_argument("--loss_epsilon", default=1e-8, type=float)
parser.add_argument("--reward_exp", default=1, type=float)
parser.add_argument("--reward_ramping", default='none', type=str)
parser.add_argument("--reward_ramping_target", default=1, type=float)
parser.add_argument("--random_action_prob", default=0, type=float)
parser.add_argument("--sampling_temperature", default=1, type=float)


# Replay
parser.add_argument("--replay_strategy", default='none', type=str) # top_k none uniform prioritized
parser.add_argument("--replay_sample_size", default=2, type=int)
parser.add_argument("--replay_buf_size", default=10e6, type=float)





_dev = [torch.device('cpu')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

def set_device(dev):
    _dev[0] = dev


def func_corners(x):
    ax = abs(x)
    return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-1

def func_corners_floor_B(x):
    ax = abs(x)
    return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-2

def func_corners_floor_A(x):
    ax = abs(x)
    return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-3

def func_cos_N(x):
    ax = abs(x)
    return ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01


class Func_NModes:
    def __init__(self, n, ndim, track=True):
        self.modes = np.random.RandomState(142857).uniform(-1, 1, (n, ndim))
        # This tracks the per mode minimum visited
        self._tracked_min = np.ones(n) * 100 if track else None

    def __call__(self, x):
        if x.ndim == 1:
            x = x.reshape((1, self.modes.shape[1]))
        d = abs(self.modes[None, :] - x[:, None]).sum(2)
        if self._tracked_min is not None and d.shape[0] == 1:
            self._tracked_min = np.minimum(d[0], self._tracked_min)
        return d.min(-1)

class GridEnv:

    def __init__(self, horizon, ndim=2, xrange=[-1, 1], func=None, allow_backward=False):
        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.width = xrange[1] - xrange[0]
        self.func = (
            (lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01)
            if func is None else func)
        self.xspace = np.linspace(*xrange, horizon)
        self.allow_backward = allow_backward  # If true then this is a
                                              # MCMC ergodic env,
                                              # otherwise a DAG
        self._true_density = None

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z

    def s2x(self, s):
        return (self.obs(s).reshape((self.ndim, self.horizon)) * self.xspace[None, :]).sum(1)

    def reset(self):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        return self.obs(), self.func(self.s2x(self._state)), self._state

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
        if self.allow_backward:
            return self.step_chain(a, s)
        return self.step_dag(a, s)

    def step_dag(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.func(self.s2x(s)).item(), done, s

    def step_chain(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        sc = s + 0
        if a < self.ndim:
            s[a] = min(s[a]+1, self.horizon-1)
        if a >= self.ndim:
            s[a-self.ndim] = max(s[a-self.ndim]-1,0)

        reverse_a = ((a + self.ndim) % (2 * self.ndim)) if any(sc != s) else a

        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), self.func(self.s2x(s)), s, reverse_a

    def true_density(self):
        if self._true_density is not None:
            return self._true_density
        all_int_states = np.int32(list(itertools.product(*[list(range(self.horizon))]*self.ndim)))
        state_mask = np.array([len(self.parent_transitions(s, False)[0]) > 0 or sum(s) == 0
                               for s in all_int_states])
        all_xs = (np.float32(all_int_states) / (self.horizon-1) *
                  (self.xspace[-1] - self.xspace[0]) + self.xspace[0])
        traj_rewards = self.func(all_xs)[state_mask]
        self._true_density = (traj_rewards / traj_rewards.sum(),
                              list(map(tuple,all_int_states[state_mask])),
                              traj_rewards)
        return self._true_density

    def all_possible_states(self):
        """Compute quantities for debugging and analysis"""
        # all possible action sequences
        def step_fast(a, s):
            s = s + 0
            s[a] += 1
            return s
        f = lambda a, s: (
            [np.int32(a)] if np.max(s) == self.horizon - 1 else
            [np.int32(a+[self.ndim])]+sum([f(a+[i], step_fast(i, s)) for i in range(self.ndim)], []))
        all_act_seqs = f([], np.zeros(self.ndim, dtype='int32'))
        # all RL states / intermediary nodes
        all_int_states = list(itertools.product(*[list(range(self.horizon))]*self.ndim))
        # Now we need to know for each partial action sequence what
        # the corresponding states are. Here we can just count how
        # many times we moved in each dimension:
        all_traj_states = np.int32([np.bincount(i[:j], minlength=self.ndim+1)[:-1]
                                   for i in all_act_seqs
                                   for j in range(len(i))])
        # all_int_states is ordered, so we can map a trajectory to its
        # index via a sum
        arr_mult = np.int32([self.horizon**(self.ndim-i-1)
                             for i in range(self.ndim)])
        all_traj_states_idx = (
            all_traj_states * arr_mult[None, :]
        ).sum(1)
        # For each partial trajectory, we want the index of which trajectory it belongs to
        all_traj_idxs = [[j]*len(i) for j,i in enumerate(all_act_seqs)]
        # For each partial trajectory, we want the index of which state it leads to
        all_traj_s_idxs = [(np.bincount(i, minlength=self.ndim+1)[:-1] * arr_mult).sum()
                           for i in all_act_seqs]
        # Vectorized
        a = torch.cat(list(map(torch.LongTensor, all_act_seqs)))
        u = torch.LongTensor(all_traj_states_idx)
        v1 = torch.cat(list(map(torch.LongTensor, all_traj_idxs)))
        v2 = torch.LongTensor(all_traj_s_idxs)
        # With all this we can do an index_add, given
        # pi(all_int_states):
        def compute_all_probs(policy_for_all_states):
            """computes p(x) given pi(a|s) for all s"""
            dev = policy_for_all_states.device
            pi_a_s = torch.log(policy_for_all_states[u, a])
            q = torch.exp(torch.zeros(len(all_act_seqs), device=dev)
                                      .index_add_(0, v1, pi_a_s))
            q_sum = (torch.zeros((all_xs.shape[0],), device=dev)
                     .index_add_(0, v2, q))
            return q_sum[state_mask]
        # some states aren't actually reachable
        state_mask = np.bincount(all_traj_s_idxs, minlength=len(all_int_states)) > 0
        # Let's compute the reward as well
        all_xs = (np.float32(all_int_states) / (self.horizon-1) *
                  (self.xspace[-1] - self.xspace[0]) + self.xspace[0])
        traj_rewards = self.func(all_xs)[state_mask]
        # All the states as the agent sees them:
        all_int_obs = np.float32([self.obs(i) for i in all_int_states])
        print(all_int_obs.shape, a.shape, u.shape, v1.shape, v2.shape)
        return all_int_obs, traj_rewards, all_xs, compute_all_probs

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))


class ReplayBuffer:
    def __init__(self, args, env):
        self.buf = []
        self.strat = args.replay_strategy
        self.sample_size = args.replay_sample_size
        self.bufsize = int(args.replay_buf_size)
        self.env = env
        self.tree = SumTree(self.bufsize) if self.strat == 'prioritized' else None

    def add_endpoint(self, x, r_x):
        if self.strat == 'top_k':
            if len(self.buf) < self.bufsize or r_x > self.buf[0][0]:
                self.buf = sorted(self.buf + [(r_x, x)])[-self.bufsize:]

    def add_batch(self, batch):
        if self.strat == 'uniform':
            self.buf += batch
        elif self.strat == 'prioritized':
            s, e = len(self.buf), len(self.buf) + len(batch)
            self.buf += batch
            for i in range(s, e):
                self.tree.set(i, 1)

    def sample_extra(self):
        if len(self.buf) and self.strat == 'top_k':
            idxs = np.random.randint(0, len(self.buf), self.sample_size)
            return sum([self.generate_backward(*self.buf[i]) for i in idxs], [])
        return []

    def sample_batch(self, mbsize):
        if self.strat == 'uniform':
            self.last_idxs = np.random.randint(0, len(self.buf), mbsize)
            return [self.buf[i] for i in self.last_idxs]
        if self.strat == 'prioritized':
            self.last_idxs = np.int32([self.tree.sample() for i in range(mbsize)])
            return [self.buf[i] for i in self.last_idxs]
        return []

    def update_last_batch(self, losses):
        if self.strat == 'prioritized':
            for i, l in zip(self.last_idxs, losses):
                self.tree.set(i, l)


    def generate_backward(self, r, s0):
        s = np.int8(s0)
        os0 = self.env.obs(s)
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        used_stop_action = s.max() < self.env.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while s.sum() > 0:
            parents, actions = self.env.parent_transitions(s, used_stop_action)
            # add the transition
            traj.append([tf(i) for i in (parents, actions, [r], [self.env.obs(s)], [done])])
            # Then randomly choose a parent state
            if not used_stop_action:
                i = np.random.randint(0, len(parents))
                a = actions[i]
                s[a] -= 1
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r = 0
        return traj

class FlowNetAgent:
    def __init__(self, args, envs):
        self.do_v = args.objective in ['v_pi_trans']
        self.do_backward = args.objective in ['v_pi_trans']
        self.n_forward_logits = args.ndim+1
        self.n_logits = self.n_forward_logits * (int(self.do_backward) + 1)
        self.bootstrap_style = args.bootstrap_style
        if self.bootstrap_style == 'double':
            self.model = nn.Sequential(*
                [make_mlp([args.horizon * args.ndim] +
                          [args.n_hid] * args.n_layers +
                          [self.n_logits + int(self.do_v)])
                 for i in range(2)])
            self.model.to(args.dev)
            self.target = None
            self.detached_target = True
        else:
            self.model = make_mlp([args.horizon * args.ndim] +
                                  [args.n_hid] * args.n_layers +
                                  [self.n_logits + int(self.do_v)])
            self.model.to(args.dev)
            self.detached_target = False
            #self.model = extend(self.model)
            if self.bootstrap_style != 'none':
                self.target = copy.deepcopy(self.model)
            else:
                self.target = None

        self.envs = envs
        self.horizon = args.horizon
        self.ndim = args.ndim
        self.tau = args.bootstrap_tau
        self.bootstrap_update_steps = args.bootstrap_update_steps
        self.balanced_loss = args.balanced_loss
        self.replay = ReplayBuffer(args, envs[0])
        self.logsoftmax = torch.nn.LogSoftmax(1)
        self.epsilon = torch.tensor(args.loss_epsilon)
        self.sampling_temperature = torch.tensor(args.sampling_temperature)
        self.random_action_prob = args.random_action_prob

        self.reward_exp = beta = args.reward_exp
        self.reward_ramping = args.reward_ramping
        self.reward_ramping_target = tau = args.reward_ramping_target
        print(beta, tau,
              min(beta, 1 + (beta - 1) * (0) / tau),
              min(beta, 1 + (beta - 1) * (500) / tau),
              min(beta, 1 + (beta - 1) * (1000) / tau))
        if self.reward_ramping == 'none':
            self.transform_reward = lambda r, it: r ** beta
        elif self.reward_ramping == 'linear':
            self.transform_reward = lambda r, it: r ** min(beta, 1 + (beta - 1) * it / tau)
        elif self.reward_ramping == 'exp':
            self.transform_reward = lambda r, it: r ** ((tau + beta * it) / (tau + it))

        if args.objective == 'q_full':
            self.lf_fun = self.learn_from_q_full
        elif args.objective == 'v_pi_trans':
            self.lf_fun = self.learn_from_v_pi_trans

    def forward_logits(self, x):
        return self.model(x)[:, :self.n_forward_logits]

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited):
        batch = []
        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        while not all(done):
            # Note to self: this is ugly, ugly code
            with torch.no_grad():
                acts = Categorical(logits=self.forward_logits(s)/self.sampling_temperature).sample()
            if self.random_action_prob > 0:
                m = np.random.uniform(0,1,acts.shape[0]) < self.random_action_prob
                acts[m] = torch.tensor(np.random.randint(0, self.n_forward_logits, m.sum()))
            step = [i.step(a) for i,a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            p_a = [self.envs[0].parent_transitions(sp_state, a == self.ndim)
                   for a, (sp, r, done, sp_state) in zip(acts, step)]
            batch += [[tf(i) for i in (p, a, [r], [sp], [d])] + [sin.unsqueeze(0), ain.unsqueeze(0)]
                      for (p, a), (sp, r, d, _), sin, ain in zip(p_a, step, s, acts)]
            c = count(0)
            m = {j:next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])
            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(tuple(sp))
                    self.replay.add_endpoint(tuple(sp), r)
        batch += self.replay.sample_extra()
        return batch

    def _update_target(self, it):
        if self.tau > 0:
            for a,b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(1-self.tau).add_(self.tau*a)
        if self.bootstrap_update_steps > 0 and (it % self.bootstrap_update_steps == 0):
            for a,b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(0).add_(a)

    def _losses(self, in_flow, out_flow, done):
        if self.detached_target:
            out_flow = out_flow.detach()
        per_trans_loss = (in_flow - out_flow).pow(2)
        term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
        flow_loss = ((in_flow - out_flow) * (1-done)).pow(2).sum() / ((1-done).sum() + 1e-20)
        if self.balanced_loss:
            loss = 0.5 * (term_loss + flow_loss)
        else:
            loss = per_trans_loss.mean()

        self._in_flow = in_flow
        self._out_flow = out_flow
        self._per_trans_loss = per_trans_loss
        return loss, term_loss, flow_loss


    def learn_from(self, it, batch):
        if self.bootstrap_style == 'double':
            in_flow, out_flow, done = self.lf_fun(it, batch, self.model[it % 2], self.model[1 - it % 2])
            losses = self._losses(in_flow, out_flow, done)
        else:
            target = self.target if self.target is not None else self.model
            in_flow, out_flow, done = self.lf_fun(it, batch, self.model, target)
            self._update_target(it)
            losses = self._losses(in_flow, out_flow, done)
        return losses


    def learn_from_q_full(self, it, batch, model, target):
        loginf = tf([1000])
        batch_idxs = tl(sum([[i]*len(parents) for i, (parents,*_) in enumerate(batch)], []))
        parents, actions, r, sp, done, _, _ = map(torch.cat, zip(*batch))
        r = self.transform_reward(r, it)
        parents_Qsa = model(parents)[:parents.shape[0]][torch.arange(parents.shape[0]), actions.long()]
        in_flow = torch.log(torch.zeros((sp.shape[0],))
                            .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))
        next_q = target(sp)
        next_qd = next_q * (1-done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        out_flow = torch.logsumexp(torch.cat([torch.log(r)[:, None], next_qd], 1), 1)
        return in_flow, out_flow, done

    def learn_from_v_pi_trans(self, it, batch, model, target):
        loginf = tf([1000])
        parents, actions, r, sp, done, s, acts = map(torch.cat, zip(*batch))
        #acts = acts.long()
        #out = self.model(torch.cat([s, sp],0))
        r = self.transform_reward(r, it)
        out_s = model(s)#out[:s.shape[0]]
        log_pi_F_s = self.logsoftmax(out_s[:, :self.n_forward_logits])
        log_V_s = out_s[:, -1]
        out_sp = target(sp) #out[s.shape[0]:]

        allowed_backward = torch.cat([
            sp.reshape((sp.shape[0], self.ndim, self.horizon)).argmax(2).gt(0).float(),
            torch.zeros((sp.shape[0], 1))], 1)

        log_pi_B_sp = self.logsoftmax(
            out_sp[:, self.n_forward_logits:self.n_forward_logits*2] * allowed_backward +
            -loginf * (1-allowed_backward))
        log_V_sp = out_sp[:, -1] #
        # log(e + V(s) * pi(a|s))
        in_flow = torch.logaddexp(torch.log(self.epsilon),
                                  log_V_s + log_pi_F_s[torch.arange(acts.shape[0]), acts])
        # log(e + V(s') * pi(a|s') * (1-d) + R(s'))
        out_flow = torch.logaddexp(
            torch.log(self.epsilon),
            torch.logaddexp(
                # Here we're reusing acts, instead of having a proper
                # backward action.  For this grid environment it's
                # fine, but more complex envs might require more
                # tought.
                (log_V_sp + log_pi_B_sp[torch.arange(acts.shape[0]), acts]) * (1-done) - loginf * done,
                torch.log(r)))
        return in_flow, out_flow, done



def make_opt(params, args):
    params = list(params)
    if not len(params):
        return None
    if args.opt == 'adam':
        opt = torch.optim.Adam(params, args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2))
    elif args.opt == 'msgd':
        opt = torch.optim.SGD(params, args.learning_rate, momentum=args.momentum)
    elif args.opt == 'rmsprop':
        opt = torch.optim.RMSprop(params, args.learning_rate, args.adam_beta2)
    elif args.opt == 'scmsgd':
        opt = SCMTDProp(params, args.learning_rate, momentum=args.scm_beta1, beta2=args.scm_beta2,
                        diagonal=args.scm_diagonal, block_diagonal=args.scm_block_diagonal,
                        true_batched=args.scm_true_batched)
    return opt


def compute_empirical_distribution_error(env, visited, beta):
    if not len(visited):
        return 1, 100
    hist = defaultdict(int)
    for i in visited:
        hist[i] += 1
    td, end_states, true_r = env.true_density()
    true_density = tf(true_r ** beta / (true_r ** beta).sum())
    Z = sum([hist[i] for i in end_states])
    estimated_density = tf([hist[i] / Z for i in end_states])
    k1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
    return k1, kl

def main(args):
    args.dev = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_device(args.dev)
    f = {'default': None,
         'cos_N': func_cos_N,
         'corners': func_corners,
         'corners_floor_A': func_corners_floor_A,
         'corners_floor_B': func_corners_floor_B,
         'nmodes': Func_NModes(100, args.ndim),
    }[args.func]

    env = GridEnv(args.horizon, args.ndim, func=f, allow_backward=False)
    envs = [GridEnv(args.horizon, args.ndim, func=f, allow_backward=False)
            for i in range(args.mbsize)]
    ndim = args.ndim
    if args.progress:
        print(f'Approx n states {args.horizon ** args.ndim:_}')

    agent = FlowNetAgent(args, envs)

    opt = make_opt(agent.parameters(), args)
    do_self_supervised_step = args.opt in ['scmsgd']
    # metrics
    all_losses = []
    all_visited = []
    empirical_distrib_losses = []
    min_mode_distances = []
    transitions_trained_on = [0]
    do_empirical = args.do_empirical
    do_tracking = args.do_tracking
    do_replay_sampling = args.replay_strategy in ['uniform', 'prioritized']
    thresh_mod_dist = (2 / args.horizon) * args.ndim

    ttsr = max(int(args.train_to_sample_ratio), 1)
    sttr = max(int(1/args.train_to_sample_ratio), 1) # sample to train ratio


    for i in tqdm(range(args.n_train_steps+1), disable=not args.progress):
        data = []
        for j in range(sttr):
            data += agent.sample_many(args.mbsize, all_visited)
        if do_replay_sampling:
            agent.replay.add_batch(data)
        for j in range(ttsr):
            if do_replay_sampling:
                data = agent.replay.sample_batch(args.replay_sample_size)
            losses = agent.learn_from(i * ttsr + j, data) # returns (opt loss, *metrics)
            if do_replay_sampling:
                agent.replay.update_last_batch(agent._per_trans_loss.data.cpu().numpy())
            transitions_trained_on.append(transitions_trained_on[-1] + int(agent._per_trans_loss.shape[0]))
            if do_self_supervised_step and losses is not None:
                opt.backward_and_step(
                    agent._in_flow, agent._out_flow,
                    f_batch_idx=None if not hasattr(agent, '_batch_idxs') else agent._batch_idxs)
            elif losses is not None:
                losses[0].backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                                   args.clip_grad_norm)
                opt.step()
                opt.zero_grad()
            if losses is not None:
                all_losses.append([i.item() for i in losses])

        if not i % 100:
            if do_empirical:
                empirical_distrib_losses.append(
                    compute_empirical_distribution_error(env,
                                                         all_visited[-args.num_empirical_loss:],
                                                         args.reward_exp))
            if do_tracking:
                min_mode_distances.append(f._tracked_min + 0)
            if args.progress:
                if do_empirical:
                    k1, kl = empirical_distrib_losses[-1]
                    print('empirical L1 distance', k1, 'KL', kl)
                if do_tracking:
                    print('mode tracking', f._tracked_min.mean(), (f._tracked_min < thresh_mod_dist).mean())
                if len(all_losses):
                    print(*[f'{np.mean([i[j] for i in all_losses[-100:]]):.5f}'
                            for j in range(len(all_losses[0]))])

    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    pickle.dump(
        {'losses': np.float32(all_losses),
         'params': [i.data.to('cpu').numpy() for i in agent.parameters()],
         'visited': np.int8(all_visited),
         'emp_dist_loss': empirical_distrib_losses,
         'min_mode_dist': min_mode_distances,
         'true_d': env.true_density()[0] if do_empirical else None,
         'transitions_trained_on': np.float32(transitions_trained_on),
         'args':args},
        gzip.open(args.save_path, 'wb'))

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
