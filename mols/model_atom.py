"""
Code for an atom-based graph representation and architecture

"""

import warnings

from pandas.io.pytables import dropna_doc
warnings.filterwarnings('ignore')
import sys
import time
import os
import os.path as osp
import pickle
import gzip
import psutil
import subprocess


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

from utils import chem
from utils.chem import atomic_numbers

warnings.filterwarnings('ignore')


class MPNNet_v2(nn.Module):
    def __init__(self, num_feat=14, num_vec=3, dim=64,
                 num_out_per_mol=1, num_out_per_stem=105,
                 num_out_per_bond=1,
                 num_conv_steps=12, version='v1', dropout_rate=None):
        super().__init__()
        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_ops = num_out_per_stem
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps
        self.version = int(version[1:])
        self.dropout_rate = dropout_rate
        print('v:', self.version)
        assert 1<= self.version <= 6

        if self.version < 5:
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.SiLU()

        if self.version < 4:
            net = nn.Sequential(nn.Linear(4, 128), self.act, nn.Linear(128, dim * dim))
            self.conv = NNConv(dim, dim, net, aggr='mean')
        elif self.version == 4 or self.version == 6:
            self.conv = gnn.TransformerConv(dim, dim, edge_dim=4)
        else:
            self.convs = nn.Sequential(*[gnn.TransformerConv(dim, dim, edge_dim=4)
                                         for i in range(num_conv_steps)])

        #if self.version >= 6:
        #    self.g_conv = gnn.TransformerConv(dim, dim, heads=4)

        if self.version < 3:
            self.gru = nn.GRU(dim, dim)

        if self.version < 4:
            self.lin1 = nn.Linear(dim, dim * 8)
            self.lin2 = nn.Linear(dim * 8, num_out_per_stem)
        else:
            self.stem2out = nn.Sequential(nn.Linear(dim * 2, dim), self.act,
                                          nn.Linear(dim, dim), self.act,
                                          nn.Linear(dim, num_out_per_stem))
            #self.stem2out = nn.Sequential(nn.Linear(dim * 2, num_out_per_stem))

        if self.version < 3:
            self.set2set = Set2Set(dim, processing_steps=3)
        if self.version < 4:
            self.lin3 = nn.Linear(dim * 2 if self.version < 3 else dim, num_out_per_mol)
        else:
            self.lin3 = nn.Sequential(nn.Linear(dim, dim), self.act,
                                      nn.Linear(dim, dim), self.act,
                                      nn.Linear(dim, num_out_per_mol))
        self.bond2out = nn.Sequential(nn.Linear(dim * 2, dim), self.act,
                                      nn.Linear(dim, dim), self.act,
                                      nn.Linear(dim, num_out_per_bond))



    def forward(self, data, vec_data=None, do_stems=True, do_bonds=False, k=None, do_dropout=False):
        if self.version == 1:
            batch_vec = vec_data[data.batch]
            out = self.act(self.lin0(torch.cat([data.x, batch_vec], 1)))
        elif self.version > 1:
            out = self.act(self.lin0(data.x))
        h = out.unsqueeze(0)
        h = F.dropout(h, training=do_dropout, p=self.dropout_rate)

        if self.version < 4:
            for i in range(self.num_conv_steps):
                m = self.act(self.conv(out, data.edge_index, data.edge_attr))
                m = F.dropout(m, training=do_dropout, p=self.dropout_rate)
                out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
                h = F.dropout(h, training=do_dropout, p=self.dropout_rate)
                out = out.squeeze(0)
        elif self.version == 4 or self.version == 6:
            for i in range(self.num_conv_steps):
                out = self.act(self.conv(out, data.edge_index, data.edge_attr))
        else:
            for i in range(self.num_conv_steps):
                out = self.act(self.convs[i](out, data.edge_index, data.edge_attr))
        if self.version >= 4:
            global_out = gnn.global_mean_pool(out, data.batch)

        if do_stems:
            # Index of the origin atom of each stem in the batch, we
            # need to adjust for the batch packing)
            stem_batch_idx = (
                torch.tensor(data.__slices__['x'], device=out.device)[data.stems_batch]
                + data.stems)
            stem_atom_out = out[stem_batch_idx]
            #if self.version >= 6:
            #    per_stem_out = self.g_conv(stem)
            #    import pdb; pdb.set_trace()
            if self.version >= 4:
                stem_atom_out = torch.cat([stem_atom_out, global_out[data.stems_batch]], 1)
                per_stem_out = self.stem2out(stem_atom_out)
            else:
                per_stem_out = self.lin2(self.act(self.lin1(stem_atom_out)))
        else:
            per_stem_out = None

        if do_bonds:
            bond_data = out[data.bonds.flatten()].reshape((data.bonds.shape[0], -1))
            per_bond_out = self.bond2out(bond_data)

        if self.version < 3:
            global_out = self.set2set(out, data.batch)
            global_out = F.dropout(global_out, training=do_dropout, p=self.dropout_rate)
        per_mol_out = self.lin3(global_out) # per mol scalar outputs

        if hasattr(data, 'nblocks'):
            per_stem_out = per_stem_out * data.nblocks[data.stems_batch].unsqueeze(1)
            per_mol_out = per_mol_out * data.nblocks.unsqueeze(1)
            if do_bonds:
                per_bond_out = per_bond_out * data.nblocks[data.bonds_batch]

        if do_bonds:
            return per_stem_out, per_mol_out, per_bond_out
        return per_stem_out, per_mol_out


class MolAC_GCN(nn.Module):
    def __init__(self, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version, dropout_rate=0, do_stem_mask=True, do_nblocks=False):
        nn.Module.__init__(self)
        self.training_steps = 0
        # atomfeats + stem_mask + atom one hot + nblocks
        num_feat = (14 + int(do_stem_mask) + len(atomic_numbers) + int(do_nblocks))
        self.mpnn = MPNNet_v2(
            num_feat=num_feat,
            num_vec=nvec,
            dim=nhid,
            num_out_per_mol=num_out_per_mol,
            num_out_per_stem=num_out_per_stem,
            num_conv_steps=num_conv_steps,
            version=version,
            dropout_rate=dropout_rate)

    def out_to_policy(self, s, stem_o, mol_o):
        stem_e = torch.exp(stem_o)
        mol_e = torch.exp(mol_o[:, 0])
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
        stem_e = torch.exp(stem_o)
        mol_e = torch.exp(mol_o[:, 0])
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e
        mol_lsm = torch.log(mol_e / Z)
        stem_lsm = torch.log(stem_e / Z[s.stems_batch, None])
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_lsm.device)
        return -(
            stem_lsm[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_lsm * (a[:, 0] == -1))

    def index_output_by_action(self, s, stem_o, mol_o, a):
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
        return (
            stem_o[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_o * (a[:, 0] == -1))
    #(stem_o[stem_slices + a[:, 1]][torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0) + mol_o * (a[:, 0] == -1))

    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o

    def forward(self, graph, vec=None, do_stems=True, do_bonds=False, k=None, do_dropout=False):
        return self.mpnn(graph, vec, do_stems=do_stems, do_bonds=do_bonds, k=k, do_dropout=do_dropout)

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

def mol2graph(mol, mdp, floatX=torch.float, bonds=False, nblocks=False):
    rdmol = mol.mol
    if rdmol is None:
        g = Data(x=torch.zeros((1, 14 + len(atomic_numbers))),
                 edge_attr=torch.zeros((0, 4)),
                 edge_index=torch.zeros((0, 2)).long())
    else:
        atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol.mol, ifcoord=False,
                                                    one_hot_atom=True, donor_features=False)
        g = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
    stems = mol.stem_atmidxs
    if not len(stems):
        stems = [0]
    stem_mask = torch.zeros((g.x.shape[0], 1))
    stem_mask[torch.tensor(stems).long()] = 1
    g.stems = torch.tensor(stems).long()
    if nblocks:
        nblocks = (torch.ones((g.x.shape[0], 1,)).to(floatX) *
                   ((1 + mdp._cue_max_blocks - len(mol.blockidxs)) / mdp._cue_max_blocks))
        g.x = torch.cat([g.x, stem_mask, nblocks], 1).to(floatX)
        g.nblocks = nblocks[0] * mdp._cue_max_blocks
    else:
        g.x = torch.cat([g.x, stem_mask], 1).to(floatX)
    g.edge_attr = g.edge_attr.to(floatX)
    if bonds:
        if len(mol.jbonds):
            g.bonds = torch.tensor(mol.jbond_atmidxs).long()
        else:
            g.bonds = torch.zeros((1,2)).long()
    if g.edge_index.shape[0] == 0:
        g.edge_index = torch.zeros((2, 1)).long()
        g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).to(floatX)
        g.stems = torch.zeros((1,)).long()
    return g


def mols2batch(mols, mdp):
    batch = Batch.from_data_list(
        mols, follow_batch=['stems', 'bonds'])
    batch.to(mdp.device)
    return batch
