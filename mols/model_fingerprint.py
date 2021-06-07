import numpy as np
import torch
import torch.nn as nn

import utils.chem as chem

fpe = [None]
FP_CONFIG = {
    "mol_fp_len": 512,
    "mol_fp_radiis": [3],
    "stem_fp_len": 64,
    "stem_fp_radiis": [4, 3, 2]
}

class MFP_MLP(nn.Module):

    def __init__(self, nhid, nvec, out_per_stem, out_per_mol):
        super().__init__()
        act = nn.LeakyReLU()

        self.m2h = nn.Linear(FP_CONFIG['mol_fp_len'], nhid) # mol
        self.s2h = nn.Linear(FP_CONFIG['stem_fp_len'] * FP_CONFIG['mol_fp_radiis'][0], nhid) # stem
        self.b2h = nn.Linear(FP_CONFIG['stem_fp_len'] * FP_CONFIG['mol_fp_radiis'][0], nhid) # bond

        self.h2stemp = nn.Sequential(nn.Linear(nhid * 2, nhid), act,
                                     nn.Linear(nhid, nhid), act,
                                     nn.Linear(nhid, nhid), act,
                                     nn.Linear(nhid, out_per_stem))

        self.h2molh = nn.Sequential(nn.Linear(nhid * 2 + nvec, nhid), act,
                                    nn.Linear(nhid, nhid))
        self.molh2o = nn.Sequential(nn.Linear(nhid, nhid), act,
                                    nn.Linear(nhid, out_per_mol))
        self.categorical_style = 'escort'
        self.escort_p = 4

    def forward(self, x, v):
        molx, stemx, stem_batch, bondx, bond_batch, _ = x
        molh = self.m2h(molx)
        stemh = self.s2h(stemx)
        bondh = self.b2h(bondx)

        # push bond, vec and mol info together
        per_bond_molh = self.h2molh(torch.cat([molh[bond_batch], bondh, v[bond_batch]], 1))
        # then reduce to molh
        molh = torch.zeros_like(molh).index_add_(0, bond_batch, per_bond_molh)
        # push stem and mol info (now mol+bond info) together
        per_stem_pred = self.h2stemp(torch.cat([molh[stem_batch], stemh], 1))
        # compute per-molecule outputs
        per_mol_pred = self.molh2o(molh)
        return per_stem_pred, per_mol_pred


    def out_to_policy(self, s, stem_o, mol_o):

        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o - 2)
            mol_e = torch.exp(mol_o[:, 0] - 2)
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            mol_e = abs(mol_o[:, 0])**self.escort_p
        #Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e
        Z = torch.zeros_like(mol_e).index_add_(0, s[2], stem_e.sum(1)) + mol_e + 1e-6
        mol_lsm = mol_e / Z
        stem_lsm = stem_e / Z[s[2], None]
        return mol_lsm, stem_lsm


    def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o - 2)
            mol_e = torch.exp(mol_o[:, 0] - 2)
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            mol_e = abs(mol_o[:, 0])**self.escort_p
        #Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e
        Z = torch.zeros_like(mol_e).index_add_(0, s[2], stem_e.sum(1)) + mol_e + 1e-6
        mol_lsm = torch.log(mol_e / Z + 1e-6)
        stem_lsm = torch.log(stem_e / Z[s[2], None] + 1e-6)
        #stem_slices=torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_lsm.device)
        stem_slices = s[5]
        #try:
        #    x = (stem_lsm.cpu()[stem_slices.cpu() + a.cpu()[:, 1]][
        #        torch.arange(a.shape[0]), a.cpu()[:, 0]] * (a.cpu()[:, 0] >= 0)
        #         + mol_lsm.cpu() * (a.cpu()[:, 0] == -1))
        #except:
        #    raise ValueError()

        return -(
            stem_lsm[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_lsm * (a[:, 0] == -1))

def mol2fp(mol, mdp):
    if fpe[0] is None:
        fpe[0] = chem.FPEmbedding_v2(
            FP_CONFIG['mol_fp_len'],
            FP_CONFIG['mol_fp_radiis'],
            FP_CONFIG['stem_fp_len'],
            FP_CONFIG['stem_fp_radiis'])
    # ask for non-empty stem and bond embeddings so that they at least
    # have shape (1, n), rather than (0, n) if there are not stems/bonds
    return list(map(torch.tensor,fpe[0](mol, non_empty=True)))  # mol_fp, stem_fps, jbond_fps


def mols2batch(mols, mdp):
    molx = torch.stack([i[0] for i in mols]).to(mdp.device)
    stemx = torch.cat([i[1] for i in mols], 0).to(mdp.device)
    stem_batch = torch.cat([torch.ones(i[1].shape[0], dtype=torch.long) * j
                            for j,i in enumerate(mols)]).to(mdp.device)
    bondx = torch.cat([i[2] for i in mols], 0).to(mdp.device)
    bond_batch = torch.cat([torch.ones(i[2].shape[0], dtype=torch.long) * j
                            for j,i in enumerate(mols)]).to(mdp.device)
    stem_slices = torch.tensor(np.cumsum([0]+[i[1].shape[0] for i in mols[:-1]]),
                               dtype=torch.long, device=mdp.device)
    return (molx, stemx, stem_batch, bondx, bond_batch, stem_slices)
