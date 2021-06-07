import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
from rdkit import Chem
from . import chem

class BlockMoleculeData:

    def __init__(self):
        self.blockidxs = []       # indexes of every block
        self.blocks = []          # rdkit molecule objects for every
        self.slices = [0]         # atom index at which every block starts
        self.numblocks = 0
        self.jbonds = []          # [block1, block2, bond1, bond2]
        self.stems = []           # [block1, bond1]
        self._mol = None

    def add_block(self, block_idx, block, block_r, stem_idx, atmidx):
        """

        :param block_idx:
        :param block:
        :param block_r:
        :param stem_idx:
        :param atmidx:
        :return:
        """
        self.blockidxs.append(block_idx)
        self.blocks.append(block)
        self.slices.append(self.slices[-1] + block.GetNumAtoms())
        self.numblocks += 1
        [self.stems.append([self.numblocks-1,r]) for r in block_r[1:]]

        if len(self.blocks)==1:
            self.stems.append([self.numblocks-1, block_r[0]])
        else:
            if stem_idx is None:
                assert atmidx is not None, "need stem or atom idx"
                stem_idx = np.where(self.stem_atmidxs==atmidx)[0][0]
            else:
                assert atmidx is None, "can't use stem and atom indices at the same time"

            stem = self.stems[stem_idx]
            bond = [stem[0], self.numblocks-1, stem[1], block_r[0]]
            self.stems.pop(stem_idx)
            self.jbonds.append(bond)
            # destroy properties
            self._mol = None
        return None

    def delete_blocks(self, block_mask):
        """

        :param block_mask:
        :return:
        """

        # update number of blocks
        self.numblocks = np.sum(np.asarray(block_mask, dtype=np.int32))
        self.blocks = list(np.asarray(self.blocks)[block_mask])
        self.blockidxs = list(np.asarray(self.blockidxs)[block_mask])

        # update junction bonds
        reindex = np.cumsum(np.asarray(block_mask,np.int32)) - 1
        jbonds = []
        for bond in self.jbonds:
            if block_mask[bond[0]] and block_mask[bond[1]]:
                jbonds.append(np.array([reindex[bond[0]], reindex[bond[1]], bond[2], bond[3]]))
        self.jbonds = jbonds

        # update r-groups
        stems = []
        for stem in self.stems:
            if block_mask[stem[0]]:
                stems.append(np.array([reindex[stem[0]],stem[1]]))
        self.stems = stems

        # update slices
        natms = [block.GetNumAtoms() for block in self.blocks]
        self.slices = [0] + list(np.cumsum(natms))

        # destroy properties
        self._mol = None
        return reindex

    def remove_jbond(self, jbond_idx=None, atmidx=None):

        if jbond_idx is None:
            assert atmidx is not None, "need jbond or atom idx"
            jbond_idx = np.where(self.jbond_atmidxs == atmidx)[0][0]
        else:
            assert atmidx is None, "can't use stem and atom indices at the same time"

        # find index of the junction bond to remove
        jbond = self.jbonds.pop(jbond_idx)

        # find the largest connected component; delete rest
        jbonds = np.asarray(self.jbonds, dtype=np.int32)
        jbonds = jbonds.reshape([len(self.jbonds),4]) # handle the case when single last jbond was deleted
        graph = csr_matrix((np.ones(self.numblocks-2),
                            (jbonds[:,0], jbonds[:,1])),
                           shape=(self.numblocks, self.numblocks))
        _, components = connected_components(csgraph=graph, directed=False, return_labels=True)
        block_mask = components==np.argmax(np.bincount(components))
        reindex = self.delete_blocks(block_mask)

        if block_mask[jbond[0]]:
            stem = np.asarray([reindex[jbond[0]], jbond[2]])
        else:
            stem = np.asarray([reindex[jbond[1]], jbond[3]])
        self.stems.append(stem)
        atmidx = self.slices[stem[0]] + stem[1]
        return atmidx

    @property
    def stem_atmidxs(self):
        stems = np.asarray(self.stems)
        if stems.shape[0]==0:
            stem_atmidxs = np.array([])
        else:
            stem_atmidxs = np.asarray(self.slices)[stems[:,0]] + stems[:,1]
        return stem_atmidxs

    @property
    def jbond_atmidxs(self):
        jbonds = np.asarray(self.jbonds)
        if jbonds.shape[0]==0:
            jbond_atmidxs = np.array([])
        else:
            jbond_atmidxs = np.stack([np.concatenate([np.asarray(self.slices)[jbonds[:,0]] + jbonds[:,2]]),
                                      np.concatenate([np.asarray(self.slices)[jbonds[:,1]] + jbonds[:,3]])],1)
        return jbond_atmidxs

    @property
    def mol(self):
        if self._mol == None:
            self._mol, _ = chem.mol_from_frag(jun_bonds=self.jbonds, frags=self.blocks)
        return self._mol

    @property
    def smiles(self):
        return Chem.MolToSmiles(self.mol)


class MolMDP:
    def __init__(self, blocks_file):
        blocks = pd.read_json(blocks_file)
        self.block_smi = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()
        self.block_nrs = np.asarray([len(r) for r in self.block_rs])
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in blocks["block_smi"]]
        self.block_natm = np.asarray([b.GetNumAtoms() for b in self.block_mols])
        self.reset()

    @property
    def num_blocks(self):
        "number of possible buildoing blocks in molMDP"
        return len(self.block_smi)

    def reset(self):
        self.molecule = BlockMoleculeData()
        return None

    def add_block(self, block_idx, stem_idx=None, atmidx=None):
        assert (block_idx >= 0) and (block_idx <= len(self.block_mols)), "unknown block"
        self.molecule.add_block(block_idx,
                                block=self.block_mols[block_idx],
                                block_r=self.block_rs[block_idx],
                                stem_idx=stem_idx, atmidx=atmidx)
        return None

    def remove_jbond(self, jbond_idx=None, atmidx=None):
        atmidx = self.molecule.remove_jbond(jbond_idx, atmidx)
        return atmidx

    def random_walk(self, length):
        done = False
        while not done:
            if self.molecule.numblocks==0:
                block_idx = np.random.choice(np.arange(self.num_blocks))
                stem_idx = None
                self.add_block(block_idx=block_idx, stem_idx=stem_idx)
                if self.molecule.numblocks >= length:
                    if self.molecule.slices[-1] > 1:
                        done = True
                    else:
                        self.reset()
            elif len(self.molecule.stems) > 0:
                block_idx = np.random.choice(np.arange(self.num_blocks))
                stem_idx = np.random.choice(len(self.molecule.stems))
                self.add_block(block_idx=block_idx, stem_idx=stem_idx)
                if self.molecule.numblocks >= length: done = True
            else:
                self.reset()
