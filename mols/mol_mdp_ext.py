from collections import defaultdict
import os.path
import numpy as np

from utils.molMDP import BlockMoleculeData, MolMDP
import utils.chem as chem
from rdkit import Chem
import networkx as nx

import model_atom, model_block, model_fingerprint


class BlockMoleculeDataExtended(BlockMoleculeData):

    @property
    def mol(self):
        return chem.mol_from_frag(jun_bonds=self.jbonds, frags=self.blocks)[0]

    @property
    def smiles(self):
        return Chem.MolToSmiles(self.mol)

    def copy(self): # shallow copy
        o = BlockMoleculeDataExtended()
        o.blockidxs = list(self.blockidxs)
        o.blocks = list(self.blocks)
        o.slices = list(self.slices)
        o.numblocks = self.numblocks
        o.jbonds = list(self.jbonds)
        o.stems = list(self.stems)
        return o

    def as_dict(self):
        return {'blockidxs': self.blockidxs,
                'slices': self.slices,
                'numblocks': self.numblocks,
                'jbonds': self.jbonds,
                'stems': self.stems}


class MolMDPExtended(MolMDP):

    def build_translation_table(self):
        """build a symmetry mapping for blocks. Necessary to compute parent transitions"""
        self.translation_table = {}
        for blockidx in range(len(self.block_mols)):
            # Blocks have multiple ways of being attached. By default,
            # a new block is attached to the target stem by attaching
            # it's kth atom, where k = block_rs[new_block_idx][0].
            # When computing a reverse action (from a parent), we may
            # wish to attach the new block to a different atom. In
            # the blocks library, there are duplicates of the same
            # block but with block_rs[block][0] set to a different
            # atom. Thus, for the reverse action we have to find out
            # which duplicate this corresponds to.

            # Here, we compute, for block blockidx, what is the index
            # of the duplicate block, if someone wants to attach to
            # atom x of the block.
            # So atom_map[x] == bidx, such that block_rs[bidx][0] == x
            atom_map = {}
            for j in range(len(self.block_mols)):
                if self.block_smi[blockidx] == self.block_smi[j]:
                    atom_map[self.block_rs[j][0]] = j
            self.translation_table[blockidx] = atom_map

        # We're still missing some "duplicates", as some might be
        # symmetric versions of each other. For example, block CC with
        # block_rs == [0,1] has no duplicate, because the duplicate
        # with block_rs [1,0] would be a symmetric version (both C
        # atoms are the "same").

        # To test this, let's create nonsense molecules by attaching
        # duplicate blocks to a Gold atom, and testing whether they
        # are the same.
        gold = Chem.MolFromSmiles('[Au]')
        # If we find that two molecules are the same when attaching
        # them with two different atoms, then that means the atom
        # numbers are symmetries. We can add those to the table.
        for blockidx in range(len(self.block_mols)):
            for j in self.block_rs[blockidx]:
                if j not in self.translation_table[blockidx]:
                    symmetric_duplicate = None
                    for atom, block_duplicate in self.translation_table[blockidx].items():
                        molA, _ = chem.mol_from_frag(
                            jun_bonds=[[0,1,0,j]],
                            frags=[gold, self.block_mols[blockidx]])
                        molB, _ = chem.mol_from_frag(
                            jun_bonds=[[0,1,0,atom]],
                            frags=[gold, self.block_mols[blockidx]])
                        if (Chem.MolToSmiles(molA) == Chem.MolToSmiles(molB) or
                            molA.HasSubstructMatch(molB)):
                            symmetric_duplicate = block_duplicate
                            break
                    if symmetric_duplicate is None:
                        raise ValueError('block', blockidx, self.block_smi[blockidx],
                                         'has no duplicate for atom', j,
                                         'in position 0, and no symmetrical correspondance')
                    self.translation_table[blockidx][j] = symmetric_duplicate
                    #print('block', blockidx, '+ atom', j,
                    #      'in position 0 is a symmetric duplicate of',
                    #      symmetric_duplicate)

    def parents(self, mol=None):
        """returns all the possible parents of molecule mol (or the current
        molecule if mol is None.

        Returns a list of (BlockMoleculeDataExtended, (block_idx, stem_idx)) pairs such that
        for a pair (m, (b, s)), MolMDPExtended.add_block_to(m, b, s) == mol.
        """
        if len(mol.blockidxs) == 1:
            # If there's just a single block, then the only parent is
            # the empty block with the action that recreates that block
            return [(BlockMoleculeDataExtended(), (mol.blockidxs[0], 0))]

        # Compute the how many blocks each block is connected to
        blocks_degree = defaultdict(int)
        for a,b,_,_ in mol.jbonds:
            blocks_degree[a] += 1
            blocks_degree[b] += 1
        # Keep only blocks of degree 1 (those are the ones that could
        # have just been added)
        blocks_degree_1 = [i for i, d in blocks_degree.items() if d == 1]
        # Form new molecules without these blocks
        parent_mols = []

        for rblockidx in blocks_degree_1:
            new_mol = mol.copy()
            # find which bond we're removing
            removed_bonds = [(jbidx, bond) for jbidx, bond in enumerate(new_mol.jbonds)
                             if rblockidx in bond[:2]]
            assert len(removed_bonds) == 1
            rjbidx, rbond = removed_bonds[0]
            # Pop the bond
            new_mol.jbonds.pop(rjbidx)
            # Remove the block
            mask = np.ones(len(new_mol.blockidxs), dtype=np.bool)
            mask[rblockidx] = 0
            reindex = new_mol.delete_blocks(mask)
            # reindex maps old blockidx to new blockidx, since the
            # block the removed block was attached to might have its
            # index shifted by 1.

            # Compute which stem the bond was using
            stem = ([reindex[rbond[0]], rbond[2]] if rblockidx == rbond[1] else
                    [reindex[rbond[1]], rbond[3]])
            # and add it back
            new_mol.stems = [list(i) for i in new_mol.stems] + [stem]
            #new_mol.stems.append(stem)
            # and we have a parent. The stem idx to recreate mol is
            # the last stem, since we appended `stem` in the back of
            # the stem list.
            # We also have to translate the block id to match the bond
            # we broke, see build_translation_table().
            removed_stem_atom = (
                rbond[3] if rblockidx == rbond[1] else rbond[2])
            blockid = mol.blockidxs[rblockidx]
            if removed_stem_atom not in self.translation_table[blockid]:
                raise ValueError('Could not translate removed stem to duplicate or symmetric block.')
            parent_mols.append([new_mol,
                                # action = (block_idx, stem_idx)
                                (self.translation_table[blockid][removed_stem_atom],
                                 len(new_mol.stems) - 1)])
        if not len(parent_mols):
            raise ValueError('Could not find any parents')
        return parent_mols


    def add_block_to(self, mol, block_idx, stem_idx=None, atmidx=None):
        '''out-of-place version of add_block'''
        #assert (block_idx >= 0) and (block_idx <= len(self.block_mols)), "unknown block"
        if mol.numblocks == 0:
            stem_idx = None
        new_mol = mol.copy()
        new_mol.add_block(block_idx,
                          block=self.block_mols[block_idx],
                          block_r=self.block_rs[block_idx],
                          stem_idx=stem_idx, atmidx=atmidx)
        return new_mol

    def remove_jbond_from(self, mol, jbond_idx=None, atmidx=None):
        new_mol = mol.copy()
        new_mol.remove_jbond(jbond_idx, atmidx)
        return new_mol

    def a2mol(self, acts):
        mol = BlockMoleculeDataExtended()
        for i in acts:
          if i[0] >= 0:
            mol = self.add_block_to(mol, *i)
        return mol

    def reset(self):
        self.molecule = BlockMoleculeDataExtended()
        return None


    def post_init(self, device, repr_type, include_bonds=False, include_nblocks=False):
        self.device = device
        self.repr_type = repr_type
        #self.max_bond_atmidx = max([max(i) for i in self.block_rs])
        self.max_num_atm = max(self.block_natm)
        # see model_block.mol2graph
        self.true_block_set = sorted(set(self.block_smi))
        self.stem_type_offset = np.int32([0] + list(np.cumsum([
            max(self.block_rs[self.block_smi.index(i)])+1 for i in self.true_block_set])))
        self.num_stem_types = self.stem_type_offset[-1]
        self.true_blockidx = [self.true_block_set.index(i) for i in self.block_smi]
        self.num_true_blocks = len(self.true_block_set)
        self.include_nblocks = include_nblocks
        self.include_bonds = include_bonds
        #print(self.max_num_atm, self.num_stem_types)
        self.molcache = {}

    def mols2batch(self, mols):
        if self.repr_type == 'block_graph':
            return model_block.mols2batch(mols, self)
        elif self.repr_type == 'atom_graph':
            return model_atom.mols2batch(mols, self)
        elif self.repr_type == 'morgan_fingerprint':
            return model_fingerprint.mols2batch(mols, self)

    def mol2repr(self, mol=None):
        if mol is None:
            mol = self.molecule
        #molhash = str(mol.blockidxs)+':'+str(mol.stems)+':'+str(mol.jbonds)
        #if molhash in self.molcache:
        #    return self.molcache[molhash]
        if self.repr_type == 'block_graph':
            r = model_block.mol2graph(mol, self, self.floatX)
        elif self.repr_type == 'atom_graph':
            r = model_atom.mol2graph(mol, self, self.floatX,
                                     bonds=self.include_bonds,
                                     nblocks=self.include_nblocks)
        elif self.repr_type == 'morgan_fingerprint':
            r = model_fingerprint.mol2fp(mol, self, self.floatX)
        #self.molcache[molhash] = r
        return r

    def get_nx_graph(self, mol: BlockMoleculeData, true_block=False):
        true_blockidx = self.true_blockidx

        G = nx.DiGraph()
        blockidxs = [true_blockidx[xx] for xx in mol.blockidxs] if true_block else mol.blockidxs

        G.add_nodes_from([(ix, {"block": blockidxs[ix]}) for ix in range(len(blockidxs))])

        if len(mol.jbonds) > 0:
            edges = []
            for jbond in mol.jbonds:
                edges.append((jbond[0], jbond[1],
                              {"bond": [jbond[2], jbond[3]]}))
                edges.append((jbond[1], jbond[0],
                              {"bond": [jbond[3], jbond[2]]}))
            G.add_edges_from(edges)
        return G

    def graphs_are_isomorphic(self, g1, g2):
        return nx.algorithms.is_isomorphic(g1, g2, node_match=node_match, edge_match=edge_match)

    
def node_match(x1, x2):
    return x1["block"] == x2["block"]


def edge_match(x1, x2):
    return x1["bond"] == x2["bond"]


def test_mdp_parent():
    mdp = MolMDPExtended("./data/blocks_PDB_105.json")
    mdp.build_translation_table()
    import tqdm
    rng = np.random.RandomState(142)
    nblocks = mdp.num_blocks

    # First let's test that the parent-finding method is
    # correct. i.e. let's test that the (mol, (parent, action)) pairs
    # are such that add_block_to(parent, action) == mol
    for i in tqdm.tqdm(range(10000)):
        mdp.molecule = mol = BlockMoleculeDataExtended()
        nblocks = rng.randint(1, 10)
        for i in range(nblocks):
            if len(mol.blocks) and not len(mol.stems): break
            mdp.add_block(rng.randint(nblocks), rng.randint(max(1, len(mol.stems))))
        parents = mdp.parents(mol)
        s = mol.smiles
        for p, (a,b) in parents:
            c = mdp.add_block_to(p, a, b)
            if c.smiles != s:
                # SMILES might differ but this might still be the same mol
                # we can check this way but its a bit more costly
                assert c.mol.HasSubstructMatch(mol.mol)

    # Now let's test whether we can always backtrack to the root from
    # any molecule without any errors
    for i in tqdm.tqdm(range(10000)):
        mdp.molecule = mol = BlockMoleculeDataExtended()
        nblocks = rng.randint(1, 10)
        for i in range(nblocks):
            if len(mol.blocks) and not len(mol.stems): break
            mdp.add_block(rng.randint(nblocks), rng.randint(max(1, len(mol.stems))))
        while len(mol.blocks):
            parents = mdp.parents(mol)
            mol = parents[rng.randint(len(parents))][0]

if __name__ == '__main__':
    test_mdp_parent()
