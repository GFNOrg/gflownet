# Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation

Implementation for [[add arxiv link]], submitted to NeurIPS 2021.

This is a minimum working version of the code used for the paper, which is extracted from the internal repository of the [Mila Molecule Discovery](https://mila.quebec/en/ai-society/exascale-search-of-molecules/) project. Original commits are lost here, but the credit for this code goes to [@bengioe](https://github.com/bengioe), [@MJ10](https://github.com/MJ10) and [@MKorablyov](https://github.com/MKorablyov/) (see paper).

## Grid experiments

Requirements for base experiments: 
- `torch numpy scipy tqdm`

Additional requirements for active learning experiments: 
- `botorch gpytorch`


## Molecule experiments

Additional requirements:
- `pandas rdkit torch_geometric h5py`
- a few biochemistry programs, see `mols/Programs/README`

For `rdkit` in particular we found it to be easier to install through (mini)conda. [`torch_geometric`](https://github.com/rusty1s/pytorch_geometric) has non-trivial installation instructions.

We compress the 300k molecule dataset for size. To uncompress it, run `cd mols/data/; gunzip docked_mols.h5.gz`.

We omit docking routines since they are part of a separate contribution still to be submitted. These are available on demand, please do reach out to bengioe@gmail.com or mkorablyov@gmail.com.
