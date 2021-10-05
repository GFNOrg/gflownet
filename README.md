# Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation

Implementation for [our paper](https://arxiv.org/abs/2106.04399), submitted to NeurIPS 2021 (also check this high-level [blog post](http://folinoid.com/w/gflownet)).

This is a minimum working version of the code used for the paper, which is extracted from the internal repository of the [Mila Molecule Discovery](https://mila.quebec/en/ai-society/exascale-search-of-molecules/) project. Original commits are lost here, but the credit for this code goes to [@bengioe](https://github.com/bengioe), [@MJ10](https://github.com/MJ10) and [@MKorablyov](https://github.com/MKorablyov/) (see paper).

## Grid experiments

Requirements for base experiments: 
- `torch numpy scipy tqdm`

Additional requirements for active learning experiments: 
- `botorch gpytorch`


## Molecule experiments

Additional requirements:
- `pandas rdkit torch_geometric h5py ray`
- a few biochemistry programs, see `mols/Programs/README`

For `rdkit` in particular we found it to be easier to install through (mini)conda, but `rdkit-pypi` also works on `pip` in a vanilla python virtual environment. [`torch_geometric`](https://github.com/rusty1s/pytorch_geometric) has non-trivial installation instructions.

If you have CUDA 10.1 configured, you can run `pip install -r requirements.txt`. You can also change `requirements.txt` to match your CUDA version. (Replace cu101 to cuXXX, where XXX is your CUDA version).

We compress the 300k molecule dataset for size. To uncompress it, run `cd mols/data/; gunzip docked_mols.h5.gz`.
