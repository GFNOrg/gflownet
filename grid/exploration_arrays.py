import toy_grid_dag_exploration as main

import torch
import tqdm
import multiprocessing as mp
import itertools

def array_aug_18():

    global_lrs = [1e-4, 5e-4, 1e-3]
    global_seeds = list(range(10))

    base = {
        'mbsize': 16,
        'n_train_steps': 5000,
        'objective': 'q_full',
    }

    all_hps = (
        # Optimizers
        [{'opt': opt}
         for opt in ['adam', 'rmsprop', 'msgd']]
        +
        # Bootstrap
        [{'bootstrap_style': style,
          'bootstrap_tau': tau,
          'bootstrap_update_steps': update_steps,}
         for style in ['none', 'ema', 'frozen', 'double']
         for tau in ([0.25, 0.5, 0.7, 0.9] if style == 'ema' else [0])
         for update_steps in ([50, 100, 250] if style == 'frozen' else [0])]
        +
        # exploration
        [{'random_action_prob': eps,
          'sampling_temperature': temp,}
         for eps in [0, 0.01, 0.1]
         for temp in ([1, 0.8, 1.2, 2] if eps == 0 else [1])]
        +
        # replay
        [{'replay_strategy': strat,
          'replay_sample_size': sample_size,
          'replay_buf_size': bufsize,}
         for strat in ['none', 'top_k', 'uniform', 'prioritized']
         # For the n=4;H=16 corners task, the excepted #steps in a trajectory is ~30
         for sample_size in ([base['mbsize'] * 30] if strat in ['uniform', 'prioritized'] else [2,4,8])
         for bufsize in ([int(5e6)] if strat in ['uniform', 'prioritized'] else [100, 500, 1000])]
    )

    count = itertools.count(0)
    all_hps = [
        {**base, **hps,
         'learning_rate': lr,
         'seed': seed,
         'save_path': f'results/array_aug_18_q/{next(count)}.pkl.gz'}
        #'save_path': f'results/array_aug_18/{next(count)}.pkl.gz'}
        for hps in all_hps
        for lr in global_lrs
        for seed in global_seeds]

    return all_hps

def run(hps):
    args = main.parser.parse_args()
    for k,v in hps.items():
        setattr(args, k, v)
    import os
    if os.path.exists(args.save_path):
        return
    try:
        main.main(args)
    except Exception as e:
        print(args)
        raise e


if __name__ == '__main__':
    all_exps = array_aug_18()
    print(len(all_exps))

    import torch
    torch.set_num_threads(1)
    with mp.Pool(100) as pool:
        v = pool.imap(run, all_exps)
        for i in tqdm.tqdm(v, total=len(all_exps)):
            pass
