# Grid environment exploration

In `toy_grid_dag_exploration.py` we test the common tricks of deep TD-based RL on GFlowNet.


Tricks tested:
- Target:
  - [x] Same
  - [x] Frozen target (DQN)
  - [x] Exponential moving average (DDPG?)
  - [x] Double target (DDQN)
  - Forward + Backward targets
- Replay:
  - [x] N-recent (DQN)
  - [x] Prioritized N-Recent (PER)
  - [x] Noisy top-k
- Optimizer
  - [x] Adam
  - [x] RMSProp
  - SCMSGD+TDProp -- doesn't really work & expensive
- Exploration
  - [x] e-greedy (DQN)
  - [x] sampling temperature
- Sampling beta
  - [x] Fixed
  - [x] Linear ramping, c, `beta_t = min(beta, 1 + (beta-1) * t / c)`
  - [x] exponential ramping, c, `beta_t = (c + beta * t) / (c + t)`
  
We also try different ways for learning the flow:
- Objectives:
  - Q param, parent-sum
  - V+pi param, no-sum
