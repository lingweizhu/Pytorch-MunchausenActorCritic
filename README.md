# Munchausen Actor Critic using PyTorch
The code was adapted from [Toshiki Watanabe](https://github.com/ku2482/sac-discrete.pytorch), please check the original page for update and credit.

The base algorithm is SAC discrete [1] for my own research purpose, might add the continuous version later.


## Munchausen Reinforcement Learning [2]
The original paper introduces Munchausen trick only on top of DQN, here I tried to extend it to actor critic style, this requires exploration on the suitable policy loss.


## References
[[1]](https://arxiv.org/abs/1910.07207) Christodoulou, Petros. "Soft Actor-Critic for Discrete Action Settings." arXiv preprint arXiv:1910.07207 (2019).

[[2]](https://arxiv.org/abs/2007.14430) Nino Vieillard, Olivier Pietquin, Matthieu Geist, "Munchausen Reinforcement Learning." NeurIPS (2020).

