# CPPO

[![arXiv](https://img.shields.io/badge/arXiv-2206.04436-b31b1b.svg)](https://arxiv.org/abs/2206.04436)

- This is the official implementation for [Towards Safe Reinforcement Learning via Constraining Conditional Value at Risk](https://www.ijcai.org/proceedings/2022/0510.pdf) (Accepted in IJCAI 2022).

- The training code is based on [Spinning Up](https://github.com/openai/spinningup).

## Usage
- The training code is in the folder '/src'.

- These methods, including baselines and our methods, are based on [Spinning Up](https://github.com/openai/spinningup) (we delete unnecessary files to make the code clearer)

- You first should install Spinning Up by

```
cd src
pip install -e .
```

- Then you can train agents of baselines, including VPG, TRPO, PPO, and PG-CMDP, by the training code like

```
python -m spinup.run vpg --hid "[64,32]" --env Walker2d-v3 --exp_name Walker2d/vpg/vpg-seed0 --epochs 750 --seed 0
python -m spinup.run trpo --hid "[64,32]" --env Walker2d-v3 --exp_name Walker2d/trpo/trpo-seed0 --epochs 750 --seed 0
python -m spinup.run ppo --hid "[64,32]" --env Walker2d-v3 --exp_name Walker2d/ppo/ppo-seed0 --epochs 750 --seed 0
python -m spinup.run pg_cmdp --hid "[64,32]" --env Walker2d-v3 --exp_name Walker2d/pg_cmdp/pg_cmdp-seed0 --epochs 750 --seed 0 --delay 0.8 --nu_delay 0.8
```

- For PG-CMDP, you can also adjust parameters like --delay, --nu_delay and so on.

- You can train agents of CPPO for all five environments reported in the paper with the training code as below

```
python -m spinup.run cppo --hid "[64,32]" --env Ant-v3 --exp_name Ant/cppo/cppo-seed0 --epochs 750 --seed 0 --beta 2800 --nu_start 10.0 --gamma 0.99 --nu_delay 0.2 --delay 0.0024 --cvar_clip_ratio 0.018
python -m spinup.run cppo --hid "[64,32]" --env HalfCheetah-v3 --exp_name HalfCheetah/cvarppo/cppo-seed0 --epochs 750 --seed 0 --beta 2500 --nu_start 10.0 --gamma 0.99 --nu_delay 0.3 --delay 0.0002 --cvar_clip_ratio 0.01
python -m spinup.run cppo --hid "[64,32]" --env Hopper-v3 --exp_name Hopper/cvarppo/cppo-seed0 --epochs 750 --seed 0 --beta 2500 --nu_start 10.0 --gamma 0.999 --nu_delay 0.3 --delay 0.002 --cvar_clip_ratio 0.027
python -m spinup.run cppo --hid "[64,32]" --env Swimmer-v3 --exp_name Swimmer/cvarppo/cppo-seed0 --epochs 750 --seed 0 --beta 122 --nu_start -20.0 --gamma 0.999 --nu_delay 0.3 --delay 0.002 --cvar_clip_ratio 0.03
python -m spinup.run cppo --hid "[64,32]" --env Walker2d-v3 --exp_name Walker2d/cvarppo/cppo-seed0 --epochs 750 --seed 0 --beta 2500 --nu_start 10.0 --gamma 0.99 --nu_delay 0.3 --delay 0.0018 --cvar_clip_ratio 0.01
```

- For CPPO, you can also adjust parameters like --beta, --nu_start, --nu_delay, --delay, --cvar_clip_ratio and so on.

- Evaluate the performance under transition disturbance (changing mass)
```
python test_mass.py --task Walker2d --algos "vpg trpo ppo cvarvpg cppo" --mass_lower_bound 1.0 --mass_upper_bound 7.0 --mass_number 100 --episodes 5
```

- Evaluate the performance under observation disturbance (random noises)
```
python test_state.py --task Walker2d --algos "vpg trpo ppo pg_cmdp cppo" --epsilon_low 0.0 --epsilon_upp 0.4 --epsilon_num 100 --episodes 5
```

- Evaluate the performance under observation disturbance (adversarial noises)
```
python test_state_adversary.py --task Walker2d --algos "vpg trpo ppo pg_cmdp cppo" --epsilon_low 0.0 --epsilon_upp 0.2 --epsilon_num 100 --episodes 5
```


## Citation

If you find CPPO helpful, please cite our paper.

```
@inproceedings{
    ying2022towards,
    title={Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk},
    author={Ying, Chengyang and Zhou, Xinning and Su, Hang and Yan, Dong and Chen, Ning and Zhu, Jun},
    booktitle={International Joint Conference on Artificial Intelligence},
    year={2022},
    url={https://arxiv.org/abs/2206.04436}
}
```
