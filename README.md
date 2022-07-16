# CPPO

- This is the official implementation for [Towards Safe Reinforcement Learning via Constraining Conditional Value at Risk](https://arxiv.org/abs/2206.04436) (Accepted in IJCAI 2022).

- The training code is based on [Spinning Up](https://github.com/openai/spinningup).

## Usage
- The training code is in the folder '/src'.

- These methods, including baselines and our methods, are based on [Spinning Up](https://github.com/openai/spinningup) (we delete unnecessary files to make the code clearer)

- You first should install Spinning Up by

```
cd src
pip install -e .
```

- Then you can run the training code like

```
python -m spinup.run vpg --hid "[64,32]" --env Ant-v3 --exp_name Ant/vpg/vpg-seed0 --epochs 750 --seed 0
python -m spinup.run trpo --hid "[64,32]" --env Ant-v3 --exp_name Ant/trpo/trpo-seed0 --epochs 750 --seed 0
python -m spinup.run ppo --hid "[64,32]" --env Ant-v3 --exp_name Ant/ppo/ppo-seed0 --epochs 750 --seed 0
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
