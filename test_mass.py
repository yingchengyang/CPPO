import matplotlib as mpl

mpl.use('Agg')

import gym
import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph


# colors for plot
COLORS = ([
    # deepmind style
    '#0072B2',
    '#009E73',
    '#D55E00',
    '#CC79A7',
    '#F0E442',
    # built-in color
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
    'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
    'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue',
    # personal color
    '#313695',  # DARK BLUE
    '#74add1',  # LIGHT BLUE
    '#4daf4a',  # GREEN
    '#f46d43',  # ORANGE
    '#d73027',  # RED
    '#984ea3',  # PURPLE
    '#f781bf',  # PINK
    '#ffc832',  # YELLOW
    '#000000',  # BLACK
])


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x) > 8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]

        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(args, algos, max_ep_len=None, num_episodes=1):
    if args.mass_lower_bound == -1:
        print("Please input the mass.")
        return
    mass_np = np.linspace(args.mass_lower_bound, args.mass_upper_bound,
                          args.mass_number)
    ee = gym.make(args.task + '-v3')
    old_mass = ee.model.body_mass
    num = len(old_mass)
    print(old_mass)

    all_mass_results = []
    logger = EpochLogger()

    f = open('results/' + args.task + '_mass.txt', "w")
    for i in mass_np:
        print(i, end=" ", file=f)
    print('', file=f)

    plt.figure()
    sns.set(style="darkgrid", font_scale=1.5)

    for k in range(len(algos)):
        algo = algos[k]
        mass_results = []
        for i in range(10):
            seed = i
            seed = str(seed)
            print("This is algo ", algo, "seed ", seed)

            fpath = "src/data/" + args.task + "/" + algo + "/" + algo \
                    + "-seed" + seed + "/" + args.task + "/" + algo \
                    + "/" + algo + "-seed" + seed + "_s" + seed
            print('The path now is: ', fpath)
            env, get_action = load_policy_and_env(fpath,
                                                  args.itr if args.itr >= 0 else 'last',
                                                  args.deterministic)
            assert env is not None, \
                "Environment not found!\n\n It looks like the environment wasn't saved, " + \
                "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
                "page on Experiment Outputs for how to handle this situation."

            mass_result = []
            for mass in mass_np:
                print('task:', args.task, 'algo:', algo)
                print('model:', i, 'mass:', mass)
                for j in range(num):
                    env.model.body_mass[j] = old_mass[j] * mass / old_mass[1]
                # print("mass is: ", env.model.body_mass)
                # env.model.body_mass[1] = mass
                print("mass is: ", env.model.body_mass)

                # logger = EpochLogger()
                o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
                average_ret = 0.0
                while n < num_episodes:
                    # if render:
                    #     env.render()
                    #     time.sleep(1e-3)

                    a = get_action(o)
                    o, r, d, _ = env.step(a)
                    ep_ret += r
                    ep_len += 1

                    if d or (ep_len == max_ep_len):
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                        average_ret += ep_ret
                        print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
                        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                        n += 1

                average_ret = average_ret / num_episodes
                mass_result.append(average_ret)

                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.dump_tabular()
            mass_results.append(mass_result)

        mass_results = np.array(mass_results)
        mass_mean = np.mean(mass_results, axis=0)
        mass_std = np.std(mass_results, axis=0)
        # draw the mean
        plt.plot(mass_np, mass_mean, color=COLORS[k],
                 label=algo)
        # draw the mean+/-std with light color
        plt.fill_between(mass_np, mass_mean - mass_std, mass_mean + mass_std,
                         color=COLORS[k], alpha=.4)

        print(algo, file=f)
        for i in mass_mean:
            print(i, end=" ", file=f)
        print('', file=f)
        for i in mass_std:
            print(i, end=" ", file=f)
        print('', file=f)

    # print(vpg_mass_results)
    # print(cvarvpg_mass_results)


    plt.title(args.task + "-v3")
    plt.xlabel('Mass of torso')
    # plt.xlabel('Mass of torso')
    # plt.ylabel('Episode Reward', fontsize=22)
    plt.ylabel('Performance', fontsize=22)
    # loc is the position of the sign, 2 represent the top left corner
    plt.legend(loc=2).set_draggable(True)

    # plt.show()
    plt.savefig('./img/' + args.task + '_mass.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('fpath', type=str)
    parser.add_argument('--task', default='Swimmer')
    parser.add_argument('--algos', default='ppo sppo')
    parser.add_argument('--base_algo', default='vpg')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--mass_lower_bound', type=float, default=-1.0)
    parser.add_argument('--mass_upper_bound', type=float, default=-1.0)
    parser.add_argument('--mass_number', type=int, default=1)
    args = parser.parse_args()

    algos = args.algos.split(" ")
    print("algos: ", algos)
    run_policy(args, algos, args.len, args.episodes)