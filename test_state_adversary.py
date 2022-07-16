import time
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
import numpy as np
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

sess = tf.Session()

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
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action, model = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action, model = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action, model


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    # sess = tf.Session()
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
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action, model


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action, model


def run_policy(logger, env, get_action, model, epsilon=0.01,
               max_ep_len=None, num_episodes=100, algo=None):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    if algo == 'trpo':
        # loss = tf.norm(model['pi'], ord='euclidean')
        # calculate_grad = tf.gradients(loss, model['x'])[0]
        calculate_grad = tf.gradients(tf.norm(model['pi'], ord='euclidean'), model['x'])[0]
        calculate_grad = calculate_grad / tf.norm(calculate_grad, ord='euclidean')

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    average_reward = 0.0
    last_time = time.time()
    while n < num_episodes:
        ## FGSM attack ##
        if algo != "trpo":
            old_o = torch.tensor(o, dtype=torch.float32)
            old_o = old_o.clone().requires_grad_()

            a = model.pi.mu_net(old_o)
            loss = torch.norm(a)
            loss.backward()

            # v = model.v(old_o)
            # v.backward()

            if torch.norm(old_o.grad.data) == 0:
                grad = np.array(old_o.grad.data)
            else:
                grad = np.array(old_o.grad.data/torch.norm(old_o.grad.data))
            o = o - epsilon * grad
        else:
            # print(model)
            # {'x': <tf.Tensor 'Placeholder:0' shape=(?, 17) dtype=float32>,
            #  'pi': <tf.Tensor 'pi/add:0' shape=(?, 6) dtype=float32>,
            #  'v': <tf.Tensor 'v/Squeeze:0' shape=(?,) dtype=float32>}

            # action = sess.run(model['pi'], feed_dict={model['x']: o_input[None,:]})[0]
            # print(model['x'].shape)
            # model['x'] = tf.placeholder(tf.float32, shape=(None, 17))
            # sess = tf.Session()
            # loss = tf.norm(model['pi'], ord='euclidean')
            # calculate_grad = tf.gradients(loss, model['x'])[0]
            # _, grad = sess.run([loss, calculate_grad],
            #                feed_dict={model['x']: o[None, :]})
            grad = sess.run(calculate_grad,
                            feed_dict={model['x']: o[None, :]})
            # # grad = grad[0]
            # grad_norm = sess.run(tf.norm(grad, ord='euclidean'))
            # grad = grad / grad_norm
            # # print(grad)
            # # print(0, o)
            # # print(o.shape)
            # # print(grad.shape)
            o = o - epsilon * grad[0]
            # # print(o.shape) # (17, )
            # # print(1, o)
        ## FGSM attack ##

        ## random noise ##
        # old_o = torch.tensor(o, dtype=torch.float32)
        # noises = np.array(torch.randn_like(old_o) * epsilon)
        # o = o + noises
        ## random noise ##

        a = get_action(o)
        # if algo != 'trpo':
        #     a = get_action(o)
        # else:
        #     sess = tf.Session()
        #     a = sess.run(model['pi'], feed_dict={model['x']: o[None, :]})[0]
        #     sess.close()

        # print(old_o.grad.data)
        # print(old_o)
        # print(grad)
        # print(epsilon)
        # print(o)
        # print(a)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # print('time', time.time() - last_time)
        # last_time = time.time()
        # print(ep_len)

        if d or (ep_len == max_ep_len):
            average_reward += ep_ret
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('time', time.time() - last_time)
            last_time = time.time()
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    average_reward = average_reward / num_episodes
    return average_reward


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('fpath', type=str)
    parser.add_argument('--task', type=str, default='Ant')
    parser.add_argument('--version', type=str, default='-v3')
    parser.add_argument('--algos', default='ppo sppo')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon_low', type=float, default=0.0)
    parser.add_argument('--epsilon_upp', type=float, default=1.0)
    parser.add_argument('--epsilon_num', type=int, default=10)
    args = parser.parse_args()

    logger = EpochLogger()
    algos = args.algos.split(" ")
    print("algos: ", algos)

    # file_handle = open(args.task+'_'+args.base_algo+'_adversarial.txt', mode='w')

    epsilon_array = np.linspace(args.epsilon_low, args.epsilon_upp,
                                args.epsilon_num)

    plt.figure()
    sns.set(style="darkgrid", font_scale=1.5)

    f = open('results/state_adversary/' + args.task + '_state_adversary.txt', "w")
    for i in epsilon_array:
        print(i, end=" ", file=f)
    print('', file=f)
    for k in range(len(algos)):
        algo = algos[k]
        algo_rewards = []
        for i in range(10):
            seed = str(i)
            fpath = "src/data/" + args.task + "/" + algo + "/" + algo \
                    + "-seed" + seed + "/" + args.task + "/" + algo \
                    + "/" + algo + "-seed" + seed + "_s" + seed
            env, get_action, model = load_policy_and_env(fpath,
                                                         args.itr if args.itr >= 0 else 'last',
                                                         args.deterministic)
            rewards = []
            for epsi in epsilon_array:
                print('task:', args.task, 'algo:', algo)
                print('model:', i, 'epsilon:', epsi)
                average_reward = run_policy(logger, env, get_action, model,
                                            epsi, args.len, args.episodes, algo)
                print(average_reward)
                rewards.append(average_reward)
            algo_rewards.append(rewards)

        algo_rewards = np.array(algo_rewards)
        algo_mean = np.mean(algo_rewards, axis=0)
        algo_std = np.std(algo_rewards, axis=0)

        # draw the mean
        plt.plot(epsilon_array, algo_mean, color=COLORS[k], label=algo)
        plt.fill_between(epsilon_array, algo_mean - algo_std, algo_mean + algo_std,
                         color=COLORS[k], alpha=.4)
        print(algo, file=f)
        for i in algo_mean:
            print(i, end=" ", file=f)
        print('', file=f)
        for i in algo_std:
            print(i, end=" ", file=f)
        print('', file=f)

    # file_handle.write(args.task + ' ' + args.base_algo)
    # file_handle.write(base_algo_rewards)

    # file_handle.close()

    plt.title(args.task + args.version)
    plt.xlabel('Bound of state noises')
    plt.ylabel('Episode Reward')
    # loc is the position of the sign, 2 represent the top left corner
    plt.legend(loc=0).set_draggable(True)

    # plt.show()
    plt.savefig('./img/' + args.task + '_state.png')
