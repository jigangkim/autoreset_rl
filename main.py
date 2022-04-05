import argparse
import datetime
import dateutil.tz
import glob
import inspect
import IPython
import numpy as np
import os
import pickle
import random
import signal
import tensorflow as tf
import time
from tqdm import tqdm
import sys

from algo.ddpg import DDPG
from algo.ddpg_rce import DDPGRecursiveClassifier
from algo.resetenv import ResetEnv
from utils import plot_metrics, get_envs, load_params, prompt_yes_or_no, log_tqdm


def train(args, params):
    '''
    DDPG forward agent and example-based DDPG reset agent
    '''
    # internal function
    def save_current_status_as_pickle(
            forward_agent, reset_agent, reset_env,
            latest_ckpt_step, curr_best_episode_reward, curr_best_reset_success, pkl_dir,
            max_to_keep=5
        ):
        list_of_pkl = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')), key=os.path.getmtime)
        if len(list_of_pkl) >= max_to_keep:
            os.remove(list_of_pkl[0])
        
        status = {
            'latest_ckpt_step': latest_ckpt_step,
            'curr_best_episode_reward': curr_best_episode_reward,
            'curr_best_reset_success': curr_best_reset_success,
            'random.state': random.getstate(),
            'np.random.state': np.random.get_state(),
            'env.np_random': getattr(reset_agent.env, 'np_random', np.random.RandomState()).__getstate__(),
            'DDPG': {
                'forward': {
                    'num_episodes': forward_agent.num_episodes,
                    'num_steps': forward_agent.num_steps,
                    'run_time': time.time() - forward_agent.init_time,
                    'buffer': forward_agent.buffer
                },
                'reset': {
                    'num_episodes': reset_agent.num_episodes,
                    'num_steps': reset_agent.num_steps,
                    'run_time': time.time() - reset_agent.init_time,
                    'buffer': reset_agent.buffer
                }
            },
            'algo': {
                '_total_resets': reset_env._total_resets,
                'run_time': time.time() - reset_env.init_time,
                'csv': {
                    'forward_num_episodes': reset_env.csv.forward_num_episodes,
                    'forward_num_steps': reset_env.csv.forward_num_steps,
                    'baseline_hard_reset': reset_env.csv.baseline_hard_reset,
                    'baseline_hard_reset_stepcounter': reset_env.csv.baseline_hard_reset_stepcounter
                }
            }
        }
        if type(reset_agent).__name__ in ['DDPGRecursiveClassifier']:
            status['DDPG']['reset']['classifier'] = {
                'goal_examples': reset_agent.goal_examples,
                'goal_examples_validation': reset_agent.goal_examples_validation
            }

        pickle.dump(status,
            open(os.path.join(pkl_dir, 'model_%d_%d_status.pkl' % (forward_agent.num_episodes, latest_ckpt_step)), 'wb')
        )

    # start tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # reset policy
    env, reset_env_fns = get_envs(params.env)
    env.seed(params.seed)
    goals = []
    for i in range(params.algo.reset.goal.number_of_examples + params.algo.reset.goal.number_of_validation_examples):
        obs = env.reset()
        goals.append(obs)
    goal_examples = goals[:params.algo.reset.goal.number_of_examples]
    goal_examples_validation = goals[-params.algo.reset.goal.number_of_validation_examples:]
    reset_agent = DDPGRecursiveClassifier(sess, env, params.algo.reset, name='reset', jobid=args.jobid, render=False, \
        goal_examples=goal_examples, goal_examples_validation=goal_examples_validation)
    # reset agent
    reset_env_fns = reset_agent.get_env_fns()
    reset_env = ResetEnv(env, reset_env_fns, reset_agent, params, jobid=args.jobid, logging=args.logging, render=False, verbose=True, evaluation=args.evaluation)
    # forward policy
    forward_agent = DDPG(sess, reset_env, params.algo.forward, name='forward', jobid=args.jobid, logging=args.logging, render=False, record=args.record)

    # checkpoint
    ckpt_dir = '%s/%s_%s_%s' %(os.path.expanduser(params.json.dir_name), params.json.file_name.prefix, args.jobid, params.json.file_name.postfix+'ckpt')
    saver_best = tf.train.Saver(var_list=None, max_to_keep=2)
    saver_latest = tf.train.Saver(var_list=None, max_to_keep=1)
    latest_ckpt_step = -float('inf')
    curr_best_episode_reward = -np.inf
    curr_best_reset_success = False

    # run algorithm
    pbar = tqdm(total=params.iteration_steps, desc='(pid %d jobid %s)' % (os.getpid(), args.jobid), file=sys.stdout)
    log_tqdm(pbar, args)
    forward_agent._params.logger.record_video_every_n_steps = int(params.iteration_steps/200)
    reset_agent._params.logger.record_video_every_n_steps = params.iteration_steps
    while forward_agent.num_steps + reset_agent.num_steps < params.iteration_steps:
        actor_idx = np.random.randint(forward_agent.params.actor.number_of_actors)
        episodes_stats, reset_episodes_stats = forward_agent.run_episodes(actor_idx=actor_idx, num_episodes=1, return_reset_episode_stats=True)
        pbar.n = forward_agent.num_steps + reset_agent.num_steps
        pbar.update(n=0)
        log_tqdm(pbar, args)

        if args.logging:
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(ckpt_dir+'/latest', exist_ok=True)
            if forward_agent.num_steps + reset_agent.num_steps > latest_ckpt_step + getattr(params, 'checkpoint_interval_steps', 10000):
                latest_ckpt_step = forward_agent.num_steps + reset_agent.num_steps
                # save new best checkpoint (sim envs w/ evaluation arg)
                if args.evaluation:
                    forward_agent.run_episodes(actor_idx=np.random.randint(forward_agent.params.actor.number_of_actors), num_episodes=1, eval_mode=True)
                    avg_reward, episode_length, reset_success = plot_metrics(reset_env.csv.filepath_evaluation, os.path.join(os.path.expanduser(params.json.dir_name), params.json.file_name.full), eval_mode=True)
                    try: episode_reward = avg_reward*episode_length
                    except: episode_reward = -np.inf
                    reset_success = bool(reset_success)
                    # new best (higher episode reward -> reset success)
                    if (reset_success > curr_best_reset_success) or (episode_reward >= curr_best_episode_reward and reset_success==curr_best_reset_success):
                        curr_best_episode_reward = episode_reward
                        curr_best_reset_success = reset_success
                        print('curr_best_episode_reward is %f and curr_best_reset_success is %s'%(curr_best_episode_reward, curr_best_reset_success))
                        ckpt_file = 'model_%d_%d' % (forward_agent.num_episodes, latest_ckpt_step)
                        print('saving tensorflow checkpoint(best)...')
                        saver_best.save(sess, '%s/%s' % (ckpt_dir, ckpt_file))
                        print('saving status...')
                        save_current_status_as_pickle(forward_agent, reset_agent, reset_env, latest_ckpt_step, curr_best_episode_reward, curr_best_reset_success, pkl_dir=ckpt_dir, max_to_keep=min(saver_best._max_to_keep,1))
                # save latest checkpoint
                print('latest_ckpt_step is %d'%(latest_ckpt_step))
                ckpt_file = 'model_%d_%d' % (forward_agent.num_episodes, latest_ckpt_step)
                print('saving tensorflow checkpoint(latest)...')
                saver_latest.save(sess, '%s/%s' % (ckpt_dir+'/latest', ckpt_file))
                print('saving status...')
                save_current_status_as_pickle(forward_agent, reset_agent, reset_env, latest_ckpt_step, curr_best_episode_reward, curr_best_reset_success, pkl_dir=ckpt_dir+'/latest', max_to_keep=min(saver_latest._max_to_keep,1))
                # plotter
                print('saving png...')
                avg_reward, episode_length, reset_success = plot_metrics(reset_env.csv.filepath, os.path.join(os.path.expanduser(params.json.dir_name), params.json.file_name.full))

        global resume, terminate, old_print, new_print
        if terminate:
            inspect.builtins.print = old_print
            resume = prompt_yes_or_no('Resume?', default_response='n') # always respond no (False)
            if resume:
                IPython.embed()
                terminate = False
            inspect.builtins.print = new_print
        if terminate:
            print('resume prompt response is False! terminating...')
            break
    pbar.close()
    log_tqdm(pbar, args, remove=True)

    if args.logging:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(ckpt_dir+'/latest', exist_ok=True)
        latest_ckpt_step = forward_agent.num_steps + reset_agent.num_steps
        # save new best checkpoint (sim envs w/ evaluation arg)
        if args.evaluation:
            forward_agent.run_episodes(actor_idx=np.random.randint(forward_agent.params.actor.number_of_actors), num_episodes=1, eval_mode=True)
            avg_reward, episode_length, reset_success = plot_metrics(reset_env.csv.filepath_evaluation, os.path.join(os.path.expanduser(params.json.dir_name), params.json.file_name.full), eval_mode=True)
            try: episode_reward = avg_reward*episode_length
            except: episode_reward = -np.inf
            reset_success = bool(reset_success)
            # new best (higher episode reward -> reset success)
            if (reset_success > curr_best_reset_success) or (episode_reward >= curr_best_episode_reward and reset_success==curr_best_reset_success):
                curr_best_episode_reward = episode_reward
                curr_best_reset_success = reset_success
                print('curr_best_episode_reward is %f and curr_best_reset_success is %s'%(curr_best_episode_reward, curr_best_reset_success))
                ckpt_file = 'model_%d_%d' % (forward_agent.num_episodes, latest_ckpt_step)
                print('saving tensorflow checkpoint(best)...')
                saver_best.save(sess, '%s/%s' % (ckpt_dir, ckpt_file))
                print('saving status...')
                save_current_status_as_pickle(forward_agent, reset_agent, reset_env, latest_ckpt_step, curr_best_episode_reward, curr_best_reset_success, pkl_dir=ckpt_dir, max_to_keep=min(saver_best._max_to_keep,1))
        # save latest checkpoint
        print('latest_ckpt_step is %d'%(latest_ckpt_step))
        ckpt_file = 'model_%d_%d' % (forward_agent.num_episodes, latest_ckpt_step)
        print('saving tensorflow checkpoint(latest)...')
        saver_latest.save(sess, '%s/%s' % (ckpt_dir+'/latest', ckpt_file))
        print('saving status...')
        save_current_status_as_pickle(forward_agent, reset_agent, reset_env, latest_ckpt_step, curr_best_episode_reward, curr_best_reset_success, pkl_dir=ckpt_dir+'/latest', max_to_keep=min(saver_latest._max_to_keep,1))
        # plotter
        print('saving png...')
        avg_reward, episode_length, reset_success = plot_metrics(reset_env.csv.filepath, os.path.join(os.path.expanduser(params.json.dir_name), params.json.file_name.full))
    
    print('done!')
    return 0


def main(args):
    # Catch SIGINT (ctrl-c)
    # https://stackoverflow.com/questions/24426451/how-to-terminate-loop-gracefully-when-ctrlc-was-pressed-in-python
    def signal_handling(signum, frame):
        global terminate
        terminate = True
        print('pausing... Please wait!')
    signal.signal(signal.SIGINT, signal_handling)

    global resume, terminate
    resume = True
    terminate = False
    
    # assert
    if args.record: assert args.logging, 'logging is required for record'
    if args.evaluation: assert args.logging, 'logging is required for evaluation'

    args.jobid = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
    params = load_params(args)

    global old_print, new_print
    # Workaround for tqdm.write() with print()
    # https://stackoverflow.com/questions/36986929/redirect-print-command-in-python-script-through-tqdm-write/37243211
    old_print = print # store builtin print
    class custom_print(object):
        def __init__(self, path_to_file, logging=False):
            self.logging = logging
            if self.logging:
                self.fp = open(path_to_file, 'a', buffering=1) # line buffering
                sys.stderr = self.fp # redirect stderr to log file
        
        def __call__(self, *args, **kwargs):
            assert len(kwargs) == 0
            list_of_strings = [str(elem) for elem in args]
            tqdm.write(' '.join(list_of_strings))
            if self.logging:
                tqdm.write(' '.join(list_of_strings), file=self.fp)
        
        def __del__(self):
            if self.logging: self.fp.close()

    assert os.path.isdir(os.path.expanduser(params.json.dir_name))
    filename = '%s_%s_%sparams.log'%(params.json.file_name.prefix, args.jobid, params.json.file_name.postfix)
    new_print = custom_print(os.path.join(os.path.expanduser(params.json.dir_name), filename), logging=args.logging)
    inspect.builtins.print = new_print
    
    random.seed(params.seed)
    np.random.seed(params.seed)

    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(params.seed)
        train(args, params)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Implementation for RA-L submission "Automating Reinforcement Learning with Example-based Resets" by Jigang Kim')
    parser.add_argument('--config_dir', type=str, default='./experiment_configs/peg-insertion_insert.json', 
        help='Directory of the config file.'
    )
    parser.add_argument('--absolute_path', action='store_true',
        help='Flag for path type.'
    )
    parser.add_argument('--logging', action='store_true',
        help='Flag for logging statistics.'
    )
    parser.add_argument('--record', action='store_true',
        help='Flag for recording videos. (logging flag is required)'
    )
    parser.add_argument('--evaluation', action='store_true',
        help='Flag for running evaluation loop. (logging flag is required)'
    )
    parser.add_argument('--checkpoint_latest', action='store_true',
        help='Checkpoint latest instead of best (default)'
    )
    args = parser.parse_args()
    
    main(args)