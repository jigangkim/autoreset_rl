import csv
import datetime
from inputimeout import inputimeout, TimeoutOccurred
import json
import matplotlib
matplotlib.use('Agg') # enable headless plotting
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pkg_resources
import pickle
import shutil
from types import SimpleNamespace

mujoco_py_version = pkg_resources.get_distribution('mujoco-py').version
mujoco_py_import_fail = False
try:
    import mujoco_py
except:
    mujoco_py_import_fail = True
    import warnings
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn('mujoco_py import failed (possibly due to missing or invalid activation key)!\r\n')
if mujoco_py_version == '0.5.7' and not mujoco_py_import_fail:
    from envs.cliff_envs import CliffCheetahEnv, CliffWalkerEnv
    from envs.peg_insertion import PegInsertionEnv
else:
    ValueError('Invalid mujoco_py!')


def plot_metrics(csv_path, config_path, plot_name=None, output_dir=None, eval_mode=False):
    '''
    '''
    assert os.path.exists(csv_path)
    assert os.path.exists(config_path)
    if plot_name == None:
        plot_name = os.path.splitext(os.path.basename(csv_path))[0]
    if output_dir == None:
        output_dir = os.path.dirname(config_path)
    
    # csv_path
    header = [
        'avg reward',
        'episode',
        'total hard resets',
        'baseline hard reset',
        'state before soft reset',
        'state after soft reset',
        'total steps',
        'reset total steps',
        'soft reset success?',
        'in a ditch?',
        'spillage?',
        'episode length'
    ]
    episodes = []
    rewards = []
    resets = []
    baseline_resets = []
    x_before = []
    x_after = []
    steps = []
    reset_successful = []
    in_a_ditch = []
    spillage = []
    episode_length = []
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                header_colnum = [row.index(h) if h in row else None for h in header]
            else:
                try: rewards.append(float(row[header_colnum[0]]))
                except: rewards.append(None)
                try: episodes.append(float(row[header_colnum[1]]))
                except: episodes.append(None)
                try: resets.append(float(row[header_colnum[2]]))
                except: resets.append(None)
                try: baseline_resets.append(float(row[header_colnum[3]]))
                except: baseline_resets.append(None)
                try: x_before.append(float(row[header_colnum[4]]))
                except: x_before.append(None)
                try: x_after.append(float(row[header_colnum[5]]))
                except: x_after.append(None)
                try: steps.append(float(row[header_colnum[6]]) + float(row[header_colnum[7]]))
                except: steps.append(None)
                try: reset_successful.append(int(float(row[header_colnum[8]])))
                except: reset_successful.append(None)
                try: in_a_ditch.append(int(float(row[header_colnum[9]])))
                except: in_a_ditch.append(0)
                try: spillage.append(int(float(row[header_colnum[10]])))
                except: spillage.append(0)
                try: episode_length.append(int(float(row[header_colnum[11]])))
                except: episode_length.append(None)
            line_count += 1
    baseline_resets_original = episodes

    # return values
    latest_forward_average_reward = rewards[-1]
    latest_forward_episode_length = episode_length[-1]
    latest_reset_successful = reset_successful[-1]

    # Plot the data
    fig = plt.figure(figsize=(8*2, 6))
    # subplot 1 (reset rewards and hard resets)
    ax1 = plt.subplot(121)
    ax2 = ax1.twinx()
    ax1.plot(steps, rewards, 'g.')
    ax2.plot(steps, resets, 'b-')
    ax2.plot(steps, baseline_resets_original, color='grey')
    ax2.plot(steps, baseline_resets, 'k-')
    ax2.plot(steps, np.cumsum(in_a_ditch), 'b--')
    ax2.plot(steps, np.cumsum(spillage), 'b--')
    ax1.set_ylabel('average step reward', color='g', fontsize=20)
    ax1.tick_params('y', colors='g')
    ax2.set_ylabel('num. resets', color='b', fontsize=20)
    ax2.tick_params('y', colors='b')
    ax1.set_xlabel('num. steps', fontsize=20)
    # subplot 2 (state before-after and hard resets)
    ax1 = plt.subplot(122)
    ax2 = ax1.twinx()
    ax1.plot(steps, np.array(x_before) - np.array(x_after), 'g.')
    ax1.axhline(y=np.mean(x_before) - np.mean(x_after), color='g', linestyle='--')
    ax2.plot(steps, resets, 'b-')
    ax2.plot(steps, baseline_resets_original, color='grey')
    ax2.plot(steps, baseline_resets, 'k-')
    ax2.plot(steps, np.cumsum(in_a_ditch), 'b--')
    ax2.plot(steps, np.cumsum(spillage), 'b--')
    ax1.set_ylabel('state2log (before - after)', color='g', fontsize=20)
    ax1.tick_params('y', colors='g')
    ax2.set_ylabel('num. resets', color='b', fontsize=20)
    ax2.tick_params('y', colors='b')
    ax1.set_xlabel('num. steps', fontsize=20)
    plt.tight_layout(pad=2.0)
    # save
    plt.savefig(os.path.join(output_dir, '%s.png'%(plot_name)))

    # close figure and clear memory
    plt.close('all')
    return latest_forward_average_reward, latest_forward_episode_length, latest_reset_successful


def load_params(args):
    # load params (configs)
    if args.absolute_path:
        path_to_config = args.config_dir
    else:
        path_to_config = str((Path(__file__).parent / args.config_dir).resolve())        
    with open(path_to_config, 'r') as f:
        # read json as namespace
        params = json.load(f, object_hook=lambda d : SimpleNamespace(**d))
    if args.logging:
        # save a copy of configs for future reference
        assert os.path.isdir(os.path.expanduser(params.json.dir_name))
        filename = '%s_%s_%sparams.json'%(params.json.file_name.prefix, args.jobid, params.json.file_name.postfix)
        shutil.copy2(path_to_config, os.path.join(os.path.expanduser(params.json.dir_name), filename))
        params.json.file_name.full = filename
    print('loaded json configuration file')

    return params


def get_envs(env_params):
    '''
    '''
    list_of_available_envs = ['cliff-cheetah', 'cliff-walker', 'peg-insertion(insert)', 'peg-insertion(remove)']

    env_name = env_params.name
    if env_name not in list_of_available_envs: raise ValueError('Unavailable environment: %s'%(env_name))

    mujoco_py_version = pkg_resources.get_distribution('mujoco-py').version
    assert mujoco_py_version == '0.5.7', 'Invalid mujoco version for environment: %s'%(env_name)
    
    if 'cliff-cheetah' in env_name:
        if env_name == 'cliff-cheetah': env = CliffCheetahEnv()
        else: raise ValueError('Unknown option for CliffCheetahEnv')
    elif 'cliff-walker' in env_name:
        if env_name == 'cliff-walker': env = CliffWalkerEnv()
        else: raise ValueError('Unknown option for CliffWalkerEnv')
    elif 'peg-insertion' in env_name:
        if env_name == 'peg-insertion(insert)': env = PegInsertionEnv(task='insert', sparse=False)
        elif env_name == 'peg-insertion(remove)': env = PegInsertionEnv(task='remove', sparse=False)
        else: raise ValueError('Unknown option for PegInsertionEnv')
    else:
        raise ValueError('Unknown environment')

    return env, {'reset_reward_fn': lambda s, a: env._get_rewards(s,a)[1], 'reset_done_fn': env._get_reset_done}


def _get_tqdm_file_path(args):
    path_to_config = args.config_dir if args.absolute_path else str((Path(__file__).parent / args.config_dir).resolve())
    return os.path.join(os.path.dirname(path_to_config), '.tmp', os.path.basename(args.config_dir) + '.tqdm')


def log_tqdm(tqdm_obj, args, remove=False):
    tqdm_filepath = _get_tqdm_file_path(args)

    if remove:
        try: os.remove(tqdm_filepath)
        except OSError: pass
    else:
        d = tqdm_obj.format_dict
        tqdm_stat = ()
        tqdm_stat += (os.environ.get('CUDA_VISIBLE_DEVICES'),)
        tqdm_stat += (os.getpid(),)
        tqdm_stat += (int(d['n']/d['total']*100),)
        tqdm_stat += (d['n'],)
        tqdm_stat += (d['total'],)
        tqdm_stat += (str(datetime.timedelta(seconds=int(d['elapsed']))),)
        try: tqdm_stat += (str(datetime.timedelta(seconds=int((d['total'] - d['n'])/d['rate']))),)
        except: tqdm_stat += ('?',)
        try: tqdm_stat += (round(d['rate'],2),)
        except: tqdm_stat += ('?',)
        os.makedirs(os.path.dirname(tqdm_filepath), exist_ok=True)
        pickle.dump(tqdm_stat, open(tqdm_filepath, 'wb'))


def prompt_yes_or_no(query, timed=False, timeout=30, default_response=None):
    while True:
        if timed: # inputimeout does NOT work when called from a subprocess!
            assert default_response is not None
            try:
                response = inputimeout(prompt=query + ' (timeout in %ds) [Y/n] '%(timeout), timeout=timeout).lower()
            except TimeoutOccurred:
                response = default_response
        else:
            if default_response is None:
                response = input(query + ' [Y/n] ').lower()
            else:
                response = default_response 
        if response in {'y', 'yes'}:
            return True
        elif response in {'n', 'no'}:
            return False
        else:
            print('Invalid response!\n')