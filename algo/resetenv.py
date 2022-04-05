from copy import deepcopy
import csv
import numpy as np
import os
import sys
import tensorflow as tf
import time
from tqdm import tqdm
from types import SimpleNamespace

from gym import Wrapper
from .ddpg_utils import state_to_log


class ResetEnv(Wrapper):
    def __init__(self, env, reset_env_fns, reset_agent, params, name='', jobid='', logging=False, render=False, verbose=False, evaluation=False):
        '''
        Reset agent with example-based control and reset trigger

        args:
            :arg env: Environment implementing the Gym API
            :type env:
            :arg reset_env_fns: Reset reward & done functions
            :type reset_env_fns: Dictionary object
            :arg reset_agent: An agent from scratch (DDPG)
            :type reset_agent: 
            :arg params: Parameters for ResetEnv
            :type params: SimpleNamespace object
            :arg name: Name for ResetEnv object (optional)
            :type name: String object
            :arg jobid:
            :type jobid:
            :arg logging:
            :type logging:

        returns:

        '''
        # inheritance
        super(ResetEnv, self).__init__(env) # initialize gym Wrapper (self.env <- env)
        self._obs = self.env.reset() # reset env

        # input args
        self._reset_reward_fn = reset_env_fns['reset_reward_fn']
        self._reset_done_fn = reset_env_fns['reset_done_fn']
        self._reset_done_env_fn = reset_env_fns['reset_done_env_fn']
        
        self._reset_agent = reset_agent
        self.params = params.algo
        self._params = params
        self.name = name
        self.jobid = jobid
        self.logging = logging
        self._reset_agent.logging = self.logging
        self.isrender = render
        self.verbose = verbose
        self.evaluation = evaluation

        # additional params
        self._max_episode_steps = self.params.reset.ddpg.max_episode_length
        self._min_episode_steps = 5
        assert self._max_episode_steps > self._min_episode_steps
        self._p_threshold = self.params.p_threshold
        self._reset_attempts = self.params.max_reset_attempts

        self._prob_reset_criterion = 0.0
        self._prob_reset_ensemble = None
        self._prob_reset_trigger_ema = 0.0

        # set up internal structures for logging metrics
        self._total_resets = 0  # Total resets taken during training
        self._episode_rewards = []  # Rewards for the current episode

        # additional logging
        self.init_time = time.time()
        self.dt = self.env.dt

        # initialize logger
        self._init_logger()
        
        
    def _init_logger(self):
        self.csv = SimpleNamespace()
        ## 1. key stats
        self.csv.forward_num_episodes = 0 # Episode step of the outer agent
        self.csv.reset_triggered = None # 1 if reset triggered (p < p_threshold), 0 if explicitly requested by the forward agent
        self.csv.reset_done = None # reset done
        self.csv.reset_done_ground_truth = None # reset done (ground truth)
        self.csv.baseline_hard_reset = 0 # baseline hard resets
        self.csv.baseline_hard_reset_stepcounter = 0
        self.csv.avg_reward = None
        self.csv.avg_reset_reward = None # average reset reward
        self.csv.discounted_reset_return = None # discounted sum of reset rewards
        ## 2. detailed stats
        self.csv.rreset_before_reset = None # reset reward prior to soft reset attempt (using action [0, 0, ...])
        self.csv.rreset_after_reset = None # reset reward after soft reset attempt (using action [0, 0, ...])
        self.csv.probreset_before_reset_criterion = None # reset prob value prior to soft reset attempt (using action from step(a) if available, if not, from reset policy)
        self.csv.probreset_before_reset_mu = None
        self.csv.probreset_before_reset_std = None
        self.csv.probreset_before_min = None
        self.csv.probreset_before_max = None
        self.csv.probreset_after_reset_criterion = None # reset prob value after soft reset attempt (using action from reset policy)
        self.csv.probreset_after_reset_mu = None
        self.csv.probreset_after_reset_std = None
        self.csv.probreset_after_min = None
        self.csv.probreset_after_max = None
        self.csv.state_before_reset = [] # env stat prior to soft reset attempt
        self.csv.state_after_reset = [] # env stat after soft reset attempt
        self.csv.reset_msg = None # message after running the reset policy
        ## 3. environment interatcion stats
        self.csv.forward_episode_length = 0 
        self.csv.reset_episode_length = 0 # length of reset episode
        self.csv.forward_num_steps = 0
        self.csv.wall_time = None
        self.csv.sim_time = None

        if self.logging:
            # set up header
            self.csv.header = []
            self.csv.header.append('#1. Key stats')
            self.csv.header.append('episode')
            self.csv.header.append('reset episode')
            self.csv.header.append('reset triggered?')
            self.csv.header.append('soft reset success?')
            self.csv.header.append('soft reset GT success?')
            self.csv.header.append('total hard resets')
            self.csv.header.append('baseline hard reset')
            self.csv.header.append('avg reward')
            self.csv.header.append('avg reset reward')
            self.csv.header.append('reset episodic return')
            self.csv.header.append('#2. Detailed stats')
            self.csv.header.append('Rreset before soft reset')
            self.csv.header.append('Rreset after soft reset')
            self.csv.header.append('probreset before soft reset (criterion)')
            self.csv.header.append('probreset before soft reset (mu)')
            self.csv.header.append('probreset before soft reset (std)')
            self.csv.header.append('probreset before soft reset (min)')
            self.csv.header.append('probreset before soft reset (max)')
            self.csv.header.append('probreset after soft reset (criterion)')
            self.csv.header.append('probreset after soft reset (mu)')
            self.csv.header.append('probreset after soft reset (std)')
            self.csv.header.append('probreset after soft reset (min)')
            self.csv.header.append('probreset after soft reset (max)')
            self.csv.header.append('state before soft reset')
            self.csv.header.append('state after soft reset')
            self.csv.header.append('misc')
            self.csv.header.append('#3. Env. interaction stats')
            self.csv.header.append('episode length')
            self.csv.header.append('reset episode length')
            self.csv.header.append('total steps')
            self.csv.header.append('reset total steps')
            self.csv.header.append('sim time (s)')
            self.csv.header.append('wall time (s)')
            # rename a entry
            csvheader = deepcopy(self.csv.header)
            env_name = self._params.env.name
            if env_name in ['cliff-cheetah', 'cliff-walker']:
                csvheader[csvheader.index('misc')] = 'in a ditch?'
            elif 'pusher' in env_name or 'peg-insertion' in env_name or 'ball-in-cup' in env_name:
                csvheader[csvheader.index('misc')] = ''
            elif env_name == 'waiter-ur3':
                csvheader[csvheader.index('misc')] = 'no spillage?'
            elif 'dualur3-reacher' in env_name:
                csvheader[csvheader.index('misc')] = 'spillage?'
            elif 'dualur3-ball-in-cup' in env_name:
                csvheader[csvheader.index('misc')] = ''
            else:
                raise ValueError('Unknown Environment')
            # set up tensorflow summary FileWriter
            writer_dir = '%s/%s_%s_writer/%s' % (os.path.expanduser(self._params.logger.dir_name), self._params.logger.file_name.prefix, self.jobid, self._params.logger.file_name.postfix)
            self.filewriter = tf.summary.FileWriter(writer_dir, self._reset_agent.sess.graph)
            # set up csv file
            csv_dir = '%s/%s_%s_writer' % (os.path.expanduser(self._params.logger.dir_name), self._params.logger.file_name.prefix, self.jobid)
            assert os.path.isdir(csv_dir)
            if self.evaluation:
                self.csv.filename_evaluation = '%s_%s_%seval.csv'%(self._params.logger.file_name.prefix, self.jobid, self._params.logger.file_name.postfix)
                self.csv.filepath_evaluation = os.path.join(csv_dir, self.csv.filename_evaluation)
                file_exists = os.path.isfile(self.csv.filepath_evaluation)
                self.csv.file_evaluation = open(self.csv.filepath_evaluation, 'a')
                self.csv.logger_evaluation = csv.writer(self.csv.file_evaluation)
                if not file_exists:
                    self.csv.logger_evaluation.writerow(csvheader)
                    self.csv.file_evaluation.flush()
            self.csv.filename = '%s_%s_%s.csv'%(self._params.logger.file_name.prefix, self.jobid, self._params.logger.file_name.postfix)
            self.csv.filepath = os.path.join(csv_dir, self.csv.filename)
            file_exists = os.path.isfile(self.csv.filepath)
            self.csv.file = open(self.csv.filepath, 'a')
            self.csv.logger = csv.writer(self.csv.file)
            # write header
            if not file_exists:
                self.csv.logger.writerow(csvheader)
                self.csv.file.flush()


    def __del__(self):
        if self.logging:
            self.csv.file.close()
            print('closed logger csv file (ResetEnv)')
            if self.evaluation:
                self.csv.file_evaluation.close()
                print('closed evaluation logger csv file (ResetEnv)')
            self.filewriter.close()
            print('closed tensorflow FileWriter (ResetEnv)')

    
    def _get_obs(self):
        return self.env._get_obs()

    
    def _get_prob_reset_criterion(self, prob_reset_ensemble):
        '''
        args:
            :arg prob_reset_ensemble:
            :type prob_reset_ensemble: numpy array (ensemble size,)
        '''
        return np.mean(prob_reset_ensemble)


    def _run_reset_policy_logger(self, reset_rewards, reset_done, reset_done_ground_truth, reset_episode_length):
        # environment specific actions
        env_name = self._params.env.name
        if env_name == 'cliff-cheetah':
            # in a ditch?
            self.csv.reset_msg = int(self._obs[0] > 14.0 and self._obs[1] < -1.0)
        elif env_name == 'cliff-walker':
            # in a ditch?
            self.csv.reset_msg = int(self._obs[0] > 6.0 and self._obs[1] < 0.0)
        elif 'peg-insertion' in env_name:
            pass

        self.csv.probreset_after_reset_criterion = self._prob_reset_criterion
        self.csv.probreset_after_reset_mu = np.mean(self._prob_reset_ensemble)
        self.csv.probreset_after_reset_std = np.std(self._prob_reset_ensemble)
        self.csv.probreset_after_reset_min = np.min(self._prob_reset_ensemble)
        self.csv.probreset_after_reset_max = np.max(self._prob_reset_ensemble)

        assert self.params.forward.env.action_dim == self.params.reset.env.action_dim
        zero_action = np.zeros([self.params.reset.env.action_dim])
        _, self.csv.rreset_after_reset = self.env._get_rewards(self._obs, zero_action)
        self.csv.state_after_reset = state_to_log(self.env)
        self.csv.reset_episode_length = reset_episode_length
        self.csv.avg_reset_reward = np.mean(reset_rewards)
        self.csv.reset_done = int(reset_done)
        self.csv.reset_done_ground_truth = int(reset_done_ground_truth)
        gamma = self.params.reset.ddpg.discount
        self.csv.discounted_reset_return = np.dot([gamma**i for i in range(reset_episode_length)], reset_rewards)


    def _run_reset_episodes(self, actor_idx=0, videorecorder=None, eval_mode=False):
        '''
        ResetEnv implementation of run_episodes
        '''
        self._obs = self.env._get_obs()
        episode_stats = {}
        episode_stats['initial state'] = state_to_log(self.env, self._obs)
        observations = []
        actions = []
        reset_rewards = []
        reset_episode_length = 0
        timestamps = []
        gamma = self.params.reset.ddpg.discount

        reset_actor = self._reset_agent.actors[actor_idx]

        outer_iterator = tqdm(range(self._reset_attempts), file=sys.stdout)
        reset_done_ground_truth = False
        num_pending_improve_calls = 0
        for attempt in outer_iterator:
            inner_iterator = tqdm(range(self._max_episode_steps), file=sys.stdout)
            for step in inner_iterator:
                if eval_mode:
                    reset_a = reset_actor.get_action(self._obs)
                else:
                    (reset_a, action_info) = reset_actor.get_noisy_action(self._obs)
                timestamp = time.time()
                if hasattr(self.env, 'rate'):
                    if num_pending_improve_calls > 0:
                        self.env.run_before_rate_sleep(func=self._reset_agent._improve)
                        num_pending_improve_calls -= 1
                    (next_obs, r, done, info) = self.env.step(np.squeeze(reset_a))
                else:
                    (next_obs, r, done, info) = self.env.step(reset_a)
                if videorecorder is not None: videorecorder.capture_frame()
                if self.isrender:
                    self.render()
                reset_reward = self._reset_reward_fn(next_obs, reset_a)
                reset_done, _ = self._reset_done_fn(next_obs)
                reset_done_gt, _ = self._reset_done_env_fn(next_obs)
                reset_done_ground_truth = (reset_done_gt and step >= self._min_episode_steps-1) or reset_done_ground_truth
                if not eval_mode: self._reset_agent.num_steps += 1
                observations.append(self._obs)
                actions.append(reset_a)
                reset_rewards.append(reset_reward)
                timestamps.append(timestamp)
                reset_episode_length += 1
                if eval_mode:
                    transition = {'observation': deepcopy(self._obs), 'action': reset_a, 'next observation': next_obs,
                        'reward': None, 'done': reset_done
                    }
                    training_stats = {}
                else:
                    transition = {'observation': deepcopy(self._obs), 'action': reset_a, 'log_pi': action_info['log_pi'],
                        'next observation': next_obs, 'reward': None, 'done': reset_done
                    }
                    transition['episode'] = -self._reset_agent.num_episodes
                    self._reset_agent.buffer.store(transition)
                    if hasattr(self.env, 'rate'):
                        training_stats = self.env.run_before_rate_sleep_return
                        num_pending_improve_calls += 1
                    # train actor & critic
                    else:
                        training_stats = self._reset_agent._improve()
                self._obs = next_obs

                # reached initial state distribution!
                if reset_done and step >= self._min_episode_steps-1:
                    if self.verbose:
                        print('\x1b[6;30;42m' + 'soft reset (attempt: %d) successful in %d/%d steps'
                            %(attempt + 1, step, self._max_episode_steps) + '\x1b[0m'
                        )
                    self._prob_reset_ensemble = self._reset_agent.evaluate_probability_ensembles(transition['observation'], transition['action'])
                    self._prob_reset_criterion = self._get_prob_reset_criterion(self._prob_reset_ensemble)
                    self._run_reset_policy_logger(reset_rewards, reset_done, reset_done_ground_truth, reset_episode_length)
                    if not eval_mode: self._reset_agent.num_episodes += 1
                    if not eval_mode: # for rule-based done
                        # save to goal example after successful soft reset (state triggering done)
                        self._reset_agent.goal_examples.append(next_obs)
                    # Update classifier & run reset agent logger (at the end of every episode)
                    episode_stats['observations'] = observations
                    episode_stats['actions'] = actions
                    episode_stats['returns'] = []
                    episode_stats['episode lengths'] = [reset_episode_length]
                    episode_stats['average step rewards'] = []
                    if reset_episode_length > 0:
                        episode_stats['returns'].append(np.dot([gamma**i for i in range(reset_episode_length)], reset_rewards))
                        episode_stats['average step rewards'].append(np.mean(reset_rewards))
                    episode_stats['terminal state'] = state_to_log(self.env, self._obs)
                    episode_stats['timestamps'] = timestamps
                    while num_pending_improve_calls > 0:
                        training_stats = self._reset_agent._improve() # train actor & critic
                        num_pending_improve_calls -= 1
                    # update classifier and write to log
                    self._reset_agent.run_episodes(num_episodes=0, episode_stats=episode_stats, training_stats=training_stats, eval_mode=eval_mode)
                    info['reset_episode_stats'] = episode_stats

                    inner_iterator.close()
                    outer_iterator.close()
                    return (next_obs, r, info)

        while num_pending_improve_calls > 0:
            training_stats = self._reset_agent._improve() # train actor & critic
            num_pending_improve_calls -= 1
        
        # failed to reach initial state distribution!
        if self.verbose:
            print('\x1b[6;30;41m' + 'soft reset unsuccessful after %d attempts of %d steps... resorting to hard reset (# of resets: %d)'
                %(attempt + 1, self._max_episode_steps, self._total_resets) + '\x1b[0m'
            )
        # prob ensembles
        self._prob_reset_ensemble = self._reset_agent.evaluate_probability_ensembles(transition['observation'], transition['action'])
        self._prob_reset_criterion = self._get_prob_reset_criterion(self._prob_reset_ensemble)
        self._run_reset_policy_logger(reset_rewards, reset_done, reset_done_ground_truth, reset_episode_length)
        if not eval_mode: self._reset_agent.num_episodes += 1
        
        # Update classifier & run reset agent logger (at the end of every episode)
        episode_stats['observations'] = observations
        episode_stats['actions'] = actions
        episode_stats['returns'] = []
        episode_stats['episode lengths'] = [reset_episode_length]
        episode_stats['average step rewards'] = []
        if reset_episode_length > 0:
            episode_stats['returns'].append(np.dot([gamma**i for i in range(reset_episode_length)], reset_rewards))
            episode_stats['average step rewards'].append(np.mean(reset_rewards))
        episode_stats['terminal state'] = state_to_log(self.env, self._obs)
        episode_stats['timestamps'] = timestamps
        # update classifier and write to log
        self._reset_agent.run_episodes(num_episodes=0, episode_stats=episode_stats, training_stats=training_stats, eval_mode=eval_mode)
        info['reset_episode_stats'] = episode_stats

        # resorting to hard reset..
        obs = self.env.reset()

        if not eval_mode: 
            self._reset_agent.goal_examples.append(obs)
            self._total_resets += 1
        
        return (obs, r, info)
                    

    def _reset(self, flag='', videorecorder=None, eval_mode=False):
        '''Internal implementation of reset() that returns additional info.'''
        self.csv.state_before_reset = state_to_log(self.env)
        self._prob_reset_trigger_ema = 0.0

        # find reset prob value
        if flag is 'triggered': # _reset called from step()
            self.csv.probreset_before_reset_criterion = self._prob_reset_criterion
            self.csv.probreset_before_reset_mu = np.mean(self._prob_reset_ensemble)
            self.csv.probreset_before_reset_std = np.std(self._prob_reset_ensemble)
            self.csv.probreset_before_reset_min = np.min(self._prob_reset_ensemble)
            self.csv.probreset_before_reset_max = np.max(self._prob_reset_ensemble)

            self.csv.reset_triggered = 1
        elif flag is 'requested': # _reset called from reset()
            self.csv.probreset_before_reset_criterion = None
            self.csv.probreset_before_reset_mu = None
            self.csv.probreset_before_reset_std = None
            self.csv.probreset_before_reset_min = None
            self.csv.probreset_before_reset_max = None

            self.csv.reset_triggered = 0
        else:
            self.csv.reset_triggered = -1
        # find reset reward
        obs = self.env._get_obs()
        assert self.params.forward.env.action_dim == self.params.reset.env.action_dim
        zero_action = np.zeros([self.params.reset.env.action_dim])
        _, self.csv.rreset_before_reset = self.env._get_rewards(obs, zero_action)

        # run the reset policy (randomly choose one agent if multiple agents are present)
        obs, r, info = self._run_reset_episodes(
            actor_idx=np.random.randint(self._reset_agent.params.actor.number_of_actors),
            videorecorder=videorecorder,
            eval_mode=eval_mode
        )

        # log metrics
        self.csv.avg_reward = np.mean(self._episode_rewards)
        self._episode_rewards = []

        done = False

        return (obs, r, done, info)

    
    def reset(self, force_hard=False, videorecorder=None, eval_mode=False, return_reset_episode_stats=False):
        '''
        Override the original Gym environment reset function
        '''
        if force_hard: # force hard reset
            return self.env.reset()
        else: # attempt soft reset
            if self.verbose:
                print('entering reset policy! (upon request)')
            (obs, r, done, info) = self._reset(flag='requested', videorecorder=videorecorder, eval_mode=eval_mode) # flag as requested by the forward agent
            if self.logging:
                self._log_to_csv(eval_mode=eval_mode) # log to csv
            # reset forward episode length after logging
            if self.csv.forward_episode_length == self.params.forward.ddpg.max_episode_length:
                self.csv.forward_episode_length = 0
            self._obs = obs

            if return_reset_episode_stats:
                return obs, info['reset_episode_stats']
            else:
                return obs


    def step(self, action, videorecorder=None, eval_mode=False):
        '''
        Override the original Gym environment step function
        '''
        self.csv.baseline_hard_reset_stepcounter += 1

        # prob ensembles statistics
        self._prob_reset_ensemble = self._reset_agent.evaluate_probability_ensembles(self._obs, action)
        self._prob_reset_criterion = self._get_prob_reset_criterion(self._prob_reset_ensemble)

        ema_alpha = 1 - 1/getattr(self.params, 'consequtive_triggers', 1)
        trigger_cond = self._prob_reset_criterion < self._p_threshold
        self._prob_reset_trigger_ema = ema_alpha*self._prob_reset_trigger_ema + (1-ema_alpha)*trigger_cond
        if self._prob_reset_trigger_ema > 1-1/np.exp(1):
            if self.verbose:
                print('entering reset policy! (prob_reset=%f < prob_thresh=%f)'%(self._prob_reset_criterion, self._p_threshold))
            (obs, r, done, info) = self._reset(flag='triggered', videorecorder=videorecorder, eval_mode=eval_mode) # flag as triggered by the reset criterion
            info['_soft_reset_triggered'] = True
        else: # continue running forward policy
            self.csv.forward_episode_length += 1
            if not eval_mode: self.csv.forward_num_steps += 1
            (obs, r, done, info) = self.env.step(action)
            info['_soft_reset_triggered'] = False
            self._episode_rewards.append(r)
            if done:
                if not eval_mode: self.csv.baseline_hard_reset += 1
                self.csv.baseline_hard_reset_stepcounter = 0

        # update forward episode
        reached_max_episode_length = (self.csv.forward_episode_length == self.params.forward.ddpg.max_episode_length)
        forward_agent_episode_done = info['_soft_reset_triggered'] or done or reached_max_episode_length
        if forward_agent_episode_done and not eval_mode:
            self.csv.forward_num_episodes += 1
        # update baseline hard reset
        if self.csv.baseline_hard_reset_stepcounter == self.params.forward.ddpg.max_episode_length:
            if not eval_mode: self.csv.baseline_hard_reset += 1
            self.csv.baseline_hard_reset_stepcounter = 0

        self._obs = obs

        # log to csv
        if info['_soft_reset_triggered']:
            if self.logging:
                self._log_to_csv(eval_mode=eval_mode)
            # reset forward episode length after logging
            if forward_agent_episode_done:
                self.csv.forward_episode_length = 0

        return (obs, r, done, info)


    def _log_to_csv(self, eval_mode=False):
        entry = {key: None for key in self.csv.header}
        # 1. key stats
        entry['episode'] = self.csv.forward_num_episodes # number of episodes (of the forward agent)
        entry['reset episode'] = self._reset_agent.num_episodes # number of calls to _reset() function
        entry['reset triggered?'] = self.csv.reset_triggered # triggered or requested?
        entry['soft reset success?'] = self.csv.reset_done # reset policy reached goal? (estimated)
        entry['soft reset GT success?'] = self.csv.reset_done_ground_truth # reset policy reached goal? (ground truth)
        entry['total hard resets'] = self._total_resets # number of hard resets (== number of soft reset fails)
        # Number of hard resets the forward agent is expected to experience 
        # without ResetEnv. Every time done is True or environment steps
        # after the latest done is True event exceeds max episode length 
        # advances this baseline by one. A more optimistic baseline may be 
        # the number of episodes of the forward agent
        entry['baseline hard reset'] = self.csv.baseline_hard_reset
        # average forward reward (between resets) + prevent nan 
        entry['avg reward'] = self.csv.avg_reward if not np.isnan(self.csv.avg_reward) else entry['avg reward']
        entry['avg reset reward'] = self.csv.avg_reset_reward # average reset reward
        entry['reset episodic return'] = self.csv.discounted_reset_return # discounted reset return
        # 2. detailed stats
        entry['Rreset before soft reset'] = self.csv.rreset_before_reset # rreset before attempting soft reset
        entry['Rreset after soft reset'] = self.csv.rreset_after_reset # rreset after attempting soft reset
        entry['probreset before soft reset (criterion)'] = self.csv.probreset_before_reset_criterion # probreset before attempting soft reset
        entry['probreset before soft reset (mu)'] = self.csv.probreset_before_reset_mu
        entry['probreset before soft reset (std)'] = self.csv.probreset_before_reset_std
        entry['probreset before soft reset (min)'] = self.csv.probreset_before_reset_min
        entry['probreset before soft reset (max)'] = self.csv.probreset_before_reset_max
        entry['probreset after soft reset (criterion)'] = self.csv.probreset_after_reset_criterion # probreset after attempting soft reset
        entry['probreset after soft reset (mu)'] = self.csv.probreset_after_reset_mu
        entry['probreset after soft reset (std)'] = self.csv.probreset_after_reset_std
        entry['probreset after soft reset (min)'] = self.csv.probreset_after_reset_min
        entry['probreset after soft reset (max)'] = self.csv.probreset_after_reset_max
        entry['state before soft reset'] = self.csv.state_before_reset # env state before attempting soft reset
        entry['state after soft reset'] = self.csv.state_after_reset # env state after attempting soft reset
        entry['misc'] = self.csv.reset_msg # some additional messages after running the reset policy
        # 3. envrionment interaction stats
        entry['episode length'] = self.csv.forward_episode_length
        entry['reset episode length'] = self.csv.reset_episode_length # reset episode length
        entry['total steps'] = self.csv.forward_num_steps
        entry['reset total steps'] = self._reset_agent.num_steps # reset total steps
        entry['sim time (s)'] = (self.csv.forward_num_steps + self._reset_agent.num_steps) * self.dt
        entry['wall time (s)'] = time.time() - self.init_time 
        # write to csv
        assert len(entry) == len(self.csv.header) # No additional entries!
        if eval_mode:
            self.csv.logger_evaluation.writerow([entry[key] for key in self.csv.header])
            self.csv.file_evaluation.flush()
        else:
            self.csv.logger.writerow([entry[key] for key in self.csv.header])
            self.csv.file.flush()
            # tensorflow summary FileWriter
            summaries = tf.Summary()
            for key in self.csv.header:
                summaries.value.add(tag=key, simple_value=entry[key])
            self.filewriter.add_summary(summaries, global_step=entry['episode'])
            self.filewriter.flush()