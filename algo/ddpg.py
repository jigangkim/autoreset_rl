from copy import deepcopy
import csv
import numpy as np
import os
import pkg_resources
import sys
import time
import tensorflow as tf
from tqdm import tqdm

gym_version = pkg_resources.get_distribution('gym').version
if gym_version == '0.9.3':
    from gym.monitoring.video_recorder import VideoRecorder
elif gym_version == '0.17.1':
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
else:
    ValueError('Invalid gym version!')

from .ddpg_utils import ActorNetwork, CriticNetwork, state_to_log
from .memory import ReplayBuffer


class DDPG(object):
    '''
    Deep Deterministic Policy Gradient agent from scratch
    '''
    def __init__(self, sess, env, params, name='', jobid='', logging=False, render=False, record=False):
        '''
        args:
            env: Environment implementing the Gym API
            actor: ActorNetwork object
            critics: List of CriticNetwork object(s)
            buffer: Replay buffer storing transitions (s, a, s', r)
            params: Parameters for DDPG 
        '''
        # input args
        self.sess = sess
        self.env = env
        self.params = params.ddpg
        self._params = params
        self.name = name
        self.jobid = jobid
        self.logging = logging
        self.render = render
        self.record = record
        
        # define actor(s), critic(s), replay buffer
        with tf.variable_scope(self.name):
            # possible actor/critic configuration: vanilla ddpg (1/1), critic ensemble (1/N), ddpg ensemble (N/N)
            assert self.params.actor.number_of_actors == self.params.critic.number_of_critics or self.params.actor.number_of_actors == 1

            self.actors = [ActorNetwork(self.sess, self.params, self._params, name='actor%d'%(i))
                for i in range(self.params.actor.number_of_actors)
            ]
            self.critics = [CriticNetwork(self.sess, self.params, self._params, name='critic%d'%(i)) 
                for i in range(self.params.critic.number_of_critics)
            ]
        self.output_target_ensembles = [critic.output_target for critic in self.critics]
        self.buffer = ReplayBuffer(self.params, self._params, name=self.name)

        # reset env
        if type(self.env).__name__ == 'ResetEnv':
            self.obs = self.env.reset(force_hard=True)
        else: # standard Gym env
            self.obs = self.env.reset()

        # state
        self.num_episodes = 0
        self.num_steps = 0
        self.init_time = time.time()
        self.stats = None # statistics

        # logger
        logheader = []
        logheader.append('episode')
        logheader.append('return')
        logheader.append('critic TD loss (RMSE)')
        logheader.append('average step reward')
        logheader.append('initial state')
        logheader.append('terminal state')
        logheader.append('episode length')
        logheader.append('total steps')
        logheader.append('sim time (s)')
        logheader.append('wall time (s)')
        self.logheader = logheader
        self.csvfilepath = None
        self.csvfilepath_evaluation = None
        self.logentry = None
        self.logentry_evaluation = None
        self.csvfile = None
        self.csvfile_evaluation = None
        self.csvlogger = None
        self.csvlogger_evaluation = None
        self.filewriter = None
        self.filewriter_evaluation = None

        # video recorder
        self.videorecorder = None
        self.video_stats = {}
        

    def run_episodes(self, actor_idx=0, num_episodes=1, episode_stats=None, training_stats=None, eval_mode=False, return_reset_episode_stats=False):
        if self.logging:
            self._configure_tf_filewriter()
            if self.record: self._configure_video_recorder(eval_mode=eval_mode)
        
        episodes_stats = []
        reset_episodes_stats = []
        for n in range(num_episodes):
            if return_reset_episode_stats:
                episode_stats, reset_episode_stats, training_stats = self._run_episode(actor_idx=actor_idx, eval_mode=eval_mode, return_reset_episode_stats=True)
                episodes_stats.append(episode_stats)
                reset_episodes_stats.append(reset_episode_stats)
            else:
                episode_stats, training_stats = self._run_episode(actor_idx=actor_idx, eval_mode=eval_mode, return_reset_episode_stats=False)
                episodes_stats.append(episode_stats)

        # additional stats
        additional_stats = {}
        # evaluate Q ensembles at the initial state (n_ensembles,)
        observation_batches = [np.expand_dims(self.env._get_obs(), axis=0) for _ in range(len(self.critics))]
        action_batches = self._get_target_actions_at_once(observation_batches)
        Q_ensembles = self._evaluate_targets_at_once(observation_batches, action_batches)
        additional_stats['Q(s0,a0)'] = Q_ensembles
        
        self.stats = {**episode_stats, **training_stats, **additional_stats}
        if self.logging:
            self._log_entry(eval_mode=eval_mode)
            self._print_entry(eval_mode=eval_mode)

        if return_reset_episode_stats:
            return episodes_stats, reset_episodes_stats
        else:
            return episodes_stats

    
    def _run_episode(self, actor_idx=0, eval_mode=False, return_reset_episode_stats=False):
        '''
        Collect transitions (s, a, s', r, done)

        args:
            num_episodes: Number of episodes to run
        '''
        self.obs = self.env._get_obs()
        episode_stats = {}
        episode_stats['initial state'] = state_to_log(self.env, self.obs)
        
        observations = []
        actions = []
        rewards = []
        episode_length = 0
        timestamps = []
        iterator = tqdm(range(self.params.max_episode_length), file=sys.stdout)
        num_pending_improve_calls = 0
        for _ in iterator:
            if self.logging and self.record: self.videorecorder.capture_frame()
            actor = self.actors[actor_idx]
            if eval_mode:
                a = actor.get_action(self.obs)
            else:
                (a, action_info) = actor.get_noisy_action(self.obs)
            timestamp = time.time()
            if type(self.env).__name__ == 'ResetEnv':
                if hasattr(self.env.env, 'rate'):
                    if num_pending_improve_calls > 0:
                        self.env.env.run_before_rate_sleep(func=self._improve)
                        num_pending_improve_calls -= 1
                (next_obs, r, done, info) = self.env.step(np.squeeze(a), self.videorecorder, eval_mode=eval_mode)
                if return_reset_episode_stats and info['_soft_reset_triggered']: reset_episode_stats = info['reset_episode_stats']
            else: # standard Gym env
                if hasattr(self.env, 'rate'):
                    if num_pending_improve_calls > 0:
                        self.env.run_before_rate_sleep(func=self._improve)
                        num_pending_improve_calls -= 1
                (next_obs, r, done, info) = self.env.step(np.squeeze(a))
            if type(done) is tuple:
                done, done_info = done
            if self.render: self.env.render()
            
            try: 
                _store_to_buffer = not info['_soft_reset_triggered']
                _end_episode = info['_soft_reset_triggered']
            except: # do NOT affect the original code
                _store_to_buffer = True
                _end_episode = False

            if _store_to_buffer:
                if eval_mode:
                    episode_length += 1
                    observations.append(self.obs)
                    actions.append(a)
                    rewards.append(r)
                    timestamps.append(timestamp)
                    training_stats = {}
                else:
                    self.num_steps += 1
                    episode_length += 1
                    observations.append(self.obs)
                    actions.append(a)
                    rewards.append(r)
                    timestamps.append(timestamp)
                    transition = {
                        'observation': deepcopy(self.obs),
                        'action': a,
                        'log_pi': action_info['log_pi'],
                        'next observation': next_obs,
                        'reward': r,
                        'done': done,
                        'episode': self.num_episodes
                    }
                    self.buffer.store(transition) # store transition
                    if type(self.env).__name__ == 'ResetEnv' and \
                        getattr(self.params.buffer, 'save_transitions_to_reset_buffer', False):
                        self.env._reset_agent.buffer.store(transition) # save to reset agent buffer as well if shared buffer option is chosen
                    if hasattr(self.env, 'rate'):
                        training_stats = self.env.run_before_rate_sleep_return
                        num_pending_improve_calls += 1
                    else:
                        training_stats = self._improve() # train actor & critic
            else:
                training_stats = {}

            self.obs = next_obs

            if done or _end_episode:
                iterator.close()
                break
        while num_pending_improve_calls > 0:
            training_stats = self._improve() # train actor & critic
            num_pending_improve_calls -= 1

        returns = []
        episode_lengths = []
        average_step_rewards = []
        gamma = self.params.discount
        if episode_length > 0:
            returns.append(np.dot([gamma**i for i in range(episode_length)], rewards))
            average_step_rewards.append(np.mean(rewards))
        episode_lengths.append(episode_length)
        
        episode_stats['observations'] = observations
        episode_stats['actions'] = actions
        episode_stats['returns'] = returns
        episode_stats['episode lengths'] = episode_lengths
        episode_stats['average step rewards'] = average_step_rewards
        episode_stats['terminal state'] = state_to_log(self.env, self.obs)
        episode_stats['timestamps'] = timestamps 

        # Don't reset environment after soft reset attempt (which includes hard reset even after failure)
        if not _end_episode: # reset agent has not been triggered
            if type(self.env).__name__ == 'ResetEnv':
                ret = self.env.reset(videorecorder=self.videorecorder, eval_mode=eval_mode, return_reset_episode_stats=return_reset_episode_stats)
                if return_reset_episode_stats: reset_episode_stats = ret[1]
            else: # standard Gym env
                self.env.reset()
        if self.logging and self.record:
            # close and disable videorecorder
            self.videorecorder.close()
            self.videorecorder.enabled = False # disable after recording one episode        

        if self.render: self.env.render()
        if not eval_mode: self.num_episodes += 1

        if return_reset_episode_stats:
            assert type(self.env).__name__ == 'ResetEnv'
            return episode_stats, reset_episode_stats, training_stats
        else:
            return episode_stats, training_stats


    def _configure_video_recorder(self, eval_mode=False):
        if self.videorecorder is None:
            self.video_stats['dir'] = '%s/%s_%s_writer' % (os.path.expanduser(self._params.logger.dir_name), self._params.logger.file_name.prefix, self.jobid)
            assert os.path.isdir(self.video_stats['dir'])
            self.video_stats['latest_record_step'] = -1
            try: self.video_stats['record_interval'] = self._params.logger.record_video_every_n_steps
            except: self.video_stats['record_interval'] = max(int(self._params.iteration_steps/200), 2000)
            self.video_stats['vidnum'] = 0 
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
                name = os.path.basename(f.name)
            os.makedirs('%s/.tmp'%(self.video_stats['dir']), exist_ok=True)
            self.videorecorder = VideoRecorder(self.env, enabled=True, path='%s/.tmp/%s'%(self.video_stats['dir'], name))

        curr_step = self.num_steps if not hasattr(self.env, '_reset_agent') else self.num_steps + self.env._reset_agent.num_steps
        curr_eps = self.num_episodes
        if eval_mode:
            self.videorecorder.enabled = True
            self.videorecorder.path = self.video_stats['dir'] + '/eval_video_eps_%d_totalstep_%d.mp4'%(curr_eps+1, curr_step)
            if os.path.isfile(self.videorecorder.path):
                self.videorecorder.path += '.mp4'
        else:
            # initial episode or periodic
            if self.video_stats['latest_record_step'] == -1 or curr_step >= self.video_stats['record_interval'] + self.video_stats['latest_record_step']:
                self.videorecorder.enabled = True
                self.videorecorder.path = self.video_stats['dir'] + '/video_%d_eps_%d_totalstep_%d.mp4'%(self.video_stats['vidnum'], curr_eps+1, curr_step)
                self.video_stats['vidnum'] += 1
                self.video_stats['latest_record_step'] = curr_step
            else: # otherwise
                self.videorecorder.enabled = False


    def _configure_tf_filewriter(self):
        # first time setup (tf summary FileWriter)
        if self.filewriter is None:
            writer_dir = '%s/%s_%s_writer/%s' % (os.path.expanduser(self._params.logger.dir_name), self._params.logger.file_name.prefix, self.jobid, self._params.logger.file_name.postfix)
            self.filewriter = tf.summary.FileWriter(writer_dir, self.sess.graph)


    def _log_entry(self, eval_mode=False):
        # first time setup (csv logger)
        if self.csvlogger is None:
            self.logentry = {key: None for key in self.logheader}
            # set up csv
            csv_dir = '%s/%s_%s_writer' % (os.path.expanduser(self._params.logger.dir_name), self._params.logger.file_name.prefix, self.jobid)
            assert os.path.isdir(csv_dir)
            filename = '%s_%s_%s.csv'%(self._params.logger.file_name.prefix, self.jobid, self._params.logger.file_name.postfix)
            self.csvfilepath = os.path.join(csv_dir, filename)
            file_exists = os.path.isfile(self.csvfilepath)
            self.csvfile = open(self.csvfilepath, 'a')
            self.csvlogger = csv.writer(self.csvfile)
            # write header
            if not file_exists:
                self.csvlogger.writerow(self.logheader)
                self.csvfile.flush()
        if self.csvlogger_evaluation is None and eval_mode:
            self.logentry_evaluation = {key: None for key in self.logheader}
            csv_dir = '%s/%s_%s_writer' % (os.path.expanduser(self._params.logger.dir_name), self._params.logger.file_name.prefix, self.jobid)
            assert os.path.isdir(csv_dir)
            filename_eval = '%s_%s_%seval.csv'%(self._params.logger.file_name.prefix, self.jobid, self._params.logger.file_name.postfix)
            self.csvfilepath_evaluation = os.path.join(csv_dir, filename_eval)
            file_exists = os.path.isfile(self.csvfilepath_evaluation)
            self.csvfile_evaluation = open(self.csvfilepath_evaluation, 'a')
            self.csvlogger_evaluation = csv.writer(self.csvfile_evaluation)
            # write header
            if not file_exists:
                self.csvlogger_evaluation.writerow(self.logheader)
                self.csvfile_evaluation.flush()
        
        if self.stats is not None: # statistics for log entry exists
            logentry = self.logentry_evaluation if eval_mode else self.logentry

            logentry['episode'] = self.num_episodes
            logentry['total steps'] = self.num_steps
            logentry['return'] = np.mean(self.stats['returns'])
            logentry['episode length'] = np.mean(self.stats['episode lengths'])
            logentry['average step reward'] = np.mean(self.stats['average step rewards'])
            logentry['initial state'] = self.stats['initial state']
            logentry['terminal state'] = self.stats['terminal state']
            logentry['sim time (s)'] = self.num_steps * self.env.dt
            logentry['wall time (s)'] = time.time() - self.init_time
            try: logentry['critic TD loss (RMSE)'] = np.sqrt(np.mean(self.stats['critic losses']))
            except: pass

            assert len(self.logheader) == len(logentry) # no additional keys!
            for key in logentry.keys():
                try:
                    if not np.isfinite(logentry[key]): logentry[key] = None
                except:
                    logentry[key] = None


    def _print_entry(self, eval_mode=False):
        if eval_mode:
            # print to file
            self.csvlogger_evaluation.writerow([self.logentry_evaluation[key] for key in self.logheader])
            self.csvfile_evaluation.flush()
            # print to terminal
            print('(evaluation) [%s]'%(self.name), self.logheader)
            print('(evaluation) [%s]'%(self.name), [self.logentry_evaluation[key] for key in self.logheader])
            # clear entry
            self.logentry_evaluation = {key: None for key in self.logheader}
            self.stats = None
        else:
            # print to file
            self.csvlogger.writerow([self.logentry[key] for key in self.logheader])
            self.csvfile.flush()
            # tf summary FileWriter
            summaries = tf.Summary()
            for key in self.logheader:
                summaries.value.add(tag=key, simple_value=self.logentry[key])
            self.filewriter.add_summary(summaries, global_step=self.logentry['episode'])
            self.filewriter.flush()
            # print to terminal
            if self.num_episodes%10 == 0:
                print('[%s]'%(self.name), self.logheader)
                print('[%s]'%(self.name), [self.logentry[key] for key in self.logheader])
            # clear entry
            self.logentry = {key: None for key in self.logheader}
            self.stats = None


    def _improve(self):
        '''
        Update critic(s) and actor(s)

        args:

        '''
        critic_losses = None
        if self.num_steps <= self.params.heatup_steps:
            try:
                print('[%s] HEATUP (%d/%d)'%(self.name, self.num_steps, self.params.heatup_steps))
            except:
                pass
        elif self.buffer.num_transitions == 0:
            print('No transitions in buffer!')
        else:
            transitions = self.buffer.sample(self.params.batch_size, num_batches=self.params.critic.number_of_critics)
            # train critic(s) and collect action gradient(s)
            observation_batches, action_batches, next_observation_batches, reward_batches, done_batches = \
                self._transitions_to_batches(transitions)
            next_target_action_batches = self._get_target_actions_at_once(next_observation_batches) # list of (N, action_dim)
            q_next_batches = self._evaluate_targets_at_once(next_observation_batches, next_target_action_batches) # list of (N, 1)
            # q_next -> 0 for 'done=True' transitions
            if getattr(self.params.critic, 'episodic_update', True):
                q_next_batches = np.array(q_next_batches) * np.logical_not(done_batches) # (critics, N, 1)
            else:
                q_next_batches = np.array(q_next_batches) # (critics, N, 1)
            target_q_batches = reward_batches + self.params.discount*q_next_batches # (critics, N, 1)
            reward_upper = getattr(self.params, 'reward_upper', np.inf)
            reward_lower = getattr(self.params, 'reward_lower', -np.inf)
            value_upper = reward_upper*self.params.max_episode_length if self.params.discount==1.0 else reward_upper/(1 - self.params.discount)
            value_lower = reward_lower*self.params.max_episode_length if self.params.discount==1.0 else reward_lower/(1 - self.params.discount)
            target_q_batches = np.clip(target_q_batches, value_lower, value_upper)
            # train critic network(s)
            critic_losses = self._train_critics_at_once(observation_batches, action_batches, target_q_batches) # list of floats
            action_for_grad_batches = self._get_actions_at_once(observation_batches) # list of (N, action_dim)
            action_grad_batches = self._action_gradients_at_once(observation_batches, action_for_grad_batches) # list of (N, action_dim)

            # train actor network
            self._train_actors_at_once(observation_batches, action_grad_batches)
            # update actor and critic target networks
            self._update_targets_at_once()

        return {'critic losses': critic_losses}
    
    
    def _transitions_to_batches(self, transitions):
        '''
        '''
        batches = ()
        batches = (*batches, [tr['observations'] for tr in transitions]) # list of (N, obs_dim)
        batches = (*batches, [tr['actions'] for tr in transitions]) # list of (N, action_dim)
        batches = (*batches, [tr['next observations'] for tr in transitions]) # list of (N, obs_dim)
        batches = (*batches, [np.expand_dims(tr['rewards'], axis=-1) for tr in transitions]) # list of (N, 1)
        batches = (*batches, [np.expand_dims(tr['dones'], axis=-1) for tr in transitions]) # list of (N, 1)
        return batches


    # target Q value ensembles for the same obs, action
    def evaluate_target_ensembles(self, obs, action):
        if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
        if action.ndim == 1: action = np.expand_dims(action, axis=0) # batchify
        assert obs.ndim == 2 and action.ndim == 2
        feed_dict = {}
        for critic in self.critics:
            feed_dict[critic.obs] = obs
            feed_dict[critic.action] = action 
        return np.squeeze(self.sess.run(self.output_target_ensembles, feed_dict=feed_dict))


    # Q value ensembles for the same obs, action
    def evaluate_ensembles(self, obs, action):
        fetches = [critic.output for critic in self.critics]
        if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
        if action.ndim == 1: action = np.expand_dims(action, axis=0) # batchify
        assert obs.ndim == 2 and action.ndim == 2
        feed_dict = {}
        for critic in self.critics:
            feed_dict[critic.obs] = obs
            feed_dict[critic.action] = action 
        return np.squeeze(self.sess.run(fetches, feed_dict=feed_dict))

    
    # target Q values for each obs, action (batch)
    def _evaluate_targets_at_once(self, observation_batches, action_batches):
        '''
        (internal function for self._improve)

        args:
            :arg observation_batches:
            :type observation_batches: list of numpy arrays
            :arg observation_batches:
            :type observation_batches: list of numpy arrays

        returns:
            :return sess.run(...):
            :type sess.run(...): list of numpy arrays
        '''
        feed_dict = {}
        for i, critic in enumerate(self.critics):
            obs = observation_batches[i]
            action = action_batches[i]
            if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
            if action.ndim == 1: action = np.expand_dims(action, axis=0) # batchify
            assert obs.ndim == 2 and action.ndim == 2
            feed_dict[critic.obs] = obs
            feed_dict[critic.action] = action
        return self.sess.run(self.output_target_ensembles, feed_dict=feed_dict) # (critics, batches,  1)

    
    # Q values for each obs, action (batch)
    def _evaluate_at_once(self, observation_batches, action_batches):
        '''
        returns q values
        '''
        fetches = [critic.output for critic in self.critics]
        feed_dict = {}
        for i, critic in enumerate(self.critics):
            obs = observation_batches[i]
            action = action_batches[i]
            if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
            if action.ndim == 1: action = np.expand_dims(action, axis=0) # batchify
            assert obs.ndim == 2 and action.ndim == 2
            feed_dict[critic.obs] = obs
            feed_dict[critic.action] = action
        return self.sess.run(fetches, feed_dict) # list of (N, 1)


    def _evaluate_ensembles_at_once(self, observation_batches, action_batches):
        '''
        Q value ensemble for batch(es) of observation(s)/action(s)

        args:
            :arg observation_batches: batch(es) of observation(s)
            :type observation_batches: list of numpy arrays [num batch * (batch size, obs dim)]
            :arg action_batches: batch(es) of action(s)
            :type action_batches: list of numpy arrays [num batch * (batch size, act dim)]
        
        returns:
            :return *: batch(es) of q value ensemble
            :type *: list of numpy arrays [num batch * (num critic, batch size)]
        '''
        assert len(observation_batches) == len(action_batches)
        num_batches = len(observation_batches)
        fetches = [critic.output for critic in self.critics]
        obs_concat = np.concatenate(observation_batches) # (num batch * batch size, obs dim)
        act_concat = np.concatenate(action_batches) # (num batch * batch size, act dim)
        q_concat = self.evaluate_ensembles(obs_concat, act_concat)
        if q_concat.ndim < 2: q_concat = np.expand_dims(np.atleast_1d(q_concat), axis=-1)
        return np.split(q_concat, num_batches, axis=1) # [num_batch * (num critic, batch size)]

    
    def _train_critics_at_once(self, observation_batches, action_batches, target_q_batches):
        '''
        (internal function for self._improve)

        args:
            :arg observation_batches:
            :type observation_batches: list of numpy arrays
            :arg action_batches:
            :type action_batches: list of numpy arrays
            :arg target_q_batches:
            :type target_q_batches: (critics, batches, 1) numpy array

        returns:
            :return losses:
            :type losses: list of floats
        '''
        fetches = []
        feed_dict = {}
        for i, critic in enumerate(self.critics):
            fetches.append([critic.optimize_ops, critic.weight_decay_ops, critic.mse_loss])
            feed_dict[critic.obs] = observation_batches[i]
            feed_dict[critic.action] = action_batches[i]
            feed_dict[critic.label] = target_q_batches[i]
        fetch_results = self.sess.run(fetches, feed_dict=feed_dict)
        losses = [result[2] for result in fetch_results]
        return losses


    def _get_target_actions_at_once(self, observation_batches):
        '''
        (internal function for self._improve)

        args:
            :arg observation_batches: 
            :type observation_batches: list of (N, obs_dim) numpy arrays

        returns:
            :return sess.run(...): 
            :type sess.run(...): list of numpy arrays
        '''
        if self.params.actor.number_of_actors == 1:
            actor = self.actors[0]
            num_requests = len(observation_batches)
            obs = np.concatenate(observation_batches, axis=0)
            action = self.sess.run(actor.output_target, feed_dict={actor.obs: obs})
            return np.split(action, num_requests, axis=0)
        else:
            fetches = [actor.output_target for actor in self.actors]
            feed_dict = {actor.obs: obs for actor, obs in zip(self.actors, observation_batches)}
            return self.sess.run(fetches, feed_dict=feed_dict)

    
    def _get_actions_at_once(self, observation_batches):
        '''
        (internal function for self._improve)

        args:
            :arg observation_batches: 
            :type observation_batches: list of (N, obs_dim) numpy arrays

        returns:
            :return sess.run(...): 
            :type sess.run(...): list of numpy arrays
        '''
        if self.params.actor.number_of_actors == 1:
            actor = self.actors[0]
            num_requests = len(observation_batches)
            obs = np.concatenate(observation_batches, axis=0)
            action = self.sess.run(actor.output, feed_dict={actor.obs: obs})
            return np.split(action, num_requests, axis=0)
        else:
            fetches = [actor.output for actor in self.actors]
            feed_dict = {actor.obs: obs for actor, obs in zip(self.actors, observation_batches)}
            return self.sess.run(fetches, feed_dict=feed_dict)


    def _train_actors_at_once(self, observation_batches, action_grad_batches):
        '''
        (internal function for self._improve)

        args:
            :arg observation_batches:
            :type observation_batches: list of numpy arrays
            :arg action_grad_batches:
            :type action_grad_batches: list of numpy arrays
        '''
        if self.params.actor.number_of_actors == 1:
            actor = self.actors[0]
            obs = np.vstack(observation_batches)
            action_grad = np.vstack(action_grad_batches)
            ind = np.random.randint(obs.shape[0], size=self.params.batch_size)
            actor.train(obs[ind], action_grad[ind])
        else:
            fetches = []
            feed_dict = {}
            for i, actor in enumerate(self.actors):
                fetches.append(actor.optimize_ops)
                feed_dict[actor.obs] = observation_batches[i]
                feed_dict[actor.action_grad] = action_grad_batches[i]
            self.sess.run(fetches, feed_dict=feed_dict)


    def _action_gradients_at_once(self, observation_batches, action_for_grad_batches):
        '''
        (internal function for self._improve)

        args:
            :arg observation_batches:
            :type observation_batches: list of numpy arrays
            :arg action_for_grad_batches:
            :type action_for_grad_batches: list of numpy arrays

        returns:
            :return sess.run(...):
            :type sess.run(...): list of arrays
        '''
        fetches = []
        feed_dict = {}
        for i, critic in enumerate(self.critics):
            obs = observation_batches[i]
            action = action_for_grad_batches[i]
            fetches.append(critic.action_grad)
            feed_dict[critic.obs] = obs
            feed_dict[critic.action] = action
        fetch_results = self.sess.run(fetches, feed_dict=feed_dict)
        return [result[0] for result in fetch_results]
    

    def _update_targets_at_once(self):
        '''
        (internal function for self._improve)
        '''
        fetches = []
        for actor in self.actors:
            fetches += actor.update_ops
        for critic in self.critics:
            fetches += critic.update_ops
        self.sess.run(fetches)


    def __del__(self):
        if self.logging:
            self.csvfile.close()
            print('[%s] closed logger csv file (%s)' % (self.name, type(self).__name__))
            self.csvfile_evaluation.close()
            print('[%s] closed evaluation logger csv file (%s)' % (self.name, type(self).__name__))
            self.filewriter.close()
            print('[%s] closed tensorflow FileWriter (%s)' % (self.name, type(self).__name__))