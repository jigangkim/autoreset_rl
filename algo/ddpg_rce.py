import numpy as np

from .ddpg import DDPG


def _get_statistics(activation):
    if 'min' in activation:
        return np.min
    elif 'max' in activation:
        return np.max
    elif 'mean' in activation:
        return np.mean
    else:
        raise ValueError('Undefined activation parameter')


def np_c2p(x):
    '''
    f: C(s,a) -> p(e|s,a)
    '''
    clip_x = np.clip(x, 0, 0.5)
    return clip_x/(1 - clip_x)


def np_c2l(x, gamma=1.0):
    '''
    f: C(s,a) -> label (gamma*C/(1 + (gamma-1)*C) or gamma*omega/(1 + gamma*omega))
    '''
    return gamma*x/(1 + (gamma-1)*x)


def np_l2c(x, gamma=1.0):
    '''
    f: label -> C(s,a) 
    '''
    return x/(x + gamma*(1-x))


class DDPGRecursiveClassifier(DDPG):
    '''
    Deep Deterministic Policy Gradient agent with RCE
    '''
    def __init__(self, *args, goal_examples=None, goal_examples_validation=None, **kwargs):
        # initialize actor(s), critic(s), replay buffer, log header
        super(DDPGRecursiveClassifier, self).__init__(*args, **kwargs)

        if self.name == 'forward':
            # replace orginal env reward/done functions with learned counterparts
            self.env.override_r_for_step = self._reward_fn # RCE does not explicitly learn rewards, however, implicit reward function can be derived by inverting the Bellman equation
        elif self.name == 'reset':
            # env must maintain original (forward) reward/done functions and should NOT be replaced by the (learned) reset reward/done functions
            pass
        else:
            raise ValueError('Undefined name')

        # input_args
        self.classifier_params = self._params.classifier
        if goal_examples is None:
            self.goal_examples = []
        else:
            self.goal_examples = goal_examples
        if goal_examples_validation is None:
            self.goal_examples_validation = []
        else:
            self.goal_examples_validation = goal_examples_validation

        # additional log header
        self.logheader.insert(len(self.logheader) - 2, 'critic rce loss')
        self.logheader.insert(len(self.logheader) - 2, 'critic rce loss (goal)')
        self.logheader.insert(len(self.logheader) - 2, 'critic rce loss (TD)')
        self.logheader.insert(len(self.logheader) - 2, '# of goal examples')
        self.logheader.insert(len(self.logheader) - 2, 'mean: goal p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'std: goal p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'min: goal p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'max: goal p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, '# of goal validation examples')
        self.logheader.insert(len(self.logheader) - 2, 'mean: goal validation p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'std: goal validation p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'min: goal validation p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'max: goal validation p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, '# of samples')
        self.logheader.insert(len(self.logheader) - 2, 'mean: sample p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'std: sample p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'min: sample p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'max: sample p(et+|..) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'mean: p(et+|s0,a0) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'std: p(et+|s0,a0) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'mean: p(et+|sT,aT) ensemble')
        self.logheader.insert(len(self.logheader) - 2, 'std: p(et+|sT,aT) ensemble')

        # redefine critic? no, but do an assertion check
        assert self.params.critic.loss == 'rce', 'loss type must be rce for %s'%(type(self).__name__)

    
    def get_env_fns(self):
        '''
        Return implicit (reconstructed) reward function and done functions
        '''
        reward_fn = self._reward_fn

        if self.name == 'forward':
            done_env_fn = lambda s: (self.env._get_done(s), {}) # empty done_info
        elif self.name == 'reset':
            done_env_fn = lambda s: (self.env._get_reset_done(s), {}) # empty done_info
        else:
            raise ValueError('Undefined name')

        done_fn = done_env_fn

        return {'%s_reward_fn' % (self.name): reward_fn, '%s_done_fn' % (self.name): done_fn, '%s_done_env_fn' % (self.name): done_env_fn}

    
    def _reward_fn(self, s, a, reward_type=None, statistics=None):
        '''
        Reward value(s) for observation(s)/action(s)

        args:
            :arg s: observation(s)
            :type s: numpy array (obs dim,) or (N, obs dim)
            :arg a: action(s)
            :type a: numpy array (action dim,) or (N, action dim)
            :arg reward_type: reward type (default: setting from config file)
            :type reward_type: string
            :arg statistics: statistics for ensembles
            :type statistics: numpy function with 'axis' keyword

        returns:
            :return *: reward(s)
            :type *: numpy array () or (N,)
        '''
        observation_batch = [np.atleast_2d(s)]
        action_batch = [np.atleast_2d(a)]
        reward_batch = \
            self._evaluate_rewards_at_once(observation_batch, action_batch, reward_type=reward_type, statistics=statistics)
        return np.squeeze(reward_batch)

    
    def _evaluate_rewards_at_once(self, observation_batches, action_batches, reward_type=None, statistics=None):
        '''
        Corresponding reward values for observation/action batches

        args:
            :arg observation_batches: batch(es) of observation(s)
            :type observation_batches: list of numpy arrays [num batch * (batch size, obs dim)]
            :arg action_batches: batch(es) of actions(s)
            :type action_batches: list of numpy arrays [num batch * (batch size, action dim)]
            :arg reward_type: reward type (default: setting from config file)
            :type reward_type: string
            :arg statistics: statistics for ensembles
            :type statistics: numpy function with 'axis' keyword

        returns:
            :return *: batch(es) of implicit reward(s)
            :type *: list of numpy arrays [num batch * (batch size, 1)]
        '''
        assert reward_type is None, 'RCE does not explicitly learn rewards, however, implicit reward function can be derived by inverting the Bellman equation'
        if statistics is None: statistics = _get_statistics(getattr(self.params.critic, 'reward_statistics', 'mean'))

        recursive_classifier_ensemble_batches = self._evaluate_ensembles_at_once(observation_batches, action_batches) # [num classifier * (batch size, 1)]
        event_prob_batches = [np.expand_dims(statistics(np_c2p(batch), axis=0), axis=-1) \
            for batch in recursive_classifier_ensemble_batches] # [num classifier * (batch size, 1)]
        # should be r(st,at,stplus1) = Ct/(1-Ct) - gamma*Ctplus1/(1-Ctplus1) but using (1-gamma)*Ct/(1-Ct) instead...
        return [(1-self.params.discount)*prob for prob in event_prob_batches]


    # required for compatibility reasons
    def evaluate_target_probability_ensembles(self, obs, action):
        '''
        return probabilities
        '''
        return np_c2p(self.evaluate_target_ensembles(obs, action))
    

    # required for compatibility reasons
    def evaluate_probability_ensembles(self, obs, action):
        '''
        return probabilities
        '''
        return np_c2p(self.evaluate_ensembles(obs, action))


    def run_episodes(self, actor_idx=0, num_episodes=1, episode_stats=None, training_stats=None, eval_mode=False):
        if self.logging:
            super(DDPGRecursiveClassifier, self)._configure_tf_filewriter()
            if self.record: super(DDPGRecursiveClassifier, self)._configure_video_recorder(eval_mode=eval_mode)
        
        episodes_stats = []
        # run episode
        for n in range(num_episodes):
            episode_stats, training_stats = super(DDPGRecursiveClassifier, self)._run_episode(actor_idx=actor_idx, eval_mode=eval_mode)
            latest_transition = self.buffer[(self.buffer.index - 1)%self.buffer.capacity]
            if latest_transition['done'] and not eval_mode:
                self.goal_examples.append(latest_transition['next observation'])
            else:
                # classifier-based done (use the goal examples that has been provided in the beginning)
                pass
            # modify stats
            rewards = self._reward_fn(np.array(episode_stats['observations']), np.array(episode_stats['actions']))
            rewards = np.atleast_1d(rewards)
            episode_stats['returns'] = np.sum([self.params.discount**i*reward for i, reward in enumerate(rewards)])
            episode_stats['average step rewards'] = np.mean(rewards)
            episodes_stats.append(episode_stats)
        
        # additional stats
        additional_stats = {}

        # evaluate goal (positives), goal validation (positives_validation), sample (negatives), s0, sT
        sampled_positive_indices = [np.random.randint(len(self.goal_examples), size=self.params.batch_size) \
            for _ in range(len(self.critics))]
        positives_batches = [[self.goal_examples[ind] for ind in sampled_positive_indice] \
            for sampled_positive_indice in sampled_positive_indices]
        positives_validation_batches = [self.goal_examples_validation for _ in range(len(self.critics))]
        negatives_batches = self.buffer.sample(self.params.batch_size, num_batches=len(self.critics))
        s0 = np.expand_dims(episode_stats['observations'][0], axis=0)
        sT = np.expand_dims(episode_stats['observations'][-1], axis=0)

        # combine into single batch
        ind1 = self.params.batch_size
        ind2 = ind1 + len(self.goal_examples_validation)
        ind3 = ind2 + self.params.batch_size
        ind4 = ind3 + 1
        ind5 = ind4 + 1
        observation_batches = [np.concatenate([neg['observations'], pos_val + pos, s0, sT], axis=0) \
            for neg, pos_val, pos in zip(negatives_batches, positives_validation_batches, positives_batches)]
        action_batches = self._get_actions_at_once(observation_batches) # [num batch * (batch size, action dim)]
        recursive_classifier_ensemble_batches = self._evaluate_at_once(observation_batches, action_batches) # [num classifier * (batch size, 1)]
        event_prob_batches = [np_c2p(batch) for batch in recursive_classifier_ensemble_batches]

        # save stats
        additional_stats['p(et+|..) sample'] = [batch[:ind1] for batch in event_prob_batches]
        additional_stats['p(et+|..) goal validation'] = [batch[ind1:ind2] for batch in event_prob_batches]
        additional_stats['p(et+|..) goal'] = [batch[ind2:ind3] for batch in event_prob_batches]
        additional_stats['p(et+|s0,a0)'] = [batch[ind3:ind4] for batch in event_prob_batches]
        additional_stats['p(et+|sT,aT)'] = [batch[ind4:ind5] for batch in event_prob_batches]
        
        self.stats = {**episode_stats, **training_stats, **additional_stats}
        if self.logging:
            self._log_entry(eval_mode=eval_mode)
            super(DDPGRecursiveClassifier, self)._print_entry(eval_mode=eval_mode)

        return episodes_stats


    def _log_entry(self, eval_mode):
        super(DDPGRecursiveClassifier, self)._log_entry(eval_mode=eval_mode)

        if self.stats is not None:
            logentry = self.logentry_evaluation if eval_mode else self.logentry

            try: logentry['critic rce loss'] = np.mean(self.stats['critic losses'])
            except: pass
            try: logentry['critic rce loss (goal)'] = np.mean(self.stats['critic losses (goal)'])
            except: pass
            try: logentry['critic rce loss (TD)'] = np.mean(self.stats['critic losses (TD)'])
            except: pass
            logentry['# of goal examples'] = len(self.goal_examples)
            goal_ensemble = [np.mean(ens) for ens in self.stats['p(et+|..) goal']]
            logentry['mean: goal p(et+|..) ensemble'] = np.mean(goal_ensemble)
            logentry['min: goal p(et+|..) ensemble'] = np.min(goal_ensemble)
            logentry['max: goal p(et+|..) ensemble'] = np.max(goal_ensemble)
            logentry['std: goal p(et+|..) ensemble'] = np.mean([np.std(ens) for ens in self.stats['p(et+|..) goal']])
            logentry['# of goal validation examples'] = len(self.goal_examples_validation)
            goal_validation_ensemble = [np.mean(ens) for ens in self.stats['p(et+|..) goal validation']]
            logentry['mean: goal validation p(et+|..) ensemble'] = np.mean(goal_validation_ensemble)
            logentry['min: goal validation p(et+|..) ensemble'] = np.min(goal_validation_ensemble)
            logentry['max: goal validation p(et+|..) ensemble'] = np.max(goal_validation_ensemble)
            logentry['std: goal validation p(et+|..) ensemble'] = np.mean([np.std(ens) for ens in self.stats['p(et+|..) goal validation']])
            logentry['# of samples'] = self.buffer.num_transitions
            sample_ensemble = [np.mean(ens) for ens in self.stats['p(et+|..) sample']]
            logentry['mean: sample p(et+|..) ensemble'] = np.mean(sample_ensemble)
            logentry['min: sample p(et+|..) ensemble'] = np.min(sample_ensemble)
            logentry['max: sample p(et+|..) ensemble'] = np.max(sample_ensemble)
            logentry['std: sample p(et+|..) ensemble'] = np.mean([np.std(ens) for ens in self.stats['p(et+|..) sample']])
            logentry['mean: p(et+|s0,a0) ensemble'] = np.mean(self.stats['p(et+|s0,a0)'])
            logentry['std: p(et+|s0,a0) ensemble'] = np.std(self.stats['p(et+|s0,a0)'])
            logentry['mean: p(et+|sT,aT) ensemble'] = np.mean(self.stats['p(et+|sT,aT)'])
            logentry['std: p(et+|sT,aT) ensemble'] = np.std(self.stats['p(et+|sT,aT)'])
            
            assert len(self.logheader) == len(logentry) # no additional keys!
            for key in logentry.keys():
                try:
                    if not np.isfinite(logentry[key]): logentry[key] = None
                except:
                    logentry[key] = None


    def _improve(self):
        critic_losses, critic_losses_goal, critic_losses_TD = None, None, None
        if self.num_steps <= self.params.heatup_steps:
            try:
                print('[%s] HEATUP (%d/%d)'%(self.name, self.num_steps, self.params.heatup_steps))
            except:
                pass
        elif self.buffer.num_transitions == 0 or len(self.goal_examples) == 0:
            print('No transitions in buffer or no goal examples!')
        else:
            for _ in range(getattr(self.classifier_params, 'train_steps', 1)):
                # 1. s(1), a(1), s(2), a(2), s_(2), a_(2)
                n_step = self.classifier_params.n_step
                self.classifier_params.q_combinator = getattr(self.classifier_params, 'q_combinator', 'independent')
                self.classifier_params.action_grad_q_combinator = getattr(self.classifier_params, 'action_grad_q_combinator', 'independent')
                if self.classifier_params.q_combinator == 'independent':
                    negatives_batches = self.buffer.sample(self.params.batch_size, num_batches=self.params.critic.number_of_critics, n_step=n_step)
                    sampled_positive_indices = [np.random.randint(len(self.goal_examples), size=self.params.batch_size) \
                        for _ in range(self.params.critic.number_of_critics)]
                else:
                    negatives_batches = self.buffer.sample(self.params.batch_size, num_batches=self.params.critic.number_of_critics, n_step=n_step, identical_batches=True)
                    sampled_positive_indices = [np.random.randint(len(self.goal_examples), size=self.params.batch_size)]*self.params.critic.number_of_critics
                # get s(1) from S* and a(1) <- pi(s(1))
                observation1_batches = [[self.goal_examples[ind] for ind in sampled_positive_indice] for sampled_positive_indice in sampled_positive_indices]
                action1_batches = self._get_actions_at_once(observation1_batches)
                # get s(2), a(2), s_(2) from buffer and a_(2) <- pi(s_(2))
                observation2_batches, action2_batches, next_observation2_batches, future_observation2_batches, n_steps = \
                    self._transitions_to_batches(negatives_batches)
                n_step = n_steps[0] if len(set(n_steps)) == 1 else 1
                next_action2_batches = self._get_actions_at_once(next_observation2_batches)
                if n_step > 1:
                    future_action2_batches = self._get_actions_at_once(future_observation2_batches)
                
                # 2. inputs and labels
                observation_batches = [np.concatenate([s1, s2]) for s1, s2 in zip(observation1_batches, observation2_batches)]
                action_batches = [np.concatenate([a1, a2]) for a1, a2 in zip(action1_batches, action2_batches)]
                label_ones_batches = [np.ones([self.params.batch_size,1]) for _ in range(self.params.critic.number_of_critics)] # list of (N, 1)
                if n_step > 1:
                    next_future_observation2_batches = [np.concatenate([s1, sn]) for s1, sn in zip(next_observation2_batches, future_observation2_batches)]
                    next_future_action2_batches = [np.concatenate([a0, an_1]) for a0, an_1 in zip(next_action2_batches, future_action2_batches)]
                    next_future_target_c_batches = self._evaluate_targets_at_once(next_future_observation2_batches, next_future_action2_batches) # list of (N, 1)
                    next_target_c_batches = [np.split(target_c, 2, axis=0)[0] for target_c in next_future_target_c_batches]
                    future_target_c_batches = [np.split(target_c, 2, axis=0)[1] for target_c in next_future_target_c_batches]
                    gamma, gamma_n = self.params.discount, self.params.discount**n_step
                    label_cs_batches = [np_l2c((np_c2l(next_c, gamma) + np_c2l(future_c, gamma_n))/2.0, gamma)
                        for next_c, future_c in zip(next_target_c_batches, future_target_c_batches)] # list of (N, 1)
                else:
                    next_target_c_batches = self._evaluate_targets_at_once(next_observation2_batches, next_action2_batches) # list of (N, 1)
                    label_cs_batches = next_target_c_batches
                if self.classifier_params.q_combinator == 'independent': # ensemble
                    pass
                elif self.classifier_params.q_combinator == 'mean': # consensus
                    label_cs_mean = np.mean(label_cs_batches, axis=0)
                    label_cs_batches = np.array([label_cs_mean] * self.params.critic.number_of_critics)
                elif self.classifier_params.q_combinator == 'min': # pessimistic
                    label_cs_min = np.min(label_cs_batches, axis=0)
                    label_cs_batches = np.array([label_cs_min] * self.params.critic.number_of_critics)
                elif self.classifier_params.q_combinator == 'max': # optimistic
                    label_cs_max = np.max(label_cs_batches, axis=0)
                    label_cs_batches = np.array([label_cs_max] * self.params.critic.number_of_critics)
                else:
                    raise ValueError('Invalid q_combinator')
                label_batches = [np.concatenate([ones, omegas]) for ones, omegas in zip(label_ones_batches, label_cs_batches)]

                # 3. train critic network(s)
                critic_losses, critic_losses_goal, critic_losses_TD = self._train_critics_at_once(observation_batches, action_batches, label_batches) # list of floats

            # 4. train actor network
            observation2_batch = observation2_batches[np.random.randint(len(observation2_batches))] # pick one batch
            action2_for_grad_batch = self._get_actions_at_once([observation2_batch])[0]
            observation2_batches_actor = [observation2_batch]*self.params.critic.number_of_critics
            action2_for_grad_batches_actor = [action2_for_grad_batch]*self.params.critic.number_of_critics
            action2_grad_batches = self._action_gradients_at_once(observation2_batches_actor, action2_for_grad_batches_actor) # list of (N, action_dim)
            self._train_actors_at_once(observation2_batches_actor, action2_grad_batches)
            
            # 5. update actor and critic target networks
            self._update_targets_at_once()

        return {'critic losses': critic_losses, 'critic losses (goal)': critic_losses_goal, 'critic losses (TD)': critic_losses_TD}


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
            fetches.append([critic.action_grad, critic.output])
            feed_dict[critic.obs] = obs
            feed_dict[critic.action] = action
        fetch_results = self.sess.run(fetches, feed_dict=feed_dict)
        q_batches = [result[1] for result in fetch_results]
        action_grad_batches = [result[0][0] for result in fetch_results]
        if self.classifier_params.action_grad_q_combinator == 'independent':
            return action_grad_batches
        elif self.classifier_params.action_grad_q_combinator == 'mean':
            action_grad_mean = np.mean(action_grad_batches, axis=0)
            return [action_grad_mean]*len(action_grad_batches)
        elif self.classifier_params.action_grad_q_combinator == 'min':
            argmin = np.argmin(q_batches, axis=0)
            action_grad_min = np.take_along_axis(np.array(action_grad_batches), np.expand_dims(argmin, axis=0), axis=0)
            action_grad_min = np.squeeze(action_grad_min)
            return [action_grad_min]*len(action_for_grad_batches)
        elif self.classifier_params.action_grad_q_combinator == 'max':
            argmax = np.argmax(q_batches, axis=0)
            action_grad_max = np.take_along_axis(np.array(action_grad_batches), np.expand_dims(argmax, axis=0), axis=0)
            action_grad_max = np.squeeze(action_grad_max)
            return [action_grad_max]*len(action_for_grad_batches)
        else:
            raise ValueError('Invalid action_grad_q_combinator')


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
            fetches.append([critic.optimize_ops, critic.weight_decay_ops, critic.mse_loss, critic.rce_loss_goal, critic.rce_loss_TD])
            feed_dict[critic.obs] = observation_batches[i]
            feed_dict[critic.action] = action_batches[i]
            feed_dict[critic.label] = target_q_batches[i]
        fetch_results = self.sess.run(fetches, feed_dict=feed_dict)
        losses = [result[2] for result in fetch_results]
        losses_goal = [result[3] for result in fetch_results]
        losses_TD = [result[4] for result in fetch_results]
        return losses, losses_goal, losses_TD


    def _transitions_to_batches(self, transitions):
        batches = ()
        batches = (*batches, [tr['observations'] for tr in transitions]) # list of (N, obs_dim)
        batches = (*batches, [tr['actions'] for tr in transitions]) # list of (N, action_dim)
        batches = (*batches, [tr['next observations'] for tr in transitions]) # list of (N, obs_dim)
        n_steps = [tr['n-step'] for tr in transitions]
        future_observations = [tr['next observations_%d'%(n_step-1)] if n_step > 1 else None for n_step, tr in zip(n_steps, transitions)]
        batches = (*batches, future_observations)
        batches = (*batches, n_steps)
        return batches