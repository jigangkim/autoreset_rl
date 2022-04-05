from copy import deepcopy
import numpy as np
import tensorflow as tf
from .exploration import OrnsteinUhlenbeck


def state_to_log(env, obs=None):
    '''
    args:
        :arg env: OpenAI gym environment, DeepMind Control Suite env
        :type env: class
        :arg obs: single observation used for sanity check (assertion)
        :type obs: numpy array (obs_dim,)

    :returns:
        :return *: scalar value that summarizes the state of the environment
        :type *: float
    '''
    # Unwrap environment
    if type(env).__name__ == 'ResetEnv':
        env = env.env

    # Choose state to log
    if type(env).__name__ == 'CliffCheetahEnv':
        if obs is not None: assert np.all(obs == env._get_obs()) # make sure the env is not tampered with
        return env._get_obs()[0] # x position of COM [m]
    elif type(env).__name__ == 'CliffWalkerEnv':
        if obs is not None: assert np.all(obs == env._get_obs()) # make sure the env is not tampered with
        return env._get_obs()[0] # x position of COM [m]
    elif type(env).__name__ == 'PegInsertionEnv':
        if obs is not None: assert np.all(obs == env._get_obs()) # make sure the env is not tampered with
        peg_pos = np.hstack([env.get_body_com('leg_bottom'), env.get_body_com('leg_top')])
        if env._task == 'insert':
            return np.linalg.norm(peg_pos - env._goal) # peg-to-goal distance [m]
        elif env._task == 'remove':
            return np.linalg.norm(peg_pos - env._start) # peg-to-start distance [m]
        else:
            raise ValueError('Unknown task: %s', env._task)
    else:
        raise ValueError('Unknown environment')


def _get_activation(activation):
    if 'relu' in activation:
        return tf.nn.relu
    elif 'tanh' in activation:
        return tf.nn.tanh
    elif 'sigmoid' in activation:
        if len(activation.split('_')) == 1:
            return tf.nn.sigmoid
        elif len(activation.split('_')) == 3:
            lower = float(activation.split('_')[1])
            upper = float(activation.split('_')[2])
            def custom_sigmoid(x, name=None):
                if name is None: name = 'CustomSigmoid'
                return tf.add(lower, (upper - lower)*tf.nn.sigmoid(x), name=name)
            return custom_sigmoid 
        else:
            raise ValueError('Unrecognized sigmoid activation')
    elif activation == 'None':
        return None
    else:
        raise ValueError('Undefined activation parameter')


def _get_initializer(initializer):
    if 'xavier' in initializer:
        return tf.contrib.layers.xavier_initializer()
    else:
        raise ValueError('Undefined initializer parameter')


class ActorNetwork(object):
    '''
    Actor network graphs & operations
    '''
    def __init__(self, sess, params, _params, name=''):
        '''
        args:
            :arg sess: Tensorflow session
            :type sess: Tensorflow Session obejct
            :arg params: Parameters for ActorNetwork
            :type params: SimpleNamespace object
            :arg _params: Original parameters
            :type _params: SimpleNamespace object
            :arg name: Name for the ActorNetwork object (optional)
            :type name: String object
        
        returns:
            
        '''
        # input args
        self.sess = sess
        self.params = params.actor
        self._params = _params
        self.name = name

        # additional params
        self.obs_dim = self._params.env.obs_dim
        self.action_dim = self._params.env.action_dim

        # build networks
        self.obs = tf.placeholder(tf.float32, [None, self.obs_dim], name='observation')
        inputs = [self.obs]
        # randomized prior function
        if self.params.priors.use_prior:
            prior_layer_specs = deepcopy(self.params.layer_specs)
            prior_layer_specs[-1].activation = 'None' # prior is inserted before the last activation layer
            _, self.prior, self.weights_prior = \
                self.build_graph(inputs, prior_layer_specs, 'prior', trainable=False, build_prior=True)
        else:
            self.prior = None
        _, self.output, self.weights = \
            self.build_graph(inputs, self.params.layer_specs, 'behavior', prior=self.prior) # behavior policy
        _, self.output_target, self.weights_target = \
            self.build_graph(inputs, self.params.layer_specs, 'target', trainable=False, prior=self.prior) # target policy
        self.synchronize_ops = [] # synchronize target weights to behavior weights
        for var, var_target in zip(self.weights, self.weights_target):
            self.synchronize_ops.append(var_target.assign(var))

        # compute gradients
        self.action_grad = tf.placeholder(tf.float32, [None, self.action_dim])
        self.policy_grad = tf.gradients(ys=self.output, xs=self.weights, grad_ys=-self.action_grad)
        grads_and_weights = zip(self.policy_grad, self.weights)

        # define operations
        self.optimize_ops = tf.train.AdamOptimizer(self.params.learning_rate).apply_gradients(grads_and_weights)
        self.update_ops = [] # target network update operation
        for var, var_target in zip(self.weights, self.weights_target):
            # exponential moving average
            var_target_new = self.params.tau * var + (1 - self.params.tau) * var_target
            self.update_ops.append(var_target.assign(var_target_new))
        
        # initialize weights
        self.sess.run(tf.global_variables_initializer())

        # set initial target weights to behavior weights
        self.sess.run(self.synchronize_ops)

        # misc
        self.exploration = OrnsteinUhlenbeck(self.params, self._params, name='')


    def train(self, obs, action_grad):
        self.sess.run(self.optimize_ops, feed_dict={self.obs: obs, self.action_grad: action_grad})


    def update_target(self):
        self.sess.run(self.update_ops)


    def build_graph(self, placeholders, layer_specs, name, trainable=True, prior=None, build_prior=False):
        obs = placeholders[0]
        with tf.variable_scope(self.name + '/' + name):
            layer = obs
            for n, layer_spec in enumerate(layer_specs):
                activation = _get_activation(layer_spec.activation)
                initializer = _get_initializer(layer_spec.initializer)
                if n < len(layer_specs) - 1:
                    layer_name = 'hidden%d'%(n)
                else:
                    layer_name = 'action'
                # configuration for randomized prior functions
                if prior != None and n == (len(layer_specs) - 1):
                    layer = tf.layers.dense(layer, units=layer_spec.output_dim, activation=None,
                        kernel_initializer=initializer, name=layer_name, trainable=trainable
                    )
                    beta = self.params.priors.beta
                    if activation is None:
                        layer = tf.add(layer, beta*prior, name=layer_name)
                    else:
                        layer = activation(tf.add(layer, beta*prior), name=layer_name)
                else:
                    if build_prior:
                        layer = tf.layers.dense(layer, units=layer_spec.output_dim, activation=activation,
                            kernel_initializer=initializer, bias_initializer=initializer, name=layer_name, trainable=trainable
                        )
                    else:
                        layer = tf.layers.dense(layer, units=layer_spec.output_dim, activation=activation,
                            kernel_initializer=initializer, name=layer_name, trainable=trainable
                        )
            action = layer # last layer is the output
            # get all variables from the current variable scope
            weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

        return obs, action, weights
        
    
    def get_action(self, obs):
        if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
        assert obs.ndim == 2
        return np.squeeze(self.sess.run(self.output, feed_dict={self.obs: obs})) # (*, action_dim)

    
    def get_noisy_action(self, obs):
        assert obs.ndim == 1
        obs = np.expand_dims(obs, axis=0) # batchify
        action_mean = np.squeeze(self.sess.run(self.output, feed_dict={self.obs: obs})) # (action_dim)
        action, action_noise, log_pi = self.exploration.add_noise(action_mean)
        return action, {'action_mean': action_mean, 'action_noise': action_noise, 'log_pi': log_pi}


    def get_action_target(self, obs):
        if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
        assert obs.ndim == 2
        return np.squeeze(self.sess.run(self.output_target, feed_dict={self.obs: obs})) # (*, action_dim)


class CriticNetwork(object):
    '''
        Critic network graphs & operations
    '''
    def __init__(self, sess, params, _params, name=''):
        '''
        args:
            :arg sess: Tensorflow session
            :type sess: Tensorflow Session obejct
            :arg params: Parameters for CriticNetwork
            :type params: SimpleNamespace object
            :arg _params: Original parameters
            :type _params: SimpleNamespace object
            :arg name: Name for the CriticNetwork object (optional)
            :type name: String object
        
        returns:
            
        '''
        # input args
        self.sess = sess
        self.params = params.critic
        self._params = _params
        self.name = name

        # additional params
        self.obs_dim = self._params.env.obs_dim
        self.action_dim = self._params.env.action_dim

        # build network
        self.obs = tf.placeholder(tf.float32, [None, self.obs_dim], name='observation')
        self.action = tf.placeholder(tf.float32, [None, self.action_dim], name='action')
        inputs = [self.obs, self.action]
        self.label = tf.placeholder(tf.float32, [None, 1])
        # randomized prior function
        if self.params.priors.use_prior:
            prior_layer_specs = deepcopy(self.params.layer_specs)
            prior_layer_specs[-1].activation = 'None' # prior is inserted before the last activation layer
            _, _, self.prior, self.weights_prior = \
                self.build_graph(inputs, prior_layer_specs, 'prior', trainable=False, build_prior=True)
        else:
            self.prior = None
        _, _, self.output, self.weights = \
            self.build_graph(inputs, self.params.layer_specs, 'behavior', prior=self.prior) # critic for behavior policy
        _, _, self.output_target, self.weights_target = \
            self.build_graph(inputs, self.params.layer_specs, 'target', trainable=False, prior=self.prior) # target critic
        self.synchronize_ops = [] # synchronize target weights to behavior weights
        for var, var_target in zip(self.weights, self.weights_target):
            self.synchronize_ops.append(var_target.assign(var))

        loss_type = getattr(self.params, 'loss', 'mse')

        # define operations
        if loss_type == 'mse':
            self.mse_loss = tf.reduce_mean(tf.square(self.label - self.output))
            self.action_grad = tf.gradients(ys=self.output, xs=self.action)
        elif loss_type == 'cross-entropy':
            assert self.params.layer_specs[-1].activation == 'None', ''
            # loss is named as self.mse_loss solely for compatibility reasons
            self.mse_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.label))
            self.output = tf.nn.sigmoid(self.output)
            self.output_target = tf.nn.sigmoid(self.output_target)
            self.action_grad = tf.gradients(ys=self.output, xs=self.action)
        elif loss_type == 'rce':
            assert self.params.layer_specs[-1].activation == 'None', ''
            logit, logit_target = self.output, self.output_target
            output_type = getattr(self.params, 'output_type', 'halfsigmoid')
            clip_label = getattr(self.params, 'clip_label', False)
            clip_tdloss_weight = getattr(self.params, 'clip_tdloss_weight', False)
            objective_type = getattr(self.params, 'objective_type', None)
            if output_type == 'sigmoid':
                self.output = tf.nn.sigmoid(logit) 
                self.output_target = tf.nn.sigmoid(logit_target)
            elif output_type == 'halfsigmoid':
                self.output = tf.nn.sigmoid(logit)/2.0
                self.output_target = tf.nn.sigmoid(logit_target)/2.0
            else:
                raise ValueError('Undefined output_type!')
            logit_split1, logit_split2 = tf.split(axis=0, num_or_size_splits=2, value=logit)
            output_split1, output_split2 = tf.split(axis=0, num_or_size_splits=2, value=self.output)
            label_ones, label_cs = tf.split(axis=0, num_or_size_splits=2, value=self.label) # first half is ones, latter half is omegas
            gamma = self._params.ddpg.discount
            label_rce_loss2 = gamma*label_cs / (1 + (gamma-1)*label_cs)
            label_assert_op = tf.Assert(tf.reduce_all(tf.equal(label_ones, 1)), [label_ones]) # assert label_split1 is ones
            with tf.control_dependencies([label_assert_op]):
                label1 = tf.minimum(label_ones, 0.5) if clip_label else label_ones
                label2 = tf.minimum(label_rce_loss2, 0.5) if clip_label else label_rce_loss2
                if output_type == 'sigmoid':
                    rce_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_split1, labels=label1)
                    rce_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_split2, labels=label2)
                elif output_type == 'halfsigmoid':
                    rce_loss1 = tf.keras.losses.binary_crossentropy(label1, output_split1)
                    rce_loss2 = tf.keras.losses.binary_crossentropy(label2, output_split2)
                else:
                    raise ValueError('Undefined output_type!')
                rce_loss1_min = -tf.log(label1**label1*(1-label1)**(1-label1))
                rce_loss2_min = -tf.log(label2**label2*(1-label2)**(1-label2))
                weights1 = (1 - gamma)*tf.ones_like(label1)
                label_cs_ = tf.clip_by_value(label_cs, 0, 0.5 if clip_tdloss_weight else gamma) # max gamma for numerical stability
                weights2 = 1 + gamma*label_cs_/(1 - label_cs_)
                rce_loss = tf.concat([rce_loss1, rce_loss2], axis=0)
                weights = tf.concat([weights1, weights2], axis=0)
                # loss is named as self.mse_loss solely for compatibility reasons
                self.mse_loss = tf.reduce_mean(weights*rce_loss)
                self.rce_loss_goal = tf.reduce_mean(rce_loss1 - rce_loss1_min)
                self.rce_loss_TD = tf.reduce_mean(rce_loss2 - rce_loss2_min)
            # raise NotImplementedError('Undecided action gradient..')
            if objective_type is None:
                if output_type == 'sigmoid':
                    objective_type = 'q'
                elif output_type == 'halfsigmoid':
                    objective_type = 'p'
                else:
                    raise ValueError('Undefined output_type!')
            if objective_type == 'q':
                self.action_grad = tf.gradients(ys=self.output, xs=self.action) # C(s,a)
            elif objective_type == 'p':
                self.action_grad = tf.gradients(ys=self.output/(1-self.output), xs=self.action) # p(et+|s,a)
            else:
                raise ValueError('Undefined output_type!')
        else:
            raise ValueError('Unknown loss type!')
        # Adam optimizer operation
        self.optimize_ops = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.mse_loss)
        # weigth decay operation
        self.weight_decay_ops = []
        for weight in self.weights:
            self.weight_decay_ops.append(weight.assign(weight*(1 - self.params.weight_decay_rate*self.params.learning_rate)))
        # target network update operation
        self.update_ops = []
        for var, var_target in zip(self.weights, self.weights_target):
            # exponential moving average
            var_target_new = self.params.tau * var + (1 - self.params.tau) * var_target
            self.update_ops.append(var_target.assign(var_target_new))

        # initialize weights
        self.sess.run(tf.global_variables_initializer())

        # set target weights to behavior weights
        self.sess.run(self.synchronize_ops)


    def train(self, obs, action, label):
        _, _, loss = self.sess.run([self.optimize_ops, self.weight_decay_ops, self.mse_loss], feed_dict={
            self.obs: obs,
            self.action: action,
            self.label: label
        })

        return loss


    def update_target(self):
        self.sess.run(self.update_ops)
    
    
    def action_gradients(self, obs, action):
        if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
        assert obs.ndim == 2
        if action.ndim == 1: action = np.expand_dims(action, axis=0) # batchify
        assert action.ndim == 2
        return self.sess.run(self.action_grad, feed_dict={
            self.obs: obs,
            self.action: action
        })[0]


    def build_graph(self, placeholders, layer_specs, name, trainable=True, prior=None, build_prior=False):
        assert len(layer_specs) > 1
        obs = placeholders[0]
        action = placeholders[1]
        with tf.variable_scope(self.name + '/' + name):
            layer = obs
            for n, layer_spec in enumerate(layer_specs):
                activation = _get_activation(layer_spec.activation)
                initializer = _get_initializer(layer_spec.initializer)
                if n < len(layer_specs) - 1:
                    layer_name = 'hidden%d'%(n)
                else:
                    layer_name = 'q_value'
                if n == 1: # insert action after the first hidden layer
                    layer = tf.concat([layer, action], axis=-1, name='hidden0concat')
                # configuration for randomized prior functions
                if prior != None and n == (len(layer_specs) - 1):
                    layer = tf.layers.dense(layer, units=layer_spec.output_dim, activation=None,
                        kernel_initializer=initializer, name=layer_name, trainable=trainable
                    )
                    beta = self.params.priors.beta
                    if activation is None:
                        layer = tf.add(layer, beta*prior, name=layer_name)
                    else:
                        layer = activation(tf.add(layer, beta*prior), name=layer_name)
                # configuration for randomized prior functions
                else:
                    if build_prior:
                        layer = tf.layers.dense(layer, units=layer_spec.output_dim, activation=activation,
                            kernel_initializer=initializer, bias_initializer=initializer, name=layer_name, trainable=trainable
                        )
                    else:
                        layer = tf.layers.dense(layer, units=layer_spec.output_dim, activation=activation,
                            kernel_initializer=initializer, name=layer_name, trainable=trainable
                        )
            q_value = layer # last layer is the q value
            # get all variables from the current variable scope
            weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
    
        return obs, action, q_value, weights


    def evaluate(self, obs, action):
        if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
        if action.ndim == 1: action = np.expand_dims(action, axis=0) # batchify
        assert obs.ndim == 2 and action.ndim == 2
        return np.squeeze(self.sess.run(self.output, feed_dict={self.obs: obs, self.action: action}))

    
    def evaluate_target(self, obs, action):
        if obs.ndim == 1: obs = np.expand_dims(obs, axis=0) # batchify
        if action.ndim == 1: action = np.expand_dims(action, axis=0) # batchify
        assert obs.ndim == 2 and action.ndim == 2
        return np.squeeze(self.sess.run(self.output_target, feed_dict={self.obs: obs, self.action: action}))