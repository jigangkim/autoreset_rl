'''Cliff environments.'''
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


def tolerance(x, bounds, margin):
    '''Returns 0 when x is within the bounds, and decays sigmoidally
    when x is within a certain margin outside the bounds.
    We've copied the function from [1] to reduce dependencies.

    [1] Tassa, Yuval, et al. "DeepMind Control Suite." arXiv preprint
    arXiv:1801.00690 (2018).
    '''
    (lower, upper) = bounds
    if lower <= x <= upper:
        return 0
    elif x < lower:
        dist_from_margin = lower - x
    else:
        assert x > upper
        dist_from_margin = x - upper
    loss_at_margin = 0.95
    w = np.arctanh(np.sqrt(loss_at_margin)) / margin
    s = np.tanh(w * dist_from_margin)
    return s*s


def huber(x, p):
    return np.sqrt(x*x + p*p) - p


class CliffCheetahEnv(HalfCheetahEnv):
    def __init__(self, task='forward'):
        if not task in ['forward', 'backward']:
            raise ValueError('Unknown task: %s' % task)
        self._task = task
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_cheetah.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)
        self.override_r_for_step = None
        self.override_done_for_step = None

        self._reset_model_x_pos = None

    def step(self, a):
        assert np.all(np.isfinite(a))
        (s, _, _, info) = super(CliffCheetahEnv, self).step(a)
        r = self._get_rewards(s, a)[0]
        done = self._get_done(s)
        if self.override_r_for_step is not None:
            r = self.override_r_for_step(s, a)
        if self.override_done_for_step is not None:
            done = self.override_done_for_step(s)
        return (s, r, done, info)

    def _get_obs(self):
        '''Modified to include the x coordinate.'''
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        if self._reset_model_x_pos is not None:
            qpos[0] += self._reset_model_x_pos
        elif self._task == 'forward':
            pass
        elif self._task == 'backward':
            qpos[0] += 10.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_rewards(self, s, a):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        (x, z, theta) = s[:3]
        xvel = s[9]
        # Reward the forward agent for running 9 - 11 m/s.
        forward_reward = (1.0 - tolerance(xvel, (9, 11), 7))
        theta_reward = 1.0 - tolerance(theta,
                                       bounds=(-0.05, 0.05),
                                       margin=0.1)
        # Reward the reset agent for being at the origin, plus
        # reward shaping to be near the origin and upright.
        reset_reward = 0.8 * (np.abs(x) < 0.5) + 0.1 * (1 - 0.2 * np.abs(x)) + 0.1 * theta_reward
        reset_reward = np.clip(reset_reward, 0, 1)
        if self._task == 'forward':
            return (forward_reward, reset_reward)
        elif self._task == 'backward':
            return (reset_reward, forward_reward)
    
    def _get_done(self, s):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        if self._task == 'forward':
            return False
        elif self._task == 'backward':
            qpos, qvel = s[:self.model.nq], s[-self.model.nv:]
            x = qpos[0] # COM x position in [m]
            theta = qpos[2] # pitch angle in [rad]  
            x_cond, theta_cond = np.abs(x) < 0.5, np.abs(theta) < 0.5
            return x_cond and theta_cond

    def _get_reset_done(self, s):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        qpos, qvel = s[:self.model.nq], s[-self.model.nv:]
        x = qpos[0] # COM x position in [m]
        theta = qpos[2] # pitch angle in [rad]        
        if self._task == 'forward':
            x_cond, theta_cond = np.abs(x) < 0.1, np.abs(theta) < 0.1
            return x_cond and theta_cond
        elif self._task == 'backward':
            x_cond, theta_cond = np.abs(x - 10.0) < 0.1, np.abs(theta) < 0.1
            return x_cond and theta_cond


class CliffWalkerEnv(Walker2dEnv):
    def __init__(self, task='forward'):
        if not task in ['forward', 'backward']:
            raise ValueError('Unknown task: %s' % task)
        self._task = task
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_walker.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)
        self.override_r_for_step = None
        self.override_done_for_step = None

        self._reset_model_x_pos = None

    def step(self, a):
        assert np.all(np.isfinite(a))
        (s, _, _, info) = super(CliffWalkerEnv, self).step(a)
        r = self._get_rewards(s, a)[0]
        done = self._get_done(s)
        if self.override_r_for_step is not None:
            r = self.override_r_for_step(s, a)
        if self.override_done_for_step is not None:
            done = self.override_done_for_step(s)
        return (s, r, done, info)

    def _get_obs(self):
        '''Modified to include the x coordinate.'''
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        init_qpos = np.copy(self.init_qpos)
        if self._reset_model_x_pos is not None:
            init_qpos[0] += self._reset_model_x_pos
        elif self._task == 'forward':
            pass
        elif self._task == 'backward':
            init_qpos[0] += 4.0
        self.set_state(
            init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def _get_rewards(self, s, a):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        x = s[0]

        # run x stand x upright
        torso_height = s[1]
        theta = s[2]
        is_standing = float(torso_height > 1.2)
        is_falling = float(torso_height < 0.7)
        run_reward = max(0.0, min(s[9]/2.0, 1.0)) # scaled linear velocity from Tassa et al.
        stand_reward = np.clip(0.25 * torso_height + 0.25 * is_standing + 0.5 * (1 - is_falling), 0, 1)
        upright_reward = 1.0 - tolerance(theta, bounds=(-0.75, 0.75), margin=0.5)
        reset_location_reward = np.clip(0.8 * (np.abs(x) < 0.5) + 0.2 * (1 - 0.2 * np.abs(x)), 0, 1)
        forward_reward = run_reward*stand_reward*upright_reward
        reset_reward = reset_location_reward*stand_reward*upright_reward
        if self._task == 'forward':
            return (forward_reward, reset_reward)
        elif self._task == 'backward':
            return (reset_reward, forward_reward)
    
    def _get_done(self, s):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        if self._task == 'forward':
            return False
        elif self._task == 'backward':
            qpos, qvel = s[:self.model.nq], s[-self.model.nv:]
            x = qpos[0] # COM x position in [m]
            h = qpos[1] # height of walker in [m]
            theta = qpos[2] # pitch angle in [rad]
            x_cond, h_cond, theta_cond = np.abs(x) < 0.5, np.abs(h - 1.3) < 0.3, np.abs(theta) < 1.0
            return x_cond and h_cond and theta_cond

    def _get_reset_done(self, s):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        qpos, qvel = s[:self.model.nq], s[-self.model.nv:]
        x = qpos[0] # COM x position in [m]
        h = qpos[1] # height of walker in [m]
        theta = qpos[2] # pitch angle in [rad]
        if self._task == 'forward': 
            x_cond, h_cond, theta_cond = np.abs(x) < 0.5, np.abs(h - 1.3) < 0.3, np.abs(theta) < 1.0
            return x_cond and h_cond and theta_cond
        elif self._task == 'backward':
            x_cond, h_cond, theta_cond = np.abs(x - 4.0) < 0.5, np.abs(h - 1.3) < 0.3, np.abs(theta) < 1.0
            return x_cond and h_cond and theta_cond