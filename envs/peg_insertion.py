'''PegInsertion environment.'''
import os
from gym.envs.mujoco import mujoco_env
import numpy as np


class PegInsertionEnv(mujoco_env.MujocoEnv):
    def __init__(self, task='insert', sparse=False):
        self._sparse = sparse
        self._task = task
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder, 'assets/peg_insertion.xml')
        self.override_r_for_step = None
        self.override_done_for_step = None
        super(PegInsertionEnv, self).__init__(xml_filename, 5)
        
    def _step(self, a):
        assert np.all(np.isfinite(a))
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        done = self._get_done(obs)
        r = self._get_rewards(obs, a)[0]
        if self.override_r_for_step is not None:
            r = self.override_r_for_step(obs, a)
        if self.override_done_for_step is not None:
            done = self.override_done_for_step(obs)
        return obs, r, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0
        self.viewer.cam.elevation = -45.0
        self.viewer.cam.azimuth = 270.0
        self.viewer.cam.fovy = 45.0

    def reset_model(self):
        if self._task == 'insert':
            # Reset peg above hole:
            qpos = np.array([0.44542705, 0.64189252, -0.39544481, -2.32144865,
                             -0.17935136, -0.60320289, 1.57110214])
        else:
            # Reset peg in hole
            qpos = np.array([0.52601062,  0.57254126, -2.0747581, -1.55342248,
                             0.15375072, -0.5747922,  0.70163815])
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_rewards(self, s, a, shaping_fn=[sum, lambda x,y: x*y]):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        peg_pos = np.hstack([self.get_body_com('leg_bottom'),
                            self.get_body_com('leg_top')])
        peg_bottom_z = peg_pos[2]
        goal_pos = np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2])
        start_pos = np.array([0.10600084, 0.15715909, 0.1496843, 0.24442536,
                              -0.09417238, 0.23726938])
        dist_to_goal = np.linalg.norm(goal_pos - peg_pos)
        dist_to_start = np.linalg.norm(start_pos - peg_pos)

        self._goal = goal_pos
        self._start = start_pos

        peg_to_goal_reward = np.clip(1. - dist_to_goal, 0, 1)
        peg_to_start_reward = np.clip(1. - dist_to_start, 0, 1)
        control_reward = np.clip(1 - 0.1 * np.dot(a, a), 0, 1)
        in_hole_reward = dist_to_goal < 0.1 and self.get_body_com("leg_bottom")[2] < -0.45

        insert_reward_vec = [in_hole_reward, control_reward, peg_to_goal_reward]
        remove_reward_vec = [peg_to_start_reward, control_reward]

        reward_sparse_coefs = [0.8, 0.2]
        reward_coefs = [0.5, 0.25, 0.25]
        if self._sparse:
            insert_reward = shaping_fn[0]([shaping_fn[1](r,coef) for (coef, r) in zip(reward_sparse_coefs, insert_reward_vec)])
        else:
            insert_reward = shaping_fn[0]([shaping_fn[1](r,coef) for (coef, r) in zip(reward_coefs, insert_reward_vec)])
        remove_reward = shaping_fn[0]([shaping_fn[1](r,coef) for (coef, r) in zip(reward_sparse_coefs, remove_reward_vec)])
        if self._task == 'insert':
            return (insert_reward, remove_reward)
        elif self._task == 'remove':
            return (remove_reward, insert_reward)
        else:
            raise ValueError('Unknown task: %s' % self._task)

    def _get_done(self, s):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        return False

    def _get_reset_done(self, s):
        assert np.all(s == self._get_obs()) # we assume that the reward is computed on-policy
        peg_pos = np.hstack([self.get_body_com('leg_bottom'),
                            self.get_body_com('leg_top')])
        peg_bottom_z = peg_pos[2]
        goal_pos = np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2])
        start_pos = np.array([0.10600084, 0.15715909, 0.1496843, 0.24442536,
                              -0.09417238, 0.23726938])
        dist_to_goal = np.linalg.norm(goal_pos - peg_pos)
        dist_to_start = np.linalg.norm(start_pos - peg_pos)
        in_hole = dist_to_goal < 0.1 and peg_bottom_z < -0.45
        insert_done = in_hole
        remove_done = dist_to_start < 0.20
        if self._task == 'insert':
            return remove_done
        elif self._task == 'remove':
            return insert_done
        else:
            raise ValueError('Unknown task: %s' % self._task)

    def _get_obs(self):
        obs = np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])
        return obs

    def camera_setup(self):
        pose = self.camera.get_pose()
        self.camera.set_pose(lookat=pose.lookat,
                             distance=pose.distance,
                             azimuth=270.,
                             elevation=-30.)