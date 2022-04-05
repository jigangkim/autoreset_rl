import numpy as np
from copy import deepcopy


class OrnsteinUhlenbeck(object):
    '''
    Ornstein-Uhlenbeck process
    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
    '''
    def __init__(self, params, _params, name=''):
        '''
        args:
            :arg name: Name for object (optional)
            :type name: String object
            :arg params: Parameters for OrnsteinUhlenbeck
            :type name: SimpleNamespace object
            :arg _params: Original parameters
            :type name: SimpleNamespace object

        returns:

        '''
        # input args
        self.name = name
        self.params = params.exploration
        self._params = _params

        # OU-process
        self.action_dim = self._params.env.action_dim
        self.mu = self.params.mu * np.ones(self.action_dim) # "average" value at convergence
        self.theta = self.params.theta # rate of convergence
        self.sigma = self.params.sigma # size of noise
        self.dt = self.params.dt # stochastic process time step
        self.state = self.params.xlim*(2*np.random.rand(self.action_dim) - 1) # initial state (-xlim ~ xlim)
        self.x0 = deepcopy(self.state)
        self.exp_minus_theta_t = 1 


    def reset(self):
        self.state = self.params.xlim*(2*np.random.rand(self.action_dim) - 1)
        self.x0 = deepcopy(self.state)
        self.exp_minus_theta_t = 1


    def add_noise(self, action_mean):
        x = self.state
        dx = self.theta*(self.mu - x)*self.dt + self.sigma*np.random.randn(len(x))*np.sqrt(self.dt)
        self.state = x + dx
        # log\pi from E(xt) and var(xt) = cov(xt, xt)
        # log\pi = -N/2*log(2*\pi*var(xt)) - (xt - E(xt))^T(xt - E(xt))/var(xt)
        self.exp_minus_theta_t *= np.exp(-self.theta*self.dt)
        E_xt = self.x0 * self.exp_minus_theta_t + self.mu * (1 - self.exp_minus_theta_t)
        var_xt = 0.5*self.sigma**2/self.theta*(1 - self.exp_minus_theta_t**2)
        log_pi = -0.5*len(x)*np.log(2*np.pi*var_xt) - np.linalg.norm(self.state - E_xt)/var_xt
        return action_mean + x + dx, x + dx, log_pi