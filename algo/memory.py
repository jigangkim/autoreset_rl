import copy
import numpy as np
import time


class ReplayBuffer(object):
    '''
    Replay buffer to store transitions
    '''
    def __init__(self, params, _params, name=''):
        '''

        args:
            :arg params:
            :type params:
            :arg _params:
            :type _params:
            :arg name:
            :type name:

        returns:

        '''
        # input args
        self.params = params.buffer
        self._params = _params
        self.name = name

        # buffer
        self.capacity = self.params.max_number_of_transitions
        self.num_transitions = 0
        self.index = 0
        self.data = None
        self.item_keys = None
        self.episode_stats = {}
    
    
    def __getitem__(self, index):
        return {key: d[index] for key, d in zip(self.item_keys, self.data)}


    def __setitem__(self, key, value):
        raise TypeError('Insertion forbidden!')


    def store(self, transition):
        # allocating memory (run only once)
        if self.data is None:
            self._preallocate(transition)
        
        # if episode id is provided
        if 'episode' in self.item_keys:
            if self.num_transitions == self.capacity: # consider overwrite
                overwritten_episode_id = self.data[self.item_keys.index('episode')][self.index]
                s_overwritten, e_overwritten = self.episode_stats[overwritten_episode_id]
                if s_overwritten == e_overwritten:
                    self.episode_stats.pop(overwritten_episode_id)
                else:
                    self.episode_stats[overwritten_episode_id] = ((self.index + 1) % self.capacity, e_overwritten)
            episode_id = transition['episode']
            if episode_id in self.episode_stats:
                s, _ = self.episode_stats[episode_id]
                self.episode_stats[episode_id] = (s, self.index)
            else:
                self.episode_stats[episode_id] = (self.index, self.index)

        # add/overwrite transition
        for key, d in zip(self.item_keys, self.data):
            d[self.index] = copy.deepcopy(transition[key])
        
        # update num_transitions and index
        self.num_transitions = min(self.num_transitions + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    
    def sample(self, batch_size, num_batches=1, identical_batches=False, n_step=1, *args, **kwargs):
        '''
        Sample batch(es) of transition(s) from buffer

        args:
            :arg batch_size:
            :type batch_size:
            :arg num_batches:
            :type num_batches:
            :arg identical_batches:
            :type identical_batches:
        
        returns:
            :return *: batch(es)
            :type *: list of dicts containing numpy arrays [num batch * {key:(dim,) or (batch size, dim), ...}]
        '''
        assert self.num_transitions > 0

        def get_indices(batch_size, n_step):
            if 'episode' in self.item_keys:
                episode_len = {k: (v[1] - v[0])%self.capacity + 1 for k, v in self.episode_stats.items()}
                episode_len_nstep = {k: v for k, v in episode_len.items() if v >= n_step}
                if len(episode_len_nstep) > 0:
                    episode_ids = np.array(list(episode_len_nstep.keys()))
                    sampled_ids = np.random.choice(episode_ids, size=batch_size, p=None)
                    d = self.episode_stats
                    lowers = np.array([d[idx][0] for idx in sampled_ids])
                    uppers = np.array([d[idx][1] if d[idx][1] >= d[idx][0] else d[idx][1] + self.capacity for idx in sampled_ids]) - (n_step-1)
                    return np.array([np.random.randint(lower, upper+1)%self.capacity for lower, upper in zip(lowers, uppers)]), n_step
            else:
                assert n_step == 1
            return np.random.randint(self.num_transitions, size=batch_size), 1

        if identical_batches:
            indices, n_step_ = get_indices(batch_size, n_step)
            sampled_indices = [copy.deepcopy(indices) for _ in range(num_batches)]
        else:
            sampled_indices = [get_indices(batch_size, n_step) for _ in range(num_batches)]
            n_step_ = sampled_indices[0][1]
            sampled_indices = [idxes[0] for idxes in sampled_indices]

        ret = [{key+'s': d[indices] for key, d in zip(self.item_keys, self.data)} for indices in sampled_indices]
        for r in ret:
            r['n-step'] = n_step_
        for i in range(n_step_-1):
            for indices, batch in zip(sampled_indices, ret):
                batch.update({key+'s_%d'%(i+1): d[(indices+i+1)%self.capacity] for key, d in zip(self.item_keys, self.data)})
        return ret


    def clear(self):
        self.num_transitions = 0
        self.index = 0

    
    def _preallocate(self, transition):
        '''
        Assume flat structure of items.
        '''
        self.item_keys = list(transition.keys())
        transition_np = [np.asarray(transition[key]) for key in self.item_keys]
        # check memory usage
        mem_usage = sum([x.nbytes for x in transition_np]) * self.capacity
        print('Replay buffer requires %dMB of memory (%s)' % (mem_usage * 1e-6, self.name))
        if mem_usage > 1e9:
            raise ResourceWarning('This replay buffer would preallocate > 1GB of memory (%s)' % (self.name))
            time.sleep(1.0)
        # preallocate buffer
        self.data = [np.ones(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]