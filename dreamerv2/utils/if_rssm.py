from collections import namedtuple
import torch.distributions as td
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import *

RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])

RSSMState = Union[RSSMDiscState, RSSMContState]


class IFRSSMUtils(object):
    '''utility functions for dealing with rssm states'''

    def __init__(self, rssm_type, info):
        self.rssm_type = rssm_type
        if rssm_type == 'continuous':
            self.deter_size_s1 = info['deter_size_s1']
            self.deter_size_s2 = info['deter_size_s2']
            self.deter_size_s3 = info['deter_size_s3']
            self.deter_size_s4 = info['deter_size_s4']
            self.stoch_size_s1 = info['stoch_size_s1']
            self.stoch_size_s2 = info['stoch_size_s2']
            self.stoch_size_s3 = info['stoch_size_s3']
            self.stoch_size_s4 = info['stoch_size_s4']
            self.deter_size = self.deter_size_s1 + self.deter_size_s2 + self.deter_size_s3 + self.deter_size_s4
            self.stoch_size = self.stoch_size_s1 + self.stoch_size_s2 + self.stoch_size_s3 + self.stoch_size_s4
            self.min_std = info['min_std']
        elif rssm_type == 'discrete':
            self.deter_size_s1 = info['deter_size_s1']
            self.deter_size_s2 = info['deter_size_s2']
            self.deter_size_s3 = info['deter_size_s3']
            self.deter_size_s4 = info['deter_size_s4']
            self.deter_size = self.deter_size_s1 + self.deter_size_s2 + self.deter_size_s3 + self.deter_size_s4
            self.class_size = info['class_size']
            self.stoch_size_s1 = info['category_size_s1'] * self.class_size
            self.stoch_size_s2 = info['category_size_s2'] * self.class_size
            self.stoch_size_s3 = info['category_size_s3'] * self.class_size
            self.stoch_size_s4 = info['category_size_s4'] * self.class_size
            self.category_size = info['category_size_s1'] + info['category_size_s2'] + info['category_size_s3'] + info[
                'category_size_s4']
            self.category_size_s1 = info['category_size_s1']
            self.category_size_s2 = info['category_size_s2']
            self.category_size_s3 = info['category_size_s3']
            self.category_size_s4 = info['category_size_s4']
            self.stoch_size = self.category_size * self.class_size
        else:
            raise NotImplementedError
        self.deter_index = {1: [0, self.deter_size_s1],
                            2: [self.deter_size_s1, self.deter_size_s1 + self.deter_size_s2],
                            3: [self.deter_size_s1 + self.deter_size_s2,
                                self.deter_size_s1 + self.deter_size_s2 + self.deter_size_s3],
                            4: [self.deter_size_s1 + self.deter_size_s2 + self.deter_size_s3,
                                self.deter_size_s1 + self.deter_size_s2 + self.deter_size_s3 + self.deter_size_s4]}
        self.stoch_index = {1: [0, self.stoch_size_s1],
                            2: [self.stoch_size_s1, self.stoch_size_s1 + self.stoch_size_s2],
                            3: [self.stoch_size_s1 + self.stoch_size_s2,
                                self.stoch_size_s1 + self.stoch_size_s2 + self.stoch_size_s3],
                            4: [self.stoch_size_s1 + self.stoch_size_s2 + self.stoch_size_s3,
                                self.stoch_size_s1 + self.stoch_size_s2 + self.stoch_size_s3 + self.stoch_size_s4]}
        assert (((self.deter_size_s1 == 0) == (self.stoch_size_s1 == 0)) and (
                    (self.deter_size_s2 == 0) == (self.stoch_size_s2 == 0)) and (
                            (self.deter_size_s3 == 0) == (self.stoch_size_s3 == 0)) and (
                            (self.deter_size_s4 == 0) == (self.stoch_size_s4 == 0)))

    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                seq_to_batch(rssm_state.logit[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                seq_to_batch(rssm_state.mean[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.std[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
            )

    def rssm_seq_retrive(self, rssm_state, seq_begin, seq_end):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                rssm_state.logit[seq_begin:seq_end],
                rssm_state.stoch[seq_begin:seq_end],
                rssm_state.deter[seq_begin:seq_end],
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                rssm_state.mean[seq_begin:seq_end],
                rssm_state.std[seq_begin:seq_end],
                rssm_state.stoch[seq_begin:seq_end],
                rssm_state.deter[seq_begin:seq_end],
            )

    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                batch_to_seq(rssm_state.logit, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter, batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                batch_to_seq(rssm_state.mean, batch_size, seq_len),
                batch_to_seq(rssm_state.std, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter, batch_size, seq_len)
            )

    def get_dist(self, rssm_state):
        if self.rssm_type == 'discrete':
            shape = rssm_state.logit.shape
            logit = torch.reshape(rssm_state.logit, shape=(*shape[:-1], self.category_size, self.class_size))
            return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
        elif self.rssm_type == 'continuous':
            return td.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)

    def get_disentangled_dist(self, rssm_state):
        if self.rssm_type == 'discrete':
            shape = rssm_state.logit.shape
            logits_s1, logits_s2, logits_s3, logits_s4 = torch.split(rssm_state.logit,
                                                                     [self.stoch_size_s1, self.stoch_size_s2,
                                                                      self.stoch_size_s3, self.stoch_size_s4], dim=-1)
            logits_s1 = torch.reshape(logits_s1, shape=(*shape[:-1], self.category_size_s1, self.class_size))
            logits_s2 = torch.reshape(logits_s2, shape=(*shape[:-1], self.category_size_s2, self.class_size))
            logits_s3 = torch.reshape(logits_s3, shape=(*shape[:-1], self.category_size_s3, self.class_size))
            logits_s4 = torch.reshape(logits_s4, shape=(*shape[:-1], self.category_size_s4, self.class_size))
            dist_s1 = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits_s1), 1)
            dist_s2 = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits_s2), 1)
            dist_s3 = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits_s3), 1)
            dist_s4 = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits_s4), 1)
            return dist_s1, dist_s2, dist_s3, dist_s4
        elif self.rssm_type == 'continuous':
            mean_s1, mean_s2, mean_s3, mean_s4 = torch.split(rssm_state.mean, [self.stoch_size_s1, self.stoch_size_s2,
                                                                               self.stoch_size_s3, self.stoch_size_s4],
                                                             dim=-1)
            std_s1, std_s2, std_s3, std_s4 = torch.split(rssm_state.std,
                                                         [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3,
                                                          self.stoch_size_s4], dim=-1)
            dist_s1 = td.Independent(td.Normal(mean_s1, std_s1), 1)
            dist_s2 = td.Independent(td.Normal(mean_s2, std_s2), 1)
            dist_s3 = td.Independent(td.Normal(mean_s3, std_s3), 1)
            dist_s4 = td.Independent(td.Normal(mean_s4, std_s4), 1)
            return dist_s1, dist_s2, dist_s3, dist_s4

    def get_stoch_state(self, stats):
        if self.rssm_type == 'discrete':
            logit = stats['logit']
            shape = logit.shape
            logit = torch.reshape(logit, shape=(*shape[:-1], self.category_size, self.class_size))
            dist = torch.distributions.OneHotCategorical(logits=logit)
            stoch = dist.sample()
            stoch += dist.probs - dist.probs.detach()
            return torch.flatten(stoch, start_dim=-2, end_dim=-1)

        elif self.rssm_type == 'continuous':
            mean = stats['mean']
            std = stats['std']
            std = F.softplus(std) + self.min_std
            return mean + std * torch.randn_like(mean), std

    def rssm_stack_states(self, rssm_states, dim):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                torch.stack([state.logit for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim),
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                torch.stack([state.mean for state in rssm_states], dim=dim),
                torch.stack([state.std for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim),
            )

    def get_model_state(self, rssm_state):
        if self.rssm_type == 'discrete':
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)
        elif self.rssm_type == 'continuous':
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

    def replace_with_state_0(self, rssm_state, rssm_state_0, replace_index=[3, 4]):
        assert (len(rssm_state.deter.shape) == 2)
        new_deter = rssm_state.deter.clone()
        new_stoch = rssm_state.stoch.clone()
        for index in replace_index:
            new_deter[:, self.deter_index[index][0]:self.deter_index[index][1]] = rssm_state_0.deter[:,
                                                                                  self.deter_index[index][0]:
                                                                                  self.deter_index[index][1]]
            new_stoch[:, self.stoch_index[index][0]:self.stoch_index[index][1]] = rssm_state_0.stoch[:,
                                                                                  self.stoch_index[index][0]:
                                                                                  self.stoch_index[index][1]]
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                rssm_state.logit,
                new_stoch,
                new_deter
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                rssm_state.mean,
                rssm_state.std,
                new_stoch,
                new_deter
            )

    def get_asr_state(self, rssm_state):
        # h12_{t-1}, s12_t
        deter_dict = self.get_deter_state_dict(rssm_state)
        stoch_dict = self.get_stoch_state_dict(rssm_state)
        return torch.cat([deter_dict['s1'], deter_dict['s2'], stoch_dict['s1'], stoch_dict['s2']], dim=-1)
        # return torch.cat([deter_dict['s2'], stoch_dict['s2']], dim=-1)

    def get_reward_state(self, rssm_state):
        deter_dict = self.get_deter_state_dict(rssm_state)
        stoch_dict = self.get_stoch_state_dict(rssm_state)
        return torch.cat([deter_dict['s1'], deter_dict['s2'], stoch_dict['s1'], stoch_dict['s2']], dim=-1)
        # return torch.cat([deter_dict['s2'], stoch_dict['s2']], dim=-1)

    def get_non_asr_state(self, rssm_state):
        # s34_t
        deter_dict = self.get_deter_state_dict(rssm_state)
        stoch_dict = self.get_stoch_state_dict(rssm_state)
        return torch.cat([deter_dict['s3'], deter_dict['s4'], stoch_dict['s3'], stoch_dict['s4']], dim=-1)

    def get_controllable_state(self, rssm_state):
        # s_{t-1}, s13_t
        stoch = rssm_state.stoch[:-1]
        stoch_next = rssm_state.stoch[1:]
        stoch_s1_old, stoch_s2_old, stoch_s3_old, stoch_s4_old = torch.split(stoch,
                                                                             [self.stoch_size_s1, self.stoch_size_s2,
                                                                              self.stoch_size_s3, self.stoch_size_s4],
                                                                             dim=-1)
        stoch_s1, stoch_s2, stoch_s3, stoch_s4 = torch.split(stoch_next, [self.stoch_size_s1, self.stoch_size_s2,
                                                                          self.stoch_size_s3, self.stoch_size_s4],
                                                             dim=-1)
        return torch.cat([stoch, stoch_s1, stoch_s3], dim=-1)

    def get_non_controllable_state(self, rssm_state):
        # s_{t-1}, s24_t
        stoch = rssm_state.stoch[1:-1]
        stoch_next = rssm_state.stoch[2:]
        stoch_s1, stoch_s2, stoch_s3, stoch_s4 = torch.split(stoch_next, [self.stoch_size_s1, self.stoch_size_s2,
                                                                          self.stoch_size_s3, self.stoch_size_s4],
                                                             dim=-1)
        return torch.cat([stoch, stoch_s2, stoch_s4], dim=-1)

    def get_aux_action_state(self, rssm_state):
        # s_{t-1}, s24_t
        stoch = rssm_state.stoch[:-1]
        stoch_next = rssm_state.stoch[1:]
        stoch_s1, stoch_s2, stoch_s3, stoch_s4 = torch.split(stoch_next, [self.stoch_size_s1, self.stoch_size_s2,
                                                                          self.stoch_size_s3, self.stoch_size_s4],
                                                             dim=-1)
        return torch.cat([stoch.detach(), stoch_s2, stoch_s4], dim=-1), stoch

    def get_aux_reward_state(self, rssm_state, action):
        # s_{t-1}, s24_t
        stoch = rssm_state.stoch[:-2]
        action_t_1 = action[:-2]
        # action_t = action[1:]
        stoch_next = rssm_state.stoch[1:-1]
        stoch_s1_t_1, stoch_s2_t_1, _, _ = torch.split(stoch,
                                                       [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3,
                                                        self.stoch_size_s4], dim=-1)
        _, _, stoch_s3, stoch_s4 = torch.split(stoch_next, [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3,
                                                            self.stoch_size_s4], dim=-1)
        return torch.cat([stoch_s1_t_1.detach(), stoch_s2_t_1.detach(), stoch_s3, stoch_s4, action_t_1],
                         dim=-1), torch.cat([stoch_s1_t_1, stoch_s2_t_1, action_t_1], dim=-1)
        # return torch.cat([stoch_s3, stoch_s4, action_t_1], dim=-1), torch.cat([action_t_1], dim=-1)

    def get_aux_state(self, rssm_state, actions, rewards):
        # I(s_t^{1, 2}, a_{t-1}, s^{1, 2}_{t-1}; R_{t})
        # I(s_t^{3, 4}, a_{t-1}, s^{1, 2}_{t-1}; R_{t})
        # I(s_t^{1, 3}, s_{t-1}; a_{t-1})
        # I(s_t^{2, 4}, s_{t-1}; a_{t-1})

        # I(s_t^{1, 2}; R_{t} | a_{t-1}, s^{1, 2}_{t-1})
        # I(s_t^{3, 4}; R_{t} | a_{t-1}, s^{1, 2}_{t-1})
        # I(s_t^{1, 3}; a_{t-1}|s_{t-1})
        # I(s_t^{2, 4}; a_{t-1} \,|s_{t-1})
        a_t_1 = actions[1:-1]
        s_t_1 = rssm_state.stoch[:-2]
        s_t = rssm_state.stoch[1:-1]
        r_t = rewards[1:-1]
        s1_t_1, s2_t_1, s3_t_1, s4_t_1 = torch.split(s_t_1, [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3,
                                                             self.stoch_size_s4], dim=-1)
        s1_t, s2_t, s3_t, s4_t = torch.split(s_t, [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3,
                                                   self.stoch_size_s4], dim=-1)

        mine_reward_1 = (torch.cat([s1_t, s2_t, a_t_1, s1_t_1.detach(), s2_t_1.detach()], dim=-1), r_t)
        mine_reward_2 = (torch.cat([s3_t, s4_t, a_t_1, s1_t_1.detach(), s2_t_1.detach()], dim=-1), r_t)
        mine_action_1 = (
        torch.cat([s1_t, s3_t, s1_t_1.detach(), s2_t_1.detach(), s3_t_1.detach(), s4_t_1.detach()], dim=-1), a_t_1)
        mine_action_2 = (
        torch.cat([s2_t, s4_t, s1_t_1.detach(), s2_t_1.detach(), s3_t_1.detach(), s4_t_1.detach()], dim=-1), a_t_1)
        return mine_reward_1, mine_reward_2, mine_action_1, mine_action_2

    def get_stoch_state_dict(self, rssm_state):
        s1, s2, s3, s4 = torch.split(rssm_state.stoch,
                                     [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3, self.stoch_size_s4],
                                     dim=-1)
        return {'s1': s1, 's2': s2, 's3': s3, 's4': s4}

    def get_deter_state_dict(self, rssm_state):
        s1, s2, s3, s4 = torch.split(rssm_state.deter,
                                     [self.deter_size_s1, self.deter_size_s2, self.deter_size_s3, self.deter_size_s4],
                                     dim=-1)
        return {'s1': s1, 's2': s2, 's3': s3, 's4': s4}

    def get_mean_state_dict(self, rssm_state):
        s1, s2, s3, s4 = torch.split(rssm_state.mean,
                                     [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3, self.stoch_size_s4],
                                     dim=-1)
        return {'s1': s1, 's2': s2, 's3': s3, 's4': s4}

    def rssm_detach(self, rssm_state):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                rssm_state.logit.detach(),
                rssm_state.stoch.detach(),
                rssm_state.deter.detach(),
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                rssm_state.mean.detach(),
                rssm_state.std.detach(),
                rssm_state.stoch.detach(),
                rssm_state.deter.detach()
            )

    def _init_rssm_state(self, batch_size, **kwargs):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
            )


def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0] * shp[1], *shp[2:]])
    return batch_data


def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
    return seq_data

