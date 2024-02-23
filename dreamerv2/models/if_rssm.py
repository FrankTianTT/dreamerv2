import torch
import torch.nn as nn
import torch.distributions as td
# from ..utils.drssm import DRSSMUtils, RSSMContState, RSSMDiscState, gumbel_sigmoid
from dreamerv2.utils.if_rssm import IFRSSMUtils, RSSMContState, RSSMDiscState
from typing import *


class TemporalPrior(nn.Module):
    def __init__(self, deter_size_s1, deter_size_s2, deter_size_s3, deter_size_s4,
                 stoch_size_s1, stoch_size_s2, stoch_size_s3, stoch_size_s4, node_size, rssm_type="continuous",
                 act_fn=nn.ELU) -> None:
        super().__init__()
        self.deter_size_s1 = deter_size_s1
        self.deter_size_s2 = deter_size_s2
        self.deter_size_s3 = deter_size_s3
        self.deter_size_s4 = deter_size_s4
        self.stoch_size_s1 = stoch_size_s1
        self.stoch_size_s2 = stoch_size_s2
        self.stoch_size_s3 = stoch_size_s3
        self.stoch_size_s4 = stoch_size_s4
        assert (((self.deter_size_s1 == 0) == (self.stoch_size_s1 == 0)) and (
                (self.deter_size_s2 == 0) == (self.stoch_size_s2 == 0)) and (
                        (self.deter_size_s3 == 0) == (self.stoch_size_s3 == 0)) and (
                        (self.deter_size_s4 == 0) == (self.stoch_size_s4 == 0)))
        self.node_size = node_size
        self.rssm_type = rssm_type
        self.act_fn = act_fn
        self.prior_s1, self.prior_s2, self.prior_s3, self.prior_s4 = None, None, None, None
        self._build_model()

    def _build_model(self):
        if self.deter_size_s1 > 0:
            temporal_prior_s1 = [nn.Linear(self.deter_size_s1, self.node_size), self.act_fn()]
        if self.deter_size_s2 > 0:
            temporal_prior_s2 = [nn.Linear(self.deter_size_s2, self.node_size), self.act_fn()]
        if self.deter_size_s3 > 0:
            temporal_prior_s3 = [nn.Linear(self.deter_size_s3, self.node_size), self.act_fn()]
        if self.deter_size_s4 > 0:
            temporal_prior_s4 = [nn.Linear(self.deter_size_s4, self.node_size), self.act_fn()]
        if self.rssm_type == 'discrete':
            if self.deter_size_s1 > 0:
                temporal_prior_s1 += [nn.Linear(self.node_size, self.stoch_size_s1)]
                self.prior_s1 = nn.Sequential(*temporal_prior_s1)
            if self.deter_size_s2 > 0:
                temporal_prior_s2 += [nn.Linear(self.node_size, self.stoch_size_s2)]
                self.prior_s2 = nn.Sequential(*temporal_prior_s2)
            if self.deter_size_s3 > 0:
                temporal_prior_s3 += [nn.Linear(self.node_size, self.stoch_size_s3)]
                self.prior_s3 = nn.Sequential(*temporal_prior_s3)
            if self.deter_size_s4 > 0:
                temporal_prior_s4 += [nn.Linear(self.node_size, self.stoch_size_s4)]
                self.prior_s4 = nn.Sequential(*temporal_prior_s4)
        elif self.rssm_type == 'continuous':
            if self.deter_size_s1 > 0:
                temporal_prior_s1 += [nn.Linear(self.node_size, 2 * self.stoch_size_s1)]
                self.prior_s1 = nn.Sequential(*temporal_prior_s1)
            if self.deter_size_s2 > 0:
                temporal_prior_s2 += [nn.Linear(self.node_size, 2 * self.stoch_size_s2)]
                self.prior_s2 = nn.Sequential(*temporal_prior_s2)
            if self.deter_size_s3 > 0:
                temporal_prior_s3 += [nn.Linear(self.node_size, 2 * self.stoch_size_s3)]
                self.prior_s3 = nn.Sequential(*temporal_prior_s3)
            if self.deter_size_s4 > 0:
                temporal_prior_s4 += [nn.Linear(self.node_size, 2 * self.stoch_size_s4)]
                self.prior_s4 = nn.Sequential(*temporal_prior_s4)

    def forward(self, input_tensor):
        input_deter_s1, input_deter_s2, input_deter_s3, input_deter_s4 = torch.split(input_tensor, [self.deter_size_s1,
                                                                                                    self.deter_size_s2,
                                                                                                    self.deter_size_s3,
                                                                                                    self.deter_size_s4],
                                                                                     dim=-1)
        mean_result_list = []
        std_result_list = []
        if self.rssm_type == 'discrete':
            logits_list = []
            if self.prior_s1 is not None:
                logits_s1 = self.prior_s1(input_deter_s1)
                logits_list.append(logits_s1)
            if self.prior_s2 is not None:
                logits_s2 = self.prior_s2(input_deter_s2)
                logits_list.append(logits_s2)
            if self.prior_s3 is not None:
                logits_s3 = self.prior_s3(input_deter_s3)
                logits_list.append(logits_s3)
            if self.prior_s4 is not None:
                logits_s4 = self.prior_s4(input_deter_s4)
                logits_list.append(logits_s4)
            return torch.cat(logits_list, dim=-1)
        if self.rssm_type == 'continuous':
            if self.prior_s1 is not None:
                output_stoch_s1_mean, output_stoch_s1_std = torch.chunk(self.prior_s1(input_deter_s1), 2, dim=-1)
                mean_result_list.append(output_stoch_s1_mean)
                std_result_list.append(output_stoch_s1_std)
            if self.prior_s2 is not None:
                output_stoch_s2_mean, output_stoch_s2_std = torch.chunk(self.prior_s2(input_deter_s2), 2, dim=-1)
                mean_result_list.append(output_stoch_s2_mean)
                std_result_list.append(output_stoch_s2_std)
            if self.prior_s3 is not None:
                output_stoch_s3_mean, output_stoch_s3_std = torch.chunk(self.prior_s3(input_deter_s3), 2, dim=-1)
                mean_result_list.append(output_stoch_s3_mean)
                std_result_list.append(output_stoch_s3_std)
            if self.prior_s4 is not None:
                output_stoch_s4_mean, output_stoch_s4_std = torch.chunk(self.prior_s4(input_deter_s4), 2, dim=-1)
                mean_result_list.append(output_stoch_s4_mean)
                std_result_list.append(output_stoch_s4_std)
            return torch.cat(mean_result_list + std_result_list, dim=-1)


class TemporalPosterior(nn.Module):
    def __init__(self, deter_size_s1, deter_size_s2, deter_size_s3, deter_size_s4,
                 stoch_size_s1, stoch_size_s2, stoch_size_s3, stoch_size_s4, embedding_size, node_size,
                 rssm_type="continuous", act_fn=nn.ELU) -> None:
        super().__init__()
        self.deter_size_s1 = deter_size_s1
        self.deter_size_s2 = deter_size_s2
        self.deter_size_s3 = deter_size_s3
        self.deter_size_s4 = deter_size_s4
        self.stoch_size_s1 = stoch_size_s1
        self.stoch_size_s2 = stoch_size_s2
        self.stoch_size_s3 = stoch_size_s3
        self.stoch_size_s4 = stoch_size_s4
        self.embedding_size = embedding_size
        assert (((self.deter_size_s1 == 0) == (self.stoch_size_s1 == 0)) and (
                (self.deter_size_s2 == 0) == (self.stoch_size_s2 == 0)) and (
                        (self.deter_size_s3 == 0) == (self.stoch_size_s3 == 0)) and (
                        (self.deter_size_s4 == 0) == (self.stoch_size_s4 == 0)))
        self.node_size = node_size
        self.rssm_type = rssm_type
        self.act_fn = act_fn
        self.posterior_s1, self.posterior_s2, self.posterior_s3, self.posterior_s4 = None, None, None, None
        self._build_model()

    def _build_model(self):
        if self.deter_size_s1 > 0:
            temporal_posterior_s1 = [
                nn.Linear(self.deter_size_s1 + self.deter_size_s2 + self.embedding_size, self.node_size), self.act_fn()]
        if self.deter_size_s2 > 0:
            temporal_posterior_s2 = [
                nn.Linear(self.deter_size_s1 + self.deter_size_s2 + self.embedding_size, self.node_size), self.act_fn()]
        if self.deter_size_s3 > 0:
            temporal_posterior_s3 = [nn.Linear(self.deter_size_s3 + self.embedding_size, self.node_size), self.act_fn()]
        if self.deter_size_s4 > 0:
            temporal_posterior_s4 = [nn.Linear(self.deter_size_s4 + self.embedding_size, self.node_size), self.act_fn()]
        if self.rssm_type == 'discrete':
            stoch_size_s1 = self.stoch_size_s1
            stoch_size_s2 = self.stoch_size_s2
            stoch_size_s3 = self.stoch_size_s3
            stoch_size_s4 = self.stoch_size_s4
        elif self.rssm_type == 'continuous':
            stoch_size_s1 = 2 * self.stoch_size_s1
            stoch_size_s2 = 2 * self.stoch_size_s2
            stoch_size_s3 = 2 * self.stoch_size_s3
            stoch_size_s4 = 2 * self.stoch_size_s4
        else:
            raise NotImplementedError
        if self.deter_size_s1 > 0:
            temporal_posterior_s1 += [nn.Linear(self.node_size, stoch_size_s1)]
            self.posterior_s1 = nn.Sequential(*temporal_posterior_s1)
        if self.deter_size_s2 > 0:
            temporal_posterior_s2 += [nn.Linear(self.node_size, stoch_size_s2)]
            self.posterior_s2 = nn.Sequential(*temporal_posterior_s2)
        if self.deter_size_s3 > 0:
            temporal_posterior_s3 += [nn.Linear(self.node_size, stoch_size_s3)]
            self.posterior_s3 = nn.Sequential(*temporal_posterior_s3)
        if self.deter_size_s4 > 0:
            temporal_posterior_s4 += [nn.Linear(self.node_size, stoch_size_s4)]
            self.posterior_s4 = nn.Sequential(*temporal_posterior_s4)

    def forward(self, input_tensor):
        input_deter_s1, input_deter_s2, input_deter_s3, input_deter_s4, input_embedding = torch.split(input_tensor, [
            self.deter_size_s1, self.deter_size_s2, self.deter_size_s3, self.deter_size_s4, self.embedding_size],
                                                                                                      dim=-1)
        mean_result_list = []
        std_result_list = []
        input_s1 = torch.cat([input_deter_s1, input_deter_s2, input_embedding], dim=-1)
        input_s2 = torch.cat([input_deter_s1, input_deter_s2, input_embedding], dim=-1)
        input_s3 = torch.cat([input_deter_s3, input_embedding], dim=-1)
        input_s4 = torch.cat([input_deter_s4, input_embedding], dim=-1)
        if self.rssm_type == 'discrete':
            logits_list = []
            if self.posterior_s1 is not None:
                logits_s1 = self.posterior_s1(input_s1)
                logits_list.append(logits_s1)
            if self.posterior_s2 is not None:
                logits_s2 = self.posterior_s2(input_s2)
                logits_list.append(logits_s2)
            if self.posterior_s3 is not None:
                logits_s3 = self.posterior_s3(input_s3)
                logits_list.append(logits_s3)
            if self.posterior_s4 is not None:
                logits_s4 = self.posterior_s4(input_s4)
                logits_list.append(logits_s4)
            return torch.cat(logits_list, dim=-1)
        if self.rssm_type == 'continuous':
            if self.posterior_s1 is not None:
                output_stoch_s1_mean, output_stoch_s1_std = torch.chunk(self.posterior_s1(input_s1), 2, dim=-1)
                mean_result_list.append(output_stoch_s1_mean)
                std_result_list.append(output_stoch_s1_std)
            if self.posterior_s2 is not None:
                output_stoch_s2_mean, output_stoch_s2_std = torch.chunk(self.posterior_s2(input_s2), 2, dim=-1)
                mean_result_list.append(output_stoch_s2_mean)
                std_result_list.append(output_stoch_s2_std)
            if self.posterior_s3 is not None:
                output_stoch_s3_mean, output_stoch_s3_std = torch.chunk(self.posterior_s3(input_s3), 2, dim=-1)
                mean_result_list.append(output_stoch_s3_mean)
                std_result_list.append(output_stoch_s3_std)
            if self.posterior_s4 is not None:
                output_stoch_s4_mean, output_stoch_s4_std = torch.chunk(self.posterior_s4(input_s4), 2, dim=-1)
                mean_result_list.append(output_stoch_s4_mean)
                std_result_list.append(output_stoch_s4_std)
            return torch.cat(mean_result_list + std_result_list, dim=-1)

    def posterior_s3_s4_param(self):
        params = []
        if self.posterior_s3 is not None:
            params += list(self.posterior_s3.parameters())
        if self.posterior_s4 is not None:
            params += list(self.posterior_s4.parameters())
        # print("param length: ",len(params))
        return params


class IFRSSM(nn.Module, IFRSSMUtils):
    def __init__(
            self,
            action_size,
            rssm_node_size,
            embedding_size,
            device,
            rssm_type,
            info,
            act_fn=nn.ELU,
    ):
        nn.Module.__init__(self)
        IFRSSMUtils.__init__(self, rssm_type=rssm_type, info=info)
        self.device = device
        self.action_size = action_size
        self.node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.act_fn = act_fn
        self.disentangle = self.deter_size_s2 != 0
        if self.deter_size_s1 > 0:
            self.rnn1 = nn.GRUCell(self.deter_size_s1, self.deter_size_s1)
        if self.deter_size_s2 > 0:
            self.rnn2 = nn.GRUCell(self.deter_size_s2, self.deter_size_s2)
        if self.deter_size_s3 > 0:
            self.rnn3 = nn.GRUCell(self.deter_size_s3, self.deter_size_s3)
        if self.deter_size_s4 > 0:
            self.rnn4 = nn.GRUCell(self.deter_size_s4, self.deter_size_s4)
        self.deter_13 = nn.Linear(self.deter_size_s1 + self.deter_size_s3 + self.action_size,
                                  self.deter_size_s1 + self.deter_size_s3)
        self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()

    @property
    def all_stoch_size(self):
        return self.stoch_size_s1 + self.stoch_size_s2 + self.stoch_size_s3 + self.stoch_size_s4

    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action
        and embed it to deter size for rnn input
        """
        self.fc_embed_s1s2, self.fc_embed_s2, self.fc_embed_s3, self.fc_embed_s4 = None, None, None, None
        if self.deter_size_s1 > 0:
            fc_embed_s1s2a = [nn.Linear(self.stoch_size_s1 + self.stoch_size_s2 + self.action_size, self.deter_size_s1),
                              self.act_fn()]
            self.fc_embed_s1 = nn.Sequential(*fc_embed_s1s2a)
        if self.deter_size_s2 > 0:
            fc_embed_s1s2 = [nn.Linear(self.stoch_size_s1 + self.stoch_size_s2, self.deter_size_s2), self.act_fn()]
            self.fc_embed_s2 = nn.Sequential(*fc_embed_s1s2)
        if self.deter_size_s3 > 0:
            fc_embed_s3a = [nn.Linear(self.stoch_size_s3 + self.action_size, self.deter_size_s3), self.act_fn()]
            self.fc_embed_s3 = nn.Sequential(*fc_embed_s3a)
        if self.deter_size_s4 > 0:
            fc_embed_s3s4 = [nn.Linear(self.stoch_size_s4, self.deter_size_s4), self.act_fn()]
            self.fc_embed_s4 = nn.Sequential(*fc_embed_s3s4)

    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state
        and output prior over stochastic state
        """
        return TemporalPrior(self.deter_size_s1, self.deter_size_s2, self.deter_size_s3, self.deter_size_s4,
                             self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3, self.stoch_size_s4,
                             self.node_size, self.rssm_type, self.act_fn)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state
        and output posterior over stochastic states
        """
        # temporal_posterior = [nn.Linear(self.deter_size + self.embedding_size, self.node_size)]
        # temporal_posterior += [self.act_fn()]
        # if self.rssm_type == 'discrete':
        #     temporal_posterior += [nn.Linear(self.node_size, self.stoch_size)]
        # elif self.rssm_type == 'continuous':
        #     temporal_posterior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        # return nn.Sequential(*temporal_posterior)
        return TemporalPosterior(self.deter_size_s1, self.deter_size_s2, self.deter_size_s3, self.deter_size_s4,
                                 self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3, self.stoch_size_s4,
                                 self.embedding_size, self.node_size, self.rssm_type, self.act_fn)

    def forward_embed_state(self, stoch_state, prev_action):
        s1, s2, s3, s4 = torch.split(stoch_state,
                                     [self.stoch_size_s1, self.stoch_size_s2, self.stoch_size_s3, self.stoch_size_s4],
                                     dim=-1)
        result_list = []
        if self.fc_embed_s1 is not None:
            result_list.append(self.fc_embed_s1(torch.cat([s1, s2, prev_action], dim=-1)))
        if self.fc_embed_s2 is not None:
            result_list.append(self.fc_embed_s2(torch.cat([s1, s2], dim=-1)))
        if self.fc_embed_s3 is not None:
            result_list.append(self.fc_embed_s3(torch.cat([s3, prev_action], dim=-1)))
        if self.fc_embed_s4 is not None:
            result_list.append(self.fc_embed_s4(torch.cat([s4], dim=-1)))
        state_embed = torch.cat(result_list, dim=-1)
        return state_embed

    def forward_rnn(self, state_embed, prev_deter_state):
        prev_deter_state_s1, prev_deter_state_s2, prev_deter_state_s3, prev_deter_state_s4 = torch.split(
            prev_deter_state, [self.deter_size_s1, self.deter_size_s2, self.deter_size_s3, self.deter_size_s4], dim=-1)
        state_embed_s1, state_embed_s2, state_embed_s3, state_embed_s4 = torch.split(state_embed, [self.deter_size_s1,
                                                                                                   self.deter_size_s2,
                                                                                                   self.deter_size_s3,
                                                                                                   self.deter_size_s4],
                                                                                     dim=-1)
        result_list = []
        if self.deter_size_s1 > 0:
            result_list.append(self.rnn1(state_embed_s1, prev_deter_state_s1))
        if self.deter_size_s2 > 0:
            result_list.append(self.rnn2(state_embed_s2, prev_deter_state_s2))
        if self.deter_size_s3 > 0:
            result_list.append(self.rnn3(state_embed_s3, prev_deter_state_s3))
        if self.deter_size_s4 > 0:
            result_list.append(self.rnn4(state_embed_s4, prev_deter_state_s4))
        deter_state = torch.cat(result_list, dim=-1)
        return deter_state

    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        # deter_dict = self.get_deter_state_dict(prev_rssm_state)
        state_embed = self.forward_embed_state(prev_rssm_state.stoch * nonterms, prev_action)
        deter_state = self.forward_rnn(state_embed, prev_rssm_state.deter * nonterms)
        if self.rssm_type == 'discrete':
            prior_logit = self.fc_prior(deter_state)
            stats = {'logit': prior_logit}
            prior_stoch_state = self.get_stoch_state(stats)
            prior_rssm_state = RSSMDiscState(prior_logit, prior_stoch_state, deter_state)
        elif self.rssm_type == 'continuous':
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            stats = {'mean': prior_mean, 'std': prior_std}
            prior_stoch_state, std = self.get_stoch_state(stats)
            prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state

    def rollout_imagination(self, horizon: int, actor: nn.Module, prev_rssm_state):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        action_normal_std = []
        imag_log_probs = []
        for t in range(horizon):
            action, action_dist = actor((self.get_asr_state(rssm_state)).detach())
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            if actor.dist != 'one_hot':
                action_normal_std.append(action_dist.normal_std())
                imag_log_probs.append(action_dist.log_prob(action.detach()))
            else:
                action_normal_std.append(torch.zeros_like(action))
                imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        action_normal_std = torch.stack(action_normal_std, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        if self.rssm_type == 'discrete':
            posterior_logit = self.fc_posterior(x)
            stats = {'logit': posterior_logit}
            posterior_stoch_state = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)

        elif self.rssm_type == 'continuous':
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            stats = {'mean': posterior_mean, 'std': posterior_std}
            posterior_stoch_state, std = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, seq_len: int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor,
                            prev_rssm_state):
        priors = []
        posteriors = []
        for t in range(seq_len):
            prev_action = action[t] * nonterms[t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[t], prev_action, nonterms[t],
                                                                       prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=0)
        post = self.rssm_stack_states(posteriors, dim=0)
        return prior, post
