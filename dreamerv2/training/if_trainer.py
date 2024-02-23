import numpy as np
import torch
import torch.optim as optim
import os
import torch.nn.functional as F

from dreamerv2.utils.module import get_parameters, FreezeParameters
from dreamerv2.utils.algorithm import compute_return

from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.if_rssm import IFRSSM
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.utils.buffer import TransitionBuffer


class IFTrainer(object):
    def __init__(
            self,
            config,
            device,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.pixel = config.pixel
        self.kl_info = config.kl
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_steps = config.seed_steps
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip

        self._model_initialize(config)
        self._optim_initialize(config)

    def collect_seed_episodes(self, env):
        (s, _), done = env.reset(), False
        for i in range(self.seed_steps):
            a = env.action_space.sample()
            ns, r, done, timeout, _ = env.step(a)
            if done:
                self.buffer.add(s, a, r, done)
                (s, _), done = env.reset(), False
            else:
                self.buffer.add(s, a, r, done)
                s = ns

    def train_batch(self, train_metrics):
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_s1_l = []
        kl_s2_l = []
        kl_s3_l = []
        kl_s4_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            nonterms = torch.tensor(1 - terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1

            (model_loss, kl_s1, kl_s2, kl_s3, kl_s4, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist,
             posterior) = self.representation_loss(obs, actions, rewards, nonterms)

            self.model_optimizer.zero_grad()
            model_loss.backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
            self.model_optimizer.step()

            actor_loss, value_loss, target_info = self.actorcritc_loss(posterior)

            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_list), self.grad_clip_norm)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_list), self.grad_clip_norm)

            self.actor_optimizer.step()
            self.value_optimizer.step()

            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                post_ent = torch.mean(post_dist.entropy())

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_s1_l.append(kl_s1.item())
            kl_s2_l.append(kl_s2.item())
            kl_s3_l.append(kl_s3.item())
            kl_s4_l.append(kl_s4.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info['mean_targ'])
            min_targ.append(target_info['min_targ'])
            max_targ.append(target_info['max_targ'])
            std_targ.append(target_info['std_targ'])

        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_s1_loss'] = np.mean(kl_s1_l)
        train_metrics['kl_s2_loss'] = np.mean(kl_s2_l)
        train_metrics['kl_s3_loss'] = np.mean(kl_s3_l)
        train_metrics['kl_s4_loss'] = np.mean(kl_s4_l)
        train_metrics['reward_loss'] = np.mean(reward_l)
        train_metrics['obs_loss'] = np.mean(obs_l)
        train_metrics['value_loss'] = np.mean(value_l)
        train_metrics['actor_loss'] = np.mean(actor_l)
        train_metrics['prior_entropy'] = np.mean(prior_ent_l)
        train_metrics['posterior_entropy'] = np.mean(post_ent_l)
        train_metrics['pcont_loss'] = np.mean(pcont_l)
        train_metrics['mean_targ'] = np.mean(mean_targ)
        train_metrics['min_targ'] = np.mean(min_targ)
        train_metrics['max_targ'] = np.mean(max_targ)
        train_metrics['std_targ'] = np.mean(std_targ)

        return train_metrics

    def actorcritc_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(
                self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len - 1))

        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon,
                                                                                            self.ActionModel,
                                                                                            batched_posterior)

        imag_asrstates = self.RSSM.get_asr_state(imag_rssm_states)
        imag_rewardstates = self.RSSM.get_reward_state(imag_rssm_states)
        with FreezeParameters(self.world_list + self.value_list + [self.TargetValueModel] + [self.DiscountModel]):
            imag_reward_dist = self.RewardDecoder(imag_rewardstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_asrstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.DiscountModel(imag_asrstates)
            discount_arr = self.discount * torch.round(discount_dist.base_dist.probs)  # mean = prob(disc==1)

        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob,
                                                                policy_entropy)
        value_loss = self._value_loss(imag_asrstates, discount, lambda_returns)

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item()
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ': min_targ,
            'max_targ': max_targ,
            'std_targ': std_targ,
            'mean_targ': mean_targ,
        }

        return actor_loss, value_loss, target_info

    def representation_loss(self, obs, actions, rewards, nonterms):
        embed = self.ObsEncoder(obs)  # t to t+seq_len
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)

        post_modelstate = self.RSSM.get_model_state(posterior)  # t to t+seq_len
        post_asrstate = self.RSSM.get_asr_state(posterior)
        post_rewardstate = self.RSSM.get_reward_state(posterior)
        obs_dist = self.ObsDecoder(post_modelstate[:-1])  # t to t+seq_len-1
        reward_dist = self.RewardDecoder(post_rewardstate[:-1])  # t to t+seq_len-1
        pcont_dist = self.DiscountModel(post_asrstate[:-1])  # t to t+seq_len-1

        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, kl_s1, kl_s2, kl_s3, kl_s4 = self._kl_loss(prior, posterior)

        model_loss = (self.loss_scale['kl_s1'] * kl_s1 + self.loss_scale['kl_s2'] * kl_s2 +
                      self.loss_scale['kl_s3'] * kl_s3 + self.loss_scale['kl_s4'] * kl_s4 +
                      self.loss_scale['reward'] * reward_loss + obs_loss + self.loss_scale['discount'] * pcont_loss)
        return model_loss, kl_s1, kl_s2, kl_s3, kl_s4, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1],
                                        lambda_=self.lambda_)

        if self.config.actor_grad == 'reinforce':
            advantage = (lambda_returns - imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        # discount_arr = torch.cat([discount_arr[:1], discount_arr[1:]])
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1))
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_asrstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_asrstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates)
        value_loss = -torch.mean(value_discount * value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss

    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        prior_s1, prior_s2, prior_s3, prior_s4 = self.RSSM.get_disentangled_dist(prior)
        prior_s1_detach, prior_s2_detach, prior_s3_detach, prior_s4_detach = self.RSSM.get_disentangled_dist(
            self.RSSM.rssm_detach(prior))
        post_s1, post_s2, post_s3, post_s4 = self.RSSM.get_disentangled_dist(posterior)
        post_s1_detach, post_s2_detach, post_s3_detach, post_s4_detach = self.RSSM.get_disentangled_dist(
            self.RSSM.rssm_detach(posterior))
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_s1_lhs = torch.mean(torch.distributions.kl.kl_divergence(post_s1_detach, prior_s1))
            kl_s2_lhs = torch.mean(torch.distributions.kl.kl_divergence(post_s2_detach, prior_s2))
            kl_s3_lhs = torch.mean(torch.distributions.kl.kl_divergence(post_s3_detach, prior_s3))
            kl_s4_lhs = torch.mean(torch.distributions.kl.kl_divergence(post_s4_detach, prior_s4))
            kl_s1_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_s1, prior_s1_detach))
            kl_s2_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_s2, prior_s2_detach))
            kl_s3_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_s3, prior_s3_detach))
            kl_s4_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_s4, prior_s4_detach))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_s1_lhs = torch.max(kl_s1_lhs, kl_s1_lhs.new_full(kl_s1_lhs.size(), free_nats))
                kl_s1_rhs = torch.max(kl_s1_rhs, kl_s1_rhs.new_full(kl_s1_rhs.size(), free_nats))
                kl_s2_lhs = torch.max(kl_s2_lhs, kl_s2_lhs.new_full(kl_s2_lhs.size(), free_nats))
                kl_s2_rhs = torch.max(kl_s2_rhs, kl_s2_rhs.new_full(kl_s2_rhs.size(), free_nats))
                kl_s3_lhs = torch.max(kl_s3_lhs, kl_s3_lhs.new_full(kl_s3_lhs.size(), free_nats))
                kl_s3_rhs = torch.max(kl_s3_rhs, kl_s3_rhs.new_full(kl_s3_rhs.size(), free_nats))
                kl_s4_lhs = torch.max(kl_s4_lhs, kl_s4_lhs.new_full(kl_s4_lhs.size(), free_nats))
                kl_s4_rhs = torch.max(kl_s4_rhs, kl_s4_rhs.new_full(kl_s4_rhs.size(), free_nats))
            kl_s1_loss = alpha * kl_s1_lhs + (1 - alpha) * kl_s1_rhs
            kl_s2_loss = alpha * kl_s2_lhs + (1 - alpha) * kl_s2_rhs
            kl_s3_loss = alpha * kl_s3_lhs + (1 - alpha) * kl_s3_rhs
            kl_s4_loss = alpha * kl_s4_lhs + (1 - alpha) * kl_s4_rhs
        else:
            kl_s1_loss = torch.mean(torch.distributions.kl.kl_divergence(post_s1, prior_s1))
            kl_s2_loss = torch.mean(torch.distributions.kl.kl_divergence(post_s2, prior_s2))
            kl_s3_loss = torch.mean(torch.distributions.kl.kl_divergence(post_s3, prior_s3))
            kl_s4_loss = torch.mean(torch.distributions.kl.kl_divergence(post_s4, prior_s4))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_s1_loss = torch.max(kl_s1_loss, kl_s1_loss.new_full(kl_s1_loss.size(), free_nats))
                kl_s2_loss = torch.max(kl_s2_loss, kl_s2_loss.new_full(kl_s2_loss.size(), free_nats))
                kl_s3_loss = torch.max(kl_s3_loss, kl_s3_loss.new_full(kl_s3_loss.size(), free_nats))
                kl_s4_loss = torch.max(kl_s4_loss, kl_s4_loss.new_full(kl_s4_loss.size(), free_nats))
        return prior_dist, post_dist, kl_s1_loss, kl_s2_loss, kl_s3_loss, kl_s4_loss

    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss

    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])

    def _model_initialize(self, config):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size_s1, deter_size_s2, deter_size_s3, deter_size_s4 = config.rssm_info['deter_size_s1'], \
            config.rssm_info['deter_size_s2'], \
            config.rssm_info['deter_size_s3'], \
            config.rssm_info['deter_size_s4']
        deter_size = deter_size_s1 + deter_size_s2 + deter_size_s3 + deter_size_s4
        if config.rssm_type == 'continuous':
            stoch_size_s1, stoch_size_s2, stoch_size_s3, stoch_size_s4 = config.rssm_info['stoch_size_s1'], \
                config.rssm_info['stoch_size_s2'], \
                config.rssm_info['stoch_size_s3'], \
                config.rssm_info['stoch_size_s4']
            stoch_size = stoch_size_s1 + stoch_size_s2 + stoch_size_s3 + stoch_size_s4
        elif config.rssm_type == 'discrete':
            class_size = config.rssm_info['class_size']
            stoch_size_s1 = config.rssm_info['category_size_s1'] * class_size
            stoch_size_s2 = config.rssm_info['category_size_s2'] * class_size
            stoch_size_s3 = config.rssm_info['category_size_s3'] * class_size
            stoch_size_s4 = config.rssm_info['category_size_s4'] * class_size
            stoch_size = stoch_size_s1 + stoch_size_s2 + stoch_size_s3 + stoch_size_s4

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size

        asrstate_size = deter_size_s1 + deter_size_s2 + stoch_size_s1 + stoch_size_s2
        reward_size = deter_size_s1 + deter_size_s2 + stoch_size_s1 + stoch_size_s2

        self.buffer = TransitionBuffer(config.capacity, obs_shape, action_size, config.seq_len, config.batch_size,
                                       config.obs_dtype, config.action_dtype)
        self.RSSM = IFRSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type,
                           config.rssm_info).to(self.device)
        self.ActionModel = DiscreteActionModel(action_size, deter_size_s1 + deter_size_s2,
                                               stoch_size_s1 + stoch_size_s2, embedding_size, config.actor,
                                               config.expl).to(self.device)
        self.RewardDecoder = DenseModel((1,), reward_size, config.reward).to(self.device)
        self.ValueModel = DenseModel((1,), asrstate_size, config.critic).to(self.device)
        self.TargetValueModel = DenseModel((1,), asrstate_size, config.critic).to(self.device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())

        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), asrstate_size, config.discount).to(self.device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device)

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        actor_lr = config.lr['actor']
        value_lr = config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.world_list_except_RSSM = [self.ObsEncoder, self.RewardDecoder, self.ObsDecoder, self.DiscountModel,
                                       self.ValueModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr)

    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount['use']:
            print('\n Discount decoder: \n', self.DiscountModel)
        print('\n Actor: \n', self.ActionModel)
        print('\n Critic: \n', self.ValueModel)

    # This function should only be used for testing
    def init_extra_decoder(self):
        config = self.config
        obs_shape = self.config.obs_shape
        action_size = self.config.action_size
        deter_size_s1, deter_size_s2, deter_size_s3, deter_size_s4 = config.rssm_info['deter_size_s1'], \
            config.rssm_info['deter_size_s2'], config.rssm_info['deter_size_s3'], config.rssm_info['deter_size_s4']
        deter_size = deter_size_s1 + deter_size_s2 + deter_size_s3 + deter_size_s4

        stoch_size_s1, stoch_size_s2, stoch_size_s3, stoch_size_s4 = config.rssm_info['stoch_size_s1'], \
            config.rssm_info['stoch_size_s2'], config.rssm_info['stoch_size_s3'], config.rssm_info['stoch_size_s4']
        stoch_size = stoch_size_s1 + stoch_size_s2 + stoch_size_s3 + stoch_size_s4
        self.TestObsDecoder1 = ObsDecoder(obs_shape, deter_size_s1 + stoch_size_s1, config.obs_decoder).to(self.device)
        self.TestObsDecoder12 = ObsDecoder(obs_shape, deter_size_s1 + deter_size_s2 + stoch_size_s1 + stoch_size_s2,
                                           config.obs_decoder).to(self.device)
        self.TestObsDecoder3 = ObsDecoder(obs_shape, deter_size_s3 + stoch_size_s3, config.obs_decoder).to(self.device)
        self.TestObsDecoder4 = ObsDecoder(obs_shape, deter_size_s4 + stoch_size_s4, config.obs_decoder).to(self.device)
        self.TestObsDecoder34 = ObsDecoder(obs_shape, deter_size_s3 + deter_size_s4 + stoch_size_s3 + stoch_size_s4,
                                           config.obs_decoder).to(self.device)
        self.TestObsOptimizer = optim.Adam(get_parameters(
            [self.TestObsDecoder1, self.TestObsDecoder12, self.TestObsDecoder3, self.TestObsDecoder4,
             self.TestObsDecoder34]), self.config.lr['model'])

    def load_test_decoder(self, model_path):
        state_dict = torch.load(model_path)
        self.TestObsDecoder1.load_state_dict(state_dict["TestObsDecoder1"])
        self.TestObsDecoder12.load_state_dict(state_dict["TestObsDecoder12"])
        self.TestObsDecoder3.load_state_dict(state_dict["TestObsDecoder3"])
        self.TestObsDecoder4.load_state_dict(state_dict["TestObsDecoder4"])
        self.TestObsDecoder34.load_state_dict(state_dict["TestObsDecoder34"])

    # This function should only be used for testing
    def train_test_decoder_one_step(self, posterior, obs):
        deter_dict = self.RSSM.get_deter_state_dict(posterior)
        stoch_dict = self.RSSM.get_stoch_state_dict(posterior)
        input_decoder1 = torch.cat([deter_dict['s1'], stoch_dict['s1']], dim=-1)
        input_decoder12 = torch.cat([deter_dict['s1'], deter_dict['s2'], stoch_dict['s1'], stoch_dict['s2']], dim=-1)
        input_decoder3 = torch.cat([deter_dict['s3'], stoch_dict['s3']], dim=-1)
        input_decoder4 = torch.cat([deter_dict['s4'], stoch_dict['s4']], dim=-1)
        input_decoder34 = torch.cat([deter_dict['s3'], deter_dict['s4'], stoch_dict['s3'], stoch_dict['s4']], dim=-1)

        obs_dist_1 = self.TestObsDecoder1(input_decoder1[:-1].detach())
        obs_dist_12 = self.TestObsDecoder12(input_decoder12[:-1].detach())
        obs_dist_3 = self.TestObsDecoder3(input_decoder3[:-1].detach())
        obs_dist_4 = self.TestObsDecoder4(input_decoder4[:-1].detach())
        obs_dist_34 = self.TestObsDecoder34(input_decoder34[:-1].detach())
        obs_loss_1, obs_mse_loss_1 = self._obs_loss(obs_dist_1, obs[:-1])
        obs_loss_12, obs_mse_loss_12 = self._obs_loss(obs_dist_12, obs[:-1])
        obs_loss_3, obs_mse_loss_3 = self._obs_loss(obs_dist_3, obs[:-1])
        obs_loss_4, obs_mse_loss_4 = self._obs_loss(obs_dist_4, obs[:-1])
        obs_loss_34, obs_mse_loss_34 = self._obs_loss(obs_dist_34, obs[:-1])
        total_test_obs_loss = obs_loss_1 + obs_loss_12 + obs_loss_3 + obs_loss_4 + obs_loss_34
        self.TestObsOptimizer.zero_grad()
        total_test_obs_loss.backward()
        self.TestObsOptimizer.step()
        return obs_mse_loss_1, obs_mse_loss_12, obs_mse_loss_3, obs_mse_loss_4, obs_mse_loss_34

    # This function should only be used for testing
    def train_batch_test_decoder(self, train_metrics, train_policy=True):
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        obs_mse_1 = []
        obs_mse_12 = []
        obs_mse_3 = []
        obs_mse_4 = []
        obs_mse_34 = []

        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            uint8_flag = True if obs.dtype == np.uint8 else False
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
            if uint8_flag:
                obs = obs.div(255).sub_(0.5)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            nonterms = torch.tensor(1 - terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            prior, posterior = self._get_prior_posterior(obs, actions, nonterms)
            obs_mse_loss_1, obs_mse_loss_12, obs_mse_loss_3, obs_mse_loss_4, obs_mse_loss_34 = self.train_test_decoder_one_step(
                posterior, obs)
            obs_mse_1.append(obs_mse_loss_1.item())
            obs_mse_12.append(obs_mse_loss_12.item())
            obs_mse_3.append(obs_mse_loss_3.item())
            obs_mse_4.append(obs_mse_loss_4.item())
            obs_mse_34.append(obs_mse_loss_34.item())
        train_metrics['obs_mse_loss_1'] = np.mean(obs_mse_1)
        train_metrics['obs_mse_loss_12'] = np.mean(obs_mse_12)
        train_metrics['obs_mse_loss_3'] = np.mean(obs_mse_3)
        train_metrics['obs_mse_loss_4'] = np.mean(obs_mse_4)
        train_metrics['obs_mse_loss_34'] = np.mean(obs_mse_34)
        self.train_steps += 1
        return train_metrics

    def save_test_decoder(self, save_path):
        save_dict = {
            "TestObsDecoder1": self.TestObsDecoder1.state_dict(),
            "TestObsDecoder12": self.TestObsDecoder12.state_dict(),
            "TestObsDecoder3": self.TestObsDecoder3.state_dict(),
            "TestObsDecoder4": self.TestObsDecoder4.state_dict(),
            "TestObsDecoder34": self.TestObsDecoder34.state_dict(),
        }
        torch.save(save_dict, save_path)
