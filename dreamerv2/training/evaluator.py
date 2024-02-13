import numpy as np
import torch
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.utils.visualize import Visualizer


class Evaluator(object):
    '''
    used this only for minigrid envs
    '''

    def __init__(
            self,
            config,
            device,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.visualizer = Visualizer(config)

    def load_model(self, config, model_path):
        saved_dict = torch.load(model_path)
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size * class_size
        else:
            raise ValueError('rssm type not supported')

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size

        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device).eval()
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(
                self.device).eval()
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()

        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor,
                                               config.expl).to(self.device).eval()
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type,
                         config.rssm_info).to(self.device).eval()

        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])

    def eval_agent(self, env, RSSM, ObsEncoder, ObsDecoder, ActionModel, train_step):
        eval_episode = self.config.eval_episode
        eval_scores = []
        for e in range(eval_episode):
            (obs, _), score = env.reset(), 0
            done = False
            prev_rssmstate = RSSM._init_rssm_state(1)
            prev_action = torch.zeros(1, self.action_size).to(self.device)
            video_frames_dict = {'poster': []}
            while not done:
                with torch.no_grad():
                    embed = ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    if e < 2:
                        self.visualizer.collect_frames((torch.tensor(obs, dtype=torch.float32)), posterior_rssm_state,
                                                       ObsDecoder, video_frames_dict)
                    model_state = RSSM.get_model_state(posterior_rssm_state)
                    action, _ = ActionModel(model_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                next_obs, rew, done, timeout, _ = env.step(action.squeeze(0).cpu().numpy())
                if self.config.eval_render:
                    env.render()
                score += rew
                obs = next_obs
            if e < 2:
                self.visualizer.output_video(train_step, e, video_frames_dict)
            eval_scores.append(score)
        print('average evaluation score for model at ' + ' = ' + str(np.mean(eval_scores)))
        env.close()
        return np.mean(eval_scores)
