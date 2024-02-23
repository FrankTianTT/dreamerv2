import torch, os, math, warnings
from torch.nn.functional import interpolate
from PIL import Image
from typing import *
import imageio
import numpy as np


class Visualizer(object):
    def __init__(self, config):
        self.gif_dir = config.gif_dir
        self.config = config

        pass

    def obs_to_image(self, obs: torch.Tensor):
        if obs.shape[-3] == 1:
            return obs.expand(*obs.shape[:-3], 3, *obs.shape[-2:])
        else:
            assert obs.shape[-3] % 3 == 0
            return obs.reshape(*obs.shape[:-3], 3, -1, obs.shape[-1])

    def save_image(self, uint8_image: torch.Tensor, fp: str):
        im = Image.fromarray(uint8_image.permute(1, 2, 0).numpy())
        im.save(fp)

    @torch.no_grad()
    def collect_frames(self, obs, rssm_state, ObsDecoder, video_frames_dict):
        # Collect real vs. predicted frames for video
        model_state = torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)
        reconstruction: torch.Tensor = ObsDecoder(model_state).mean

        rgb_array = torch.cat([
            self.obs_to_image(torch.unsqueeze(obs, dim=0).cpu()),
            self.obs_to_image(reconstruction.cpu())
        ], dim=-1)

        rgb_array = interpolate(rgb_array, size=(rgb_array.shape[-2] * 16, rgb_array.shape[-1] * 16)) * 255
        rgb_array = rgb_array.clamp_(0, 255).to(torch.uint8)
        video_frames_dict['poster'].append(rgb_array[0])

    @torch.no_grad()
    def if_collect_frames(self, obs, rssm_state, rssm_state_0, RSSM, ObsDecoder, video_frames_dict):
        # Collect real vs. predicted frames for video
        rssm_state_1234 = rssm_state

        rssm_state_1 = RSSM.replace_with_state_0(rssm_state, rssm_state_0, replace_index=[2, 3, 4])
        rssm_state_2 = RSSM.replace_with_state_0(rssm_state, rssm_state_0, replace_index=[1, 3, 4])
        rssm_state_3 = RSSM.replace_with_state_0(rssm_state, rssm_state_0, replace_index=[1, 2, 4])
        rssm_state_4 = RSSM.replace_with_state_0(rssm_state, rssm_state_0, replace_index=[1, 2, 3])
        rssm_state_5 = RSSM.replace_with_state_0(rssm_state, rssm_state_0, replace_index=[3, 4])

        rssm_state_dict = {"rssm_state_1234": rssm_state_1234, "rssm_state_1": rssm_state_1,
                           "rssm_state_2": rssm_state_2, "rssm_state_3": rssm_state_3, "rssm_state_4": rssm_state_4,
                           "rssm_state_12": rssm_state_5}
        for key, value in rssm_state_dict.items():
            model_state = RSSM.get_model_state(value)
            reconstruction: torch.Tensor = ObsDecoder(model_state).mean

            rgb_array = torch.cat([
                self.obs_to_image(torch.unsqueeze(obs, dim=0).cpu()),
                self.obs_to_image(reconstruction.cpu())
            ], dim=-1)

            rgb_array = interpolate(rgb_array, size=(rgb_array.shape[-2] * 16, rgb_array.shape[-1] * 16)) * 255
            rgb_array = rgb_array.clamp_(0, 255).to(torch.uint8)

            video_frames_dict[key].append(rgb_array[0])

    def write_video(self, frames: List[torch.Tensor], title, path="", fps=20):
        # with imageio.get_writer(os.path.join(path, "%s.mp4" % title), mode='I', fps=fps) as writer:
        with imageio.get_writer(os.path.join(path, "%s.mp4" % title), mode='I', fps=fps) as writer:
            for frame in frames:
                # VideoWrite expects H x W x C in BGR
                writer.append_data(frame.permute(1, 2, 0).numpy())

    @torch.no_grad()
    def output_video(self, step, e, video_frames_dict):
        vis_dir = os.path.join(self.gif_dir, f'{step}')
        os.makedirs(vis_dir, exist_ok=True)
        print(vis_dir)
        for key, value in video_frames_dict.items():
            self.write_video(value, f"episode_{e}_{key}", vis_dir)  # Lossy compression
            self.save_image(
                torch.as_tensor(value[-1]),
                os.path.join(vis_dir, f"episode_{e}_{key}.png"),
            )
        print('Saved visualization.')
