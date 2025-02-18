import wandb
import argparse
import os
import platform
import torch
import numpy as np
import gym
from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction
from dreamerv2.training.config import MinAtarConfig
# from dreamerv2.training.trainer import Trainer
from dreamerv2.training.if_trainer import IFTrainer
from dreamerv2.training.if_evaluator import Evaluator


def main(args):
    wandb.login()
    env_name = args.env
    exp_id = '_'.join(["if", str(args.env), str(args.seed), args.obs_type])

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')  # dir to save learnt models
    gif_dir = os.path.join(result_dir, 'visualization')

    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif args.device == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('using :', device)

    env_kwargs = dict(
        env_name=env_name,
        obs_type=args.obs_type,
        noise_type="videos",
        resource_dir="/home/featurize/.driving_car"
        if platform.platform().startswith('Linux') else "/Users/frank/Desktop/1",
        noise_alpha=args.noise_alpha,
    )
    env = OneHotAction(GymMinAtar(**env_kwargs))
    test_env = OneHotAction(GymMinAtar(**env_kwargs))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    if args.obs_type == 'pixel':
        obs_dtype = np.dtype(np.uint8)
    else:
        obs_dtype = np.dtype(np.float32)
    action_dtype = np.dtype(np.float32)
    batch_size = args.batch_size
    seq_len = args.seq_len

    config = MinAtarConfig(
        algo='if',
        rssm_type=args.rssm_type,
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
        seq_len=seq_len,
        batch_size=batch_size,
        model_dir=model_dir,
        gif_dir=gif_dir
    )

    config_dict = config.__dict__
    trainer = IFTrainer(config, device)
    evaluator = Evaluator(config, device, visualize=True if args.obs_type == 'pixel' else False)

    with wandb.init(project='mastering MinAtar with world models', config=config_dict):
        """training loop"""
        print('...training...')
        train_metrics = {}
        eval_metrics = {}
        trainer.collect_seed_episodes(env)
        (obs, _), score, length = env.reset(), 0, 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = 0
        train_episodes = 0
        best_save_path = os.path.join(model_dir, 'models_best.pth')

        for iter in range(0, trainer.config.train_steps):
            if iter % trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            if iter % trainer.config.slow_target_update == 0:
                trainer.update_target()
            if iter % trainer.config.save_every == 0:
                trainer.save_model(iter)
            if iter % trainer.config.eval_every == 0:
                eval_score, eval_length = evaluator.eval_agent(test_env, trainer.RSSM, trainer.ObsEncoder,
                                                               trainer.ObsDecoder,
                                                               trainer.ActionModel, iter)
                eval_metrics["eval_rewards"] = eval_score
                eval_metrics["eval_length"] = eval_length
                if eval_score > best_mean_score:
                    best_mean_score = eval_score
                    trainer.save_model(iter)
            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device))
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                asr_state = trainer.RSSM.get_asr_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(asr_state)
                action = trainer.ActionModel.add_exploration(action, iter).detach()
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, timeout, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew
            length += 1

            if done:
                train_episodes += 1
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                train_metrics['train_length'] = length
                train_metrics['action_ent'] = np.mean(episode_actor_ent)
                train_metrics['train_steps'] = iter
                if len(eval_metrics) > 0:
                    wandb.log(eval_metrics, step=train_episodes, commit=False)
                    eval_metrics = {}
                wandb.log(train_metrics, step=train_episodes)

                scores.append(score)
                if len(scores) > 100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average > best_mean_score:
                        best_mean_score = current_average
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)

                (obs, _), score, length = env.reset(), 0, 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    '''evaluating probably best model'''
    # evaluator.eval_saved_agent(env, best_save_path)


if __name__ == "__main__":
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="breakout", help='mini atari env name')
    # parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument("--obs_type", type=str, default='pixel')
    parser.add_argument("--rssm_type", type=str, default='discrete', help='rssm type')
    parser.add_argument("--noise_alpha", type=float, default=0.3, help='noise alpha')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    main(args)

# python test/if_mini_atari.py --env breakout --id 0 --seed 0
# python test/if_mini_atari.py --env freeway --id 0 --seed 0
# python test/if_mini_atari.py --env space_invaders --id 0 --seed 0
# python test/if_mini_atari.py --env seaquest --id 0 --seed 0
# python test/if_mini_atari.py --env asterix --id 0 --seed 0