import argparse
import logging
import os
# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import chainer
import yaml
from chainerrl import experiments, misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay

from chainer_spiral.agents import SPIRAL, SpiralStepHook
from chainer_spiral.dataset import PhotoEnhancementDataset
from chainer_spiral.environments import PhotoEnhancementEnvTest
from chainer_spiral.models import (SpiralDiscriminator, SpiralModel)
from chainer_spiral.utils.arg_utils import print_args



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('outdir', type=str, help='directory to put training log')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger_level', type=int, default=logging.INFO)
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    print_args(args)

    # init a logger
    logging.basicConfig(level=args.logger_level)

    # load yaml config file
    with open(args.config) as f:
        config = yaml.load(f)

    # set random seed
    misc.set_random_seed(config['seed'])


    # define func to create env, target data sampler, and models
    if config['problem'] == 'photo_enhancement':
        def make_env(process_idx, test):
            assert test, "error: test should be True"
            env = PhotoEnhancementEnvTest(batch_size=config['rollout_n'],
                             max_episode_steps=config['max_episode_steps'],
                             imsize=config['imsize'])
            env.set_result_dir(args.result_dir)
            return env

        sample_env = make_env(0, True)

        gen = SpiralModel(config['imsize'], sample_env.num_parameters, config['L_stages'], config['conditional'])
        dis = SpiralDiscriminator(config['imsize'], config['conditional'])

        dataset = PhotoEnhancementDataset()

    else:
        raise NotImplementedError()

    # initialize optimizers
    gen_opt = chainer.optimizers.Adam(alpha=config['lr'], beta1=0.5)
    dis_opt = chainer.optimizers.Adam(alpha=config['lr'], beta1=0.5)

    gen_opt.setup(gen)
    dis_opt.setup(dis)

    gen_opt.add_hook(chainer.optimizer.GradientClipping(40))
    dis_opt.add_hook(chainer.optimizer.GradientClipping(40))
    if config['weight_decay'] > 0:
        gen_opt.add_hook(NonbiasWeightDecay(config['weight_decay']))
        dis_opt.add_hook(NonbiasWeightDecay(config['weight_decay']))

    # init an spiral agent
    agent = SPIRAL(generator=gen,
                   discriminator=dis,
                   gen_optimizer=gen_opt,
                   dis_optimizer=dis_opt,
                   dataset=dataset,
                   conditional=config['conditional'],
                   reward_mode=config['reward_mode'],
                   imsize=config['imsize'],
                   max_episode_steps=config['max_episode_steps'],
                   rollout_n=config['rollout_n'],
                   gamma=config['gamma'],
                   alpha=config['alpha'],
                   beta=config['beta'],
                   L_stages=config['L_stages'],
                   U_update=config['U_update'],
                   gp_lambda=config['gp_lambda'],
                   n_save_final_obs_interval=config['n_save_final_obs_interval'],
                   outdir=args.outdir,
                   act_deterministically=True)

    # load from a snapshot
    assert args.load, "error: specify the weight of the model"
    if args.load:
        agent.load(args.load)

    # training mode
    max_episode_len = config['max_episode_steps'] * config['rollout_n']
    steps = config['processes'] * config['n_update'] * max_episode_len

    save_interval = config['processes'] * config['n_save_interval'] * max_episode_len
    eval_interval = config['processes'] * config['n_eval_interval'] * max_episode_len

    step_hook = SpiralStepHook(config['max_episode_steps'], save_interval, args.outdir)

    env = make_env(0, True)

    with chainer.using_config('train', False):
        eval_stats = experiments.evaluator.run_evaluation_episodes(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=25,
            max_episode_len=1)

if __name__ == '__main__':
    main()
