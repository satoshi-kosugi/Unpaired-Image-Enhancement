import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(x):
    """ function to preprocess image from observation """
    x = x.astype(np.float32) / 255.0
    x = x.transpose((0, 3, 1, 2))
    return x


def preprocess_obs(obs, imsize):
    """ function to preprocess observation from env """
    c = obs['images']

    # image
    c = preprocess_image(c)

    return c


def pack_action(act, L_stages):
    """ returns an action dictionary to environment """
    parameters = np.zeros((len(act[0]), len(act)), dtype=np.float32)
    for i in range(len(act[0])):
        for j in range(len(act)):
            parameters[i, j] = act[j][i].data

    parameters = (parameters - (L_stages-1)/2) / ((L_stages-1)/2)

    return {
        'parameters': parameters,
    }


def compute_auxiliary_reward(past_reward, past_act, n_episode, max_episode_steps):
    """ returns auxiliary rewards for drawing history """

    return past_reward


class ObservationSaver(object):
    """ Helper class to take snapshots of the observations during training process """

    def __init__(self, outdir, rollout_n, imsize):
        self.outdir = outdir
        self.rollout_n = rollout_n
        self.imsize = imsize

        # create directory to save png files
        if self.outdir is not None:
            self.target_dir = os.path.join(self.outdir, 'final_obs')
            if not os.path.exists(self.target_dir):
                os.mkdir(self.target_dir)

        # init figure
        self.fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(self.rollout_n//4, 4 * 2)
        self.ims_real, self.ims_fake = [], []
        for n in range(self.rollout_n):
            ax = plt.subplot(gs[n//4, n%4])
            self.ims_fake.append(
                ax.imshow(np.zeros((self.imsize, self.imsize, 3)),
                          vmin=0,
                          vmax=1))
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 0:
                ax.set_title('Fake data')

            ax = plt.subplot(gs[n//4, n%4+4])
            self.ims_real.append(
                ax.imshow(np.zeros((self.imsize, self.imsize, 3)),
                          vmin=0,
                          vmax=1))
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 0:
                ax.set_title('Real data')

    def save(self, fake_data, real_data, update_n):
        """ save figure of observations (drawn by agent) and real data """
        for n in range(self.rollout_n):
            self.ims_fake[n].set_data(fake_data[n].transpose(1, 2, 0))
            self.ims_real[n].set_data(real_data[n].data[0].transpose(1, 2, 0))
        self.fig.suptitle(f"Update = {update_n}")
        savename = os.path.join(self.target_dir, f"obs_update_{update_n}.png")
        plt.savefig(savename)
