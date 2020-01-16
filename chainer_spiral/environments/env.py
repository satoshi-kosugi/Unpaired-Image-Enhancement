import sys

import gym
import numpy as np
from gym import spaces
from .edit_photo import PhotoEditor, edit_demo
import cv2
import random
import logging
import os

DATASET_DIR = "./fivek_dataset/"
TARGET_DIR = "expertC/"
ORIGINAL_DIR = "original/"

class PhotoEnhancementEnv(gym.Env):
    action_space = None
    observation_space = None
    reward_range = None
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }
    def __init__(self,
                 batch_size,
                 logger=None,
                 imsize=512,
                 max_episode_steps=1):
        super().__init__()


        self.tags = {'max_episode_steps': max_episode_steps}
        self.logger = logger or logging.getLogger(__name__)
        self.imsize = imsize
        self.batch_size = batch_size

        try:
            self.file_names
        except:
            self.file_names = []
            with open(os.path.join(DATASET_DIR, "trainSource.txt")) as f:
                s = f.read()
            self.file_names.extend(s.split("\n")[:-1])
            self.file_names = \
                list(map(lambda x: os.path.join(DATASET_DIR, ORIGINAL_DIR, x), self.file_names))

        self.photo_editor = PhotoEditor()
        self.num_parameters = self.photo_editor.num_parameters


        # action space
        self.action_space = spaces.Dict({
            'parameters':
            spaces.Box(low=-1.0, high=1.0,
                            shape=(self.batch_size, self.num_parameters), dtype=np.float32),
        })

        # observation space
        self.observation_space = spaces.Dict({
            'image':
            spaces.Box(low=0,
                       high=255,
                       shape=(self.batch_size, self.imsize, self.imsize, 3),
                       dtype=np.uint8)
        })

        # reset canvas and set current position of the pen
        self.reset()


    def reset(self):
        self.logger.debug('reset the drawn picture')

        self.original_images = []
        self.editted_images = []

        for i in range(self.batch_size):
            original_image = cv2.imread(random.choice(self.file_names))
            original_image = cv2.resize(original_image, (64, 64)) / 255.0
            if random.randint(0, 1) == 0:
                original_image = original_image[:, ::-1, :]
            editted_image = original_image.copy()
            self.original_images.append(original_image)
            self.editted_images.append(editted_image)

        ob = {
            'images': self._get_rgb_array()
        }
        return ob

    def step(self, action):
        parameters_space = self.action_space.spaces['parameters']
        clipped_action = np.clip(action['parameters'] / 1.0, parameters_space.low, parameters_space.high)

        for i in range(self.batch_size):
            self.editted_images[i] = self.photo_editor(self.original_images[i].copy(), clipped_action[i])

        reward = 0.0
        done = False

        ob = {
            'images': self._get_rgb_array()
        }
        return ob, reward, done, {}

    def render(self, mode='human'):
        """ render the current drawn picture image for human """
        if mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self._get_rgb_array())

        elif mode == 'rgb_array':
            return self._get_rgb_array()
        else:
            raise NotImplementedError

    def _get_rgb_array(self, cut=True):
        """ render the current canvas as a rgb array
        """
        rgb_array = np.zeros((self.batch_size, self.imsize, self.imsize, 3), dtype=np.uint8)

        for i in range(self.batch_size):
            shape = self.original_images[i].shape
            rgb_array[i, :shape[0], :shape[1], :] = \
                        (self.editted_images[i][:, :, ::-1] * 255).astype(np.uint8)
        return rgb_array

    def calc_mse(self):
        return ((np.array(self.original_images) - np.array(self.editted_images)) ** 2).mean(axis=(1,2,3))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        # TODO: implement here
        pass


class PhotoEnhancementEnvTest(PhotoEnhancementEnv):

    def __init__(self,
                  batch_size,
                  logger=None,
                  imsize=512,
                  max_episode_steps=1):
        with open(os.path.join(DATASET_DIR, "test.txt")) as f:
            s = f.read()
        self.file_names = s.split("\n")[:-1]
        self.file_names = \
            list(map(lambda x: os.path.join(DATASET_DIR, ORIGINAL_DIR, x), self.file_names))

        super().__init__(batch_size=batch_size,
                         logger=logger,
                         imsize=imsize,
                         max_episode_steps=max_episode_steps)

    def set_result_dir(self, result_dir):
        self.RESULT_DIR = result_dir

    def reset(self):
        self.original_images = []
        self.editted_images = []
        self.original_original_images = []
        self.original_editted_images = []

        for i in range(self.batch_size):
            original_image = cv2.imread(self.file_names[i])
            self.original_original_images.append(original_image.copy() / 255.0)
            original_image = cv2.resize(original_image, (64, 64)) / 255.0
            editted_image = original_image.copy()
            self.original_images.append(original_image)
            self.editted_images.append(editted_image)

        self.done = False
        self.steps = 0

        ob = {
            'images': self._get_rgb_array()
        }
        return ob

    def step(self, action):
        parameters_space = self.action_space.spaces['parameters']
        clipped_action = np.clip(action['parameters'] / 1.0, parameters_space.low, parameters_space.high)

        for i in range(self.batch_size):
            self.editted_images[i] = self.photo_editor(self.original_images[i].copy(), clipped_action[i])
            self.original_editted_images.append(self.photo_editor(self.original_original_images[i].copy(), clipped_action[i]))

        self.steps += 1
        done = self._is_done()
        reward = 0.0

        ob = {
            'images': self._get_rgb_array()
        }
        return ob, reward, done, {}


    def _is_done(self):
        if self.steps >= 1:
            assert self.RESULT_DIR != "", "error: specify the result directory"
            if not os.path.exists(self.RESULT_DIR):
                os.mkdir(self.RESULT_DIR)

            for i in range(self.batch_size):
                cv2.imwrite(os.path.join(self.RESULT_DIR, os.path.basename(self.file_names[i])),
                            (self.original_editted_images[i] * 255).astype(np.uint8))

            self.file_names = self.file_names[self.batch_size:]
            return True
        else:
            return False


class PhotoEnhancementEnvDemo(PhotoEnhancementEnvTest):

    def __init__(self,
                  batch_size=1,
                  logger=None,
                  imsize=512,
                  max_episode_steps=1,
                  file_name=None):
        self.file_names = [file_name]
        super(PhotoEnhancementEnvTest, self).__init__(batch_size=batch_size,
                         logger=logger,
                         imsize=imsize,
                         max_episode_steps=max_episode_steps)

    def step(self, action):
        parameters_space = self.action_space.spaces['parameters']
        clipped_action = np.clip(action['parameters'] / 1.0, parameters_space.low, parameters_space.high)

        # for i in range(self.batch_size):
        #     self.editted_images[i] = self.photo_editor(self.original_images[i].copy(), clipped_action[i])
        #     self.original_editted_images.append(self.photo_editor(self.original_original_images[i].copy(), clipped_action[i]))

        self.steps += 1
        done = self._is_done()
        reward = 0.0

        if done:
            for i in range(self.batch_size):
                edit_demo(self.original_original_images[i] * 255, clipped_action[i])

        ob = {
            'images': self._get_rgb_array()
        }
        return ob, reward, done, {}

    def _is_done(self):
        if self.steps >= 1:
            return True
        else:
            return False
