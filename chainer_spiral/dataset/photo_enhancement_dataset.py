import gzip

import chainer
import cv2
import numpy as np
import os
import random


class PhotoEnhancementDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_type="Target"):
        self.DATASET_DIR = "./fivek_dataset/"
        if data_type == "Target":
            self.IMAGE_DIR = "expertC/"
        else:
            self.IMAGE_DIR = "original/"

        self._paths = []
        self.load_data(data_type)

    def __len__(self):
        return len(self._paths)

    def get_example(self):
        path = random.choice(self._paths)
        image = cv2.imread(path)[:, :, ::-1]
        shape_ = image.shape
        ratio = 64.0 / max(shape_[0], shape_[1])
        image = cv2.resize(image, (64, 64)) / 255.0
        shape = image.shape
        image_ = np.zeros((1, 3, 64, 64), dtype=np.float32)

        if random.randint(0, 1) == 0:
            image_[0, :, :shape[0], :shape[1]] = image.transpose(2, 0, 1)
        else:
            image_[0, :, :shape[0], :shape[1]] = image[:, ::-1, :].transpose(2, 0, 1)

        return chainer.Variable(image_)

    def load_data(self, data_type):
        with open(os.path.join(self.DATASET_DIR, "train" + data_type + ".txt")) as f:
            s = f.read()
            self._paths.extend(\
                map(lambda x: os.path.join(self.DATASET_DIR, self.IMAGE_DIR, x),
                s.split("\n")[:-1]))
