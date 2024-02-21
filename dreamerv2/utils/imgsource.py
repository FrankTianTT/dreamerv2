# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
np.float = float
np.int = int
import cv2
import skvideo.io


def resize_image(image, target_wh):
    h, w = image.shape[:2]
    target_width, target_height = target_wh

    if w * target_height > h * target_width:
        new_w = int(h * target_width / target_height)
        center_x = w // 2
        start_x = max(0, center_x - new_w // 2)
        end_x = min(w, center_x + new_w // 2)
        cropped_image = image[:, start_x:end_x]
    else:
        new_h = int(w * target_height / target_width)
        center_y = h // 2
        start_y = max(0, center_y - new_h // 2)
        end_y = min(h, center_y + new_h // 2)
        cropped_image = image[start_y:end_y, :]

    resized_image = cv2.resize(cropped_image, (target_width, target_height))
    return resized_image


class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """

    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    def get_image(self):
        return np.copy(self.arr) / 255.0


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))

    def get_image(self):
        arr = np.zeros((self.shape[0], self.shape[1], 3))
        arr[:, :] = self._color
        return arr / 255.0


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=50):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.shape = shape
        self.strength = strength

    def get_image(self):
        return np.maximum(np.random.randn(self.shape[0], self.shape[1], 3) * self.strength, 0) / 255.0


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.shape_wh = shape[::-1]
        self.filelist = filelist
        self.reset()

    def reset(self):
        fname = np.random.choice(self.filelist)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = im[:, :, ::-1]
        im = resize_image(im, self.shape_wh)
        self._im = im / 255.0

    def get_image(self):
        return self._im


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.shape_wh = shape[::-1]
        self.filelist = filelist
        self.reset()

    def reset(self):
        fname = np.random.choice(self.filelist)
        self.frames = skvideo.io.vread(fname)
        self.frame_idx = 0

    def get_image(self):
        if self.frame_idx >= self.frames.shape[0]:
            self.reset()
        im = self.frames[self.frame_idx][:, :, ::-1]
        self.frame_idx += 1
        im = im[:, :, ::-1]
        im = resize_image(im, self.shape_wh)
        return im / 255.0
