from dreamerv2.utils.wrapper import GymMinAtar
import numpy as np
from matplotlib import pyplot as plt
import os
import platform


def is_convertible(value):
    if isinstance(value, (str, int)):
        try:
            int(value)
            return True
        except ValueError:
            return False
    else:
        return False


image_dir = "/Users/frank/Desktop/1"

if __name__ == '__main__':
    # env = GymMinAtar(
    #     'space_invaders',
    #     obs_type='pixel',
    #     # noise_type="images",
    #     # resource_files=image_files
    #     noise_type="videos",
    #     resource_dir=image_dir
    # )
    # obs, info = env.reset()
    #
    # while True:
    #     rgb_array = env.render('rgb_array')
    #     plt.imshow(rgb_array)
    #     plt.show()
    #
    #     action = input('Enter action: ')
    #     while not is_convertible(action) or int(action) not in range(6):
    #         action = input('Enter action: ')
    #     obs, reward, done, timeout, info = env.step(int(action))
    #     print(reward)
    #     if done:
    #         break

    # print((73984 / 4 / 16) ** 0.5)
    #
    #
    # def conv_out(h_in, padding, kernel_size, stride):
    #     return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)
    #
    #
    # def output_padding(h_in, conv_out, padding, kernel_size, stride):
    #     return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1
    #
    #
    # def conv_out_shape(h_in, padding, kernel_size, stride):
    #     return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)
    #
    #
    # def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    #     return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
    #
    #
    # output_shape = (3, 40, 40)
    # k = 5
    # conv1_shape = conv_out_shape(output_shape[1:], 0, k, 2)
    # conv2_shape = conv_out_shape(conv1_shape, 0, k, 2)
    # conv3_shape = conv_out_shape(conv2_shape, 0, k, 2)
    #
    # print(conv3_shape)
    # print(24* 100* 3* 40* 40)
    #
    # print(platform.system())
    # print(platform.platform())
    # import sys
    #
    # a = np.empty((50,50, 3, 40, 40), dtype=np.uint8)
    #
    # print(sys.getsizeof(a) / (1024 ** 3))
    # input()

    print(268*6)