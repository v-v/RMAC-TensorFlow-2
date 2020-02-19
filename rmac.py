# R-MAC Layer implementation for TensorFlow 2
# copyright (c) 2020 IMATAG
# www.imatag.com
#
# Authors: Vedran Vukotic, Vivien Chappelier
#
# Based on the original implementation in MATLAB

import numpy as np
import tensorflow as tf

class RMAC:
    def __init__(self, shape, levels=3, power=None, overlap=0.4, norm_fm=False, sum_fm=True, verbose=False):
        self.shape = shape
        self.sum_fm = sum_fm
        self.norm = norm_fm
        self.power = power

        # ported from Giorgios' Matlab code
        steps = np.asarray([2, 3, 4, 5, 6, 7])
        B, H, W, D = shape
        w = min([W, H])
        w2 = w // 2 - 1
        b = np.asarray((max(H, W) - w)) / (steps - 1);
        idx = np.argmin(np.abs(((w**2 - w*b)/(w**2))-overlap))

        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx + 1
        elif H > W:
            Hd = idx + 1

        self.regions = []
        for l in range(levels):

            wl = int(2 * w/(l+2));
            wl2 = int(wl / 2 - 1);

            b = 0 if not (l + Wd) else ((W - wl) / (l + Wd))
            cenW = np.asarray(np.floor(wl2 + np.asarray(range(l+Wd+1)) * b), dtype=np.int32) - wl2
            b = 0 if not (l + Hd) else ((H - wl) / (l + Hd))
            cenH = np.asarray(np.floor(wl2 + np.asarray(range(l+Hd+1)) * b), dtype=np.int32) - wl2

            for i in cenH:
                for j in cenW:
                    if i >= W or j >= H:
                        continue
                    ie = i+wl
                    je = j+wl
                    if ie >= W:
                        ie = W
                    if je >= H:
                        je = H
                    if ie - i < 1 or je - j < 1:
                        continue
                    self.regions.append((i,j,ie,je))

        if verbose:
            print('RMAC regions = %s' % self.regions)

    def rmac(self, x):
        y = []
        for r in self.regions:
            x_sliced = x[:, r[1]:r[3], r[0]:r[2], :]
            if self.power is None:
                x_maxed = tf.reduce_max(x_sliced, axis=(1,2))
            else:
                x_maxed = tf.reduce_mean((x_sliced ** self.power), axis=(2,3)) ** (1.0 / self.power)
                x_maxed = tf.pow(tf.reduce_mean((tf.pow(x_sliced, self.power)), axis=(2,3)),(1.0 / self.power))
            y.append(x_maxed)

        y = tf.stack(y, axis=0)
        y = tf.transpose(y, [1,0,2])

        if self.norm:
            y = tf.math.l2_normalize(y, 2)


        if self.sum_fm:
            y = tf.reduce_mean(y, axis=(1))

        return y

