import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class ResidualBlock(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super(ResidualBlock, self).__init__(
            c1=L.Convolution2D(n_in, n_out, ksize, stride, 1, w),
            c2=L.Convolution2D(n_out, n_out, ksize, 1, 1, w),
            bn1=L.BatchNormalization(n_out),
            bn2=L.BatchNormalization(n_out)
        )

    def __call__(self, x, test):
        h = F.relu(self.bn1(self.c1(x), test=test))
        h = self.bn2(self.c2(h), test=test)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=test)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return h + x

class FastStyleNet(chainer.Chain):
    def __init__(self):
        super(FastStyleNet, self).__init__(
            c1=L.Convolution2D(3, 32, 9, stride=1),
            c2=L.Convolution2D(32, 64, 3, stride=2),
            c3=L.Convolution2D(64, 128, 3,stride=2),
            res1=ResidualBlock(128, 128),
            res2=ResidualBlock(128, 128),
            res3=ResidualBlock(128, 128),
            res4=ResidualBlock(128, 128),
            res5=ResidualBlock(128, 128),
            dc1=L.Deconvolution2D(128, 64, ksize=3),
            dc2=L.Deconvolution2D(3, 32, ksize=3),
            dc3=L.Deconvolution2D(32, 3, ksize=9),
            bn1=L.BatchNormalization(32),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(64),
            bn5=L.BatchNormalization(32),
        )

    def __call__(self, x, test=False):
        h = self.bn1(F.relu(self.c1(x)), test=test)
        h = self.bn2(F.relu(self.c2(h)), test=test)
        h = self.bn3(F.relu(self.c3(h)), test=test)
        h = self.res1(h),
        h = self.res2(h),
        h = self.res3(h),
        h = self.res4(h),
        h = self.res5(h),
        h = self.bh4(F.relu(self.dc1(h)), test=test),
        h = self.bh5(F.relu(self.dc2(h)), test=test),
        l = self.dc3(h)
        return l
