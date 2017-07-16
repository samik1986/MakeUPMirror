import numpy as np
import keras
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape, BatchNormalization
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard
import theano.tensor as T



class LRN(Layer):
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2  # half the local region
        input_sqr = T.sqr(x)  # square the input
        extra_channels = T.alloc(0., b, ch + 2 * half_n, r,
                                 c)  # make an empty tensor with zero pads along channel dimension
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n + ch, :, :],
                                    input_sqr)  # set the center to be the squared input
        scale = self.k  # offset for the scale
        norm_alpha = self.alpha / self.n  # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))