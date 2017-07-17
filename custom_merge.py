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



class _custom_merge(Layer):
    """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(_custom_merge, self).__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.

        # Arguments
            shape1: tuple or None. Shape of the first tensor
            shape2: tuple or None. Shape of the second tensor

        # Returns
            expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.

        # Raises
            ValueError: if shape1 and shape2 are not compatible for
                element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif len(shape2) == 0:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError('Operands could not be broadcast '
                                     'together with shapes ' +
                                     str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) > 1:
            raise ValueError('Can not merge tensors with different '
                             'batch sizes. Got tensors with shapes : ' +
                             str(input_shape))
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(K.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = K.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = K.expand_dims(x, 1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = K.ndim(x)
                    if x_ndim is None:
                        x_shape = K.shape(x)
                        batch_size = x_shape[0]
                        new_shape = K.concatenate([x_shape[1:], K.expand_dims(batch_size)])
                        x_transposed = K.reshape(x, K.stack([batch_size, K.prod(x_shape[1:])]))
                        x_transposed = K.permute_dimensions(x_transposed, (1, 0))
                        x_transposed = K.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(K.permute_dimensions(x, dims))
                        transposed = True
                    else:
                        # We don't transpose inputs if they are 1D vectors or scalars.
                        reshaped_inputs.append(x)
                y = self._merge_function(reshaped_inputs)
                y_ndim = K.ndim(y)
                if transposed:
                    # If inputs have been transposed, we have to transpose the output too.
                    if y_ndim is None:
                        y_shape = K.shape(y)
                        y_ndim = K.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = K.concatenate([K.expand_dims(batch_size), y_shape[:y_ndim - 1]])
                        y = K.reshape(y, (-1, batch_size))
                        y = K.permute_dimensions(y, (1, 0))
                        y = K.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = K.permute_dimensions(y, dims)
                return y
        else:
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        masks = [K.expand_dims(m, 0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


class Discrim(_custom_merge):
    """Layer that adds a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """

    def find_discriminative(self, input1, input2):
        # for i1 in range(K.int_shape(input2)[3]):
        #     for i2 in range(K.int_shape(input1)[3]):
        #         ip1 = input1[:,:,:,i2]
        #         ip1 = tf.expand_dims(ip1,3)
        #         ip2 = input2[:,:,:,i1]
        #         ip2 = tf.expand_dims(ip2, 3)
        #         patch1 = tf.extract_image_patches(ip2,[1,11,11,1],[1,6,6,1],[1,1,1,1],padding='SAME')
        #         print patch1
        #         for j1 in range(K.int_shape(patch1)[1]):
        #             for j2 in range(K.int_shape(patch1)[2]):
        #                 p1 =patch1[:,j1,j2,:]
        #                 p1 = Reshape((11,11))(p1)
        #                 input1[:,:,:,i2] = Conv2D(1,p1,strides=[3,3],padding='SAME')(ip1)

        mean1 = K.mean(K.mean(K.mean(input1,3,False),2,False),1,False)
        print mean1, '.....'
        output = input1
        print output
        return output

    def _merge_function(self, inputs):
        # output = inputs[0]
        # print "@@@dummy..." , inputs[0]
        # for i in range(1, len(inputs)):
        output = self.find_discriminative(inputs[0], inputs[1])
        return output



def discrim(inputs, **kwargs):
    """Functional interface to the `Add` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the sum of the inputs.

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        added = keras.layers.add([x1, x2])

        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """
    return Discrim(**kwargs)(inputs)