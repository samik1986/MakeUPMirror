import numpy as np
import keras
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape
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


# Generate dummy data
# x_train = np.random.random((100, 100, 100, 3))
# # x_train = np.zeros(100,100,100,3)
# y_train = keras.utils.to_categorical(
#     np.random.randint(10, size=(100, 1)), num_classes=10)
# y_aux_train = keras.utils.to_categorical(
#     np.random.randint(2, size=(100, 1)), num_classes=2)
# # y_aux_train = np.zeros((100, 100, 100, 3))
#
# x_test = np.random.random((2, 100, 100, 3))
# y_test = keras.utils.to_categorical(
#     np.random.randint(10, size=(2, 1)), num_classes=10)
# x1_test = np.zeros((2, 100, 100, 3))
# y_aux_test = keras.utils.to_categorical(
#     np.random.randint(2, size=(2, 1)), num_classes=2)

x_train = np.load('ae_gxTrain.npy')
# y_train = np.load('gyTrain.npy')
x_aux_train = np.load('ae_pxTrain.npy')
# y_aux_train =np.load('vyTrain.npy')

# print y_train

input_dim =  [100,100,3]

sess = tf.InteractiveSession()

def hellinger_distance(y_true,y_pred):
    y_true = K.clip()

def create_network(input_dim):

    # input_source = Input(input_dim)
    input_target = Input(input_dim)

    #---Autoencoder----

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_target)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # print encoded
    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid')(x)
    # print K.int_shape(decoded)
    # print input_source


    final = Model(inputs=input_target,
                  outputs=decoded)

    return final


model = create_network([100, 100, 3])
print(model.summary())
plot_model(model, to_file='model.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,
#           write_graph=True, write_images=True)
#
# tbCallBack.set_model(model)
#
# keras.callbacks.TensorBoard(sess)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer='adadelta')

model.fit(x_aux_train,
          x_train,
          batch_size=150, epochs=5,
          callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# score = model.evaluate([x_test, x1_test],
#                        [y_test, y_aux_test],
#                        batch_size=20)


# print score

