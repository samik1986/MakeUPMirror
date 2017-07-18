import numpy as np
import keras
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape, BatchNormalization
from keras.layers.merge import add
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D,concatenate
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
from keras.callbacks import TensorBoard, ModelCheckpoint
from LRN import LRN
from custom_merge import Discrim


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

# print y_train
mu_ip = np.load('VMU/MU_IP_S.npy')
nm_ip = np.load('VMU/NM_IP_S.npy')
mu_op = np.load('VMU/MU_OP_S.npy')

mu_ip = mu_ip.astype('float32') / 255.
nm_ip = nm_ip.astype('float32') / 255.
mu_op = mu_op.astype('float32') / 255.

input_dim =  [100,100,3]

# sess = tf.InteractiveSession()

def create_network(input_dim):

    #---Inputs--------
    input_mu = Input(input_dim)
    input_nm = Input(input_dim)

    #---Make-up Path----

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_mu)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = LRN()(x)
    x_end = BatchNormalization()(x)
    print x_end

    #---No Make-up Path------------

    y = Conv2D(64, (3, 3), activation='relu', padding='same')(input_nm)
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same')(y)
    y_end = Conv2D(512, (3, 3), activation='relu', padding='same')(y)

    print y_end

    #---Common Path----------

    comb = concatenate([x_end,y_end],axis=-1)
    print comb

    z = Conv2D(512, (3, 3), activation='relu', padding='same')(comb)
    # z = Flatten()(z)
    # z = Dense(512*100*100,activation='relu')(z)
    # z = Dense(4096, activation='relu')(z)
    z = Dropout(0.5)(z)
    # z = Dense(4096,activation='relu')(z)
    # z = Dense(512*100*100, activation='relu')(z)
    # z = Reshape((100,100,512))(z)
    z = Conv2D(512, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(256, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(128, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(32, (3, 3), activation='relu', padding='same')(z)

    op = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(z)

    print op

    final = Model(inputs=[input_mu, input_nm],
                  outputs=op)

    return final


model = create_network(input_dim)
print(model.summary())
# plot_model(model, to_file='model.png')


# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
adadelta = keras.optimizers.adadelta(lr=0.001,decay=1e-5)
model.compile(loss='mse',
              optimizer='adadelta')
filepath="models/ckpt{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


model.fit([mu_ip,nm_ip],
          mu_op,
          batch_size=20, epochs=10000,
          callbacks=[TensorBoard(log_dir='models/'),checkpoint])



# score = model.evaluate([mu_ip, nm_ip],
#                        mu_op,
#                        batch_size=20)


# print score

