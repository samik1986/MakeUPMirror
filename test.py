import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model, Model, save_model

x_test = np.load('MU_IP.npy')
y_test = np.load('NM_IP.npy')
z_test = np.load('MU_OP.npy')

plt.imshow(z_test[106])
plt.show()


# x_test = x_test.astype('float32')
# # x_test = (x_test-x_test.min())/(x_test.max()-x_test.min())
#
# print x_test[106]
#
# x_train1 = np.load('trainImagesIITM.npy')
# x_train1 = x_train1.astype('float32')
# x_train1 = (x_train1-x_train1.min())/(x_train1.max()-x_train1.min())
#
model = load_model('models/YMU.hdf5')
#
decoded_imgs = model.predict([x_test[106],y_test[106]])
x_dec = decoded_imgs
print np.shape(decoded_imgs)

# np.save('dec_images',decoded_imgs)


# x_dec = np.load('dec_images.npy')
# x_aux_train = np.load('ae_pxTrain.npy')
# x_train = np.load('ae_gxTrain.npy')

# print x_dec[106]
# x_dec1 =x_dec
# # x_dec = x_dec.astype('uint8')
# # x_dec1 = (x_dec+x_dec.min())*(x_dec.max()-x_dec.min())
# x_dec1 = (x_dec1-x_dec1.min())/(x_dec1.max()-x_dec1.min())
# x_dec1 = x_dec1 * 255
# x_dec1 = x_dec1.astype('uint8')
# print x_dec1[106]

# x_dec = x_dec.astype('int8')

# print x_dec[106]

# p = np.zeros([1,100,100,3],dtype='int8')
# decoded_imgs_p = model.predict(p)
#
# decoded_imgs_p = decoded_imgs_p.astype('float32')
# decoded_imgs_p = (decoded_imgs_p-decoded_imgs_p.min())/(decoded_imgs_p.max()-decoded_imgs_p.min())
# decoded_imgs_p = decoded_imgs_p * 255
# decoded_imgs_p = decoded_imgs_p.astype('int8')
#
# plt.imshow(p[0])
# plt.show()
#
# plt.imshow(decoded_imgs_p[0]*255)
# plt.show()

# print x_dec[907]
#
#
#
plt.imshow(decoded_imgs[106])
plt.show()

# plt.imshow(x_train1[106])
# plt.show()
#
# plt.imshow(x_train[106])
# plt.show()