import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model, Model, save_model
import Image


x_test = np.load('VMU/MU_IP_S.npy')
y_test = np.load('VMU/NM_IP_S.npy')
z_test = np.load('VMU/MU_OP_S.npy')

x_test = x_test.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.
z_test = z_test.astype('float32') / 255.


n = 6000

p1 = p2 = []

plt.imshow(x_test[n])
plt.show()

plt.imshow(y_test[n])
plt.show()

plt.imshow(z_test[n])
plt.show()



# x_test = x_test.astype('float32')
# x_test = (x_test-x_test.min())/(x_test.max()-x_test.min())
# y_test = y_test.astype('float32')
# y_test = (y_test-y_test.min())/(y_test.max()-y_test.min())
#
# print x_test[106]
#
# x_train1 = np.load('trainImagesIITM.npy')
# x_train1 = x_train1.astype('float32')
# x_train1 = (x_train1-x_train1.min())/(x_train1.max()-x_train1.min())
#
model = load_model('models/ckpt12.hdf5')
#

# p1[1,:,:,:]=x_test[n,]
# print p1
# p2[1,:,:,:]=y_test[n,]

decoded_imgs = model.predict([x_test[n:n+1,],y_test[n:n+1]])
x_dec = decoded_imgs*255.
# print x_dec[1]
# print decoded_imgs


np.save('dec_images',decoded_imgs)


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
plt.imshow(x_dec[0,].astype('uint8'))
# plt.gray()
plt.show()
# plt.imshow(x_train1[106])
# plt.show()
#
# plt.imshow(x_train[106])
# plt.show()