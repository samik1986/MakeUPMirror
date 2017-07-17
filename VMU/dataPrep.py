import os
import numpy as np
import Image
from scipy import misc
import glob


read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

dsDir = '/home/dell/Desktop/MU_ds_0pool/IMG/VMU/'
muDir = dsDir + 'MU/'
nmDir = dsDir + 'NM/'

mu_em = []
mu_fm = []
mu_ls = []
nm = []
count = 1

directories_mu = os.listdir(muDir)
dir_mu_fm = os.path.join(muDir,directories_mu[0])
dir_mu_em = os.path.join(muDir,directories_mu[1])
dir_mu_ls = os.path.join(muDir,directories_mu[2])

# directories_nm = os.listdir(nmDir)
# dir_nm = os.path.join(nmDir,directories_nm)


# count = 1
for files in os.listdir(dir_mu_fm):
    print files
    # s = "%03d.jpg" % count
    filepath = os.path.join(dir_mu_fm,files)
    print filepath
    img = misc.imread(filepath)
    img = misc.imresize(img, [100,100,3],'bicubic')
    mu_fm.append(img)
    trImages_fm = np.asarray(mu_fm)

    filepath = os.path.join(dir_mu_em, files)
    print filepath
    img = misc.imread(filepath)
    img = misc.imresize(img, [100, 100, 3], 'bicubic')
    mu_em.append(img)
    trImages_em = np.asarray(mu_em)

    filepath = os.path.join(dir_mu_ls, files)
    print filepath
    img = misc.imread(filepath)
    img = misc.imresize(img, [100, 100, 3], 'bicubic')
    mu_ls.append(img)
    trImages_ls = np.asarray(mu_ls)

    filepath = os.path.join(nmDir, files)
    print filepath
    img = misc.imread(filepath)
    img = misc.imresize(img, [100, 100, 3], 'bicubic')
    nm.append(img)
    trImages_nm = np.asarray(nm)

np.save('VMU/VMU_MU_FM.npy',trImages_fm)
np.save('VMU/VMU_MU_EM.npy',trImages_em)
np.save('VMU/VMU_MU_LS.npy',trImages_ls)
np.save('VMU/VMU_NM_S.npy',trImages_nm)

# for directories_nm in os.listdir(nmDir):
#     dir_nm = os.path.join(nmDir,directories_nm)
#     count = 1
#     for files in os.listdir(dir_nm):
#         s = "%03d.jpg" % count
#         filepath = os.path.join(dir_nm,s)
#         print filepath
#         img = misc.imread(filepath)
#         img = misc.imresize(img, [100,100,3],'bicubic')
#         nm.append(img)
#         trImages = np.asarray(nm)
#         count = count+1
# np.save('YMU_NM',trImages)


