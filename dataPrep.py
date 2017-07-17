import os
import numpy as np
import Image
from scipy import misc
import glob


read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

dsDir = '/home/dell/Desktop/MU_ds_0pool/IMG/YMU/'
muDir = dsDir + 'MU/'
nmDir = dsDir + 'NM/'

mu = []
nm = []
count = 1

for directories_mu in os.listdir(muDir):
    dir_mu = os.path.join(muDir,directories_mu)
    count = 1
    for files in os.listdir(dir_mu):
        s = "%03d.jpg" % count
        filepath = os.path.join(dir_mu,s)
        print filepath
        img = misc.imread(filepath)
        img = misc.imresize(img, [100,100,3],'bicubic')
        mu.append(img)
        trImages = np.asarray(mu)
        count = count+1
np.save('YMU_MU',trImages)


for directories_nm in os.listdir(nmDir):
    dir_nm = os.path.join(nmDir,directories_nm)
    count = 1
    for files in os.listdir(dir_nm):
        s = "%03d.jpg" % count
        filepath = os.path.join(dir_nm,s)
        print filepath
        img = misc.imread(filepath)
        img = misc.imresize(img, [100,100,3],'bicubic')
        nm.append(img)
        trImages = np.asarray(nm)
        count = count+1
np.save('YMU_NM',trImages)


