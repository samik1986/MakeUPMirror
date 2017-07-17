import os
import numpy as np
import Image
from scipy import misc
import glob

mu_ip = []
nm_ip = []
mu_op = []

mu = np.load('YMU_MU.npy')
mu_s1 = mu[0:151,]
mu_s2 = mu[151:302,]
nm = np.load('YMU_NM.npy')
nm_s1 = nm[0:151,]
nm_s2 = nm[151:302,]

np.save('YMU_MU_S1.npy',mu_s1)
np.save('YMU_MU_S2.npy',mu_s2)
np.save('YMU_NM_S1.npy',nm_s1)
np.save('YMU_NM_S2.npy',nm_s2)

for i in range(np.size(nm_s1,0)):
    for j in range(np.size(mu_s1,0)):
        tempMu = mu_s1[j,:,:,:]
        mu_ip.append(tempMu)
        tempNm = nm_s1[i,:,:,:]
        nm_ip.append(tempNm)
        tempMuOp = mu_s1[i,:,:,:]
        mu_op.append(tempMuOp)
    for j in range(np.size(mu_s2,0)):
        tempMu = mu_s2[j,:,:,:]
        mu_ip.append(tempMu)
        tempNm = nm_s1[i,:,:,:]
        nm_ip.append(tempNm)
        tempMuOp = mu_s2[i,:,:,:]
        mu_op.append(tempMuOp)

for i in range(np.size(nm_s2,0)):
    for j in range(np.size(mu_s1,0)):
        tempMu = mu_s1[j,:,:,:]
        mu_ip.append(tempMu)
        tempNm = nm_s1[i,:,:,:]
        nm_ip.append(tempNm)
        tempMuOp = mu_s1[i,:,:,:]
        mu_op.append(tempMuOp)
    for j in range(np.size(mu_s2,0)):
        tempMu = mu_s2[j,:,:,:]
        mu_ip.append(tempMu)
        tempNm = nm_s2[i,:,:,:]
        nm_ip.append(tempNm)
        tempMuOp = mu_s2[i,:,:,:]
        mu_op.append(tempMuOp)

mu_ip = np.asarray(mu_ip)
nm_ip = np.asarray(nm_ip)
mu_op = np.asarray(mu_op)

np.save('MU_IP.npy',mu_ip)
np.save('MU_OP.npy',mu_op)
np.save('NM_IP.npy',nm_ip)