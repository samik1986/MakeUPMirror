import os
import numpy as np
import Image
from scipy import misc
import glob

mu_ip = []
nm_ip = []
mu_op = []

mu_fm = np.load('VMU/VMU_MU_FM.npy')
mu_em = np.load('VMU/VMU_MU_EM.npy')
mu_ls = np.load('VMU/VMU_MU_LS.npy')


# nm = np.load('VMU/VMU_NM_S.npy')
nm = np.load('VMU/VMU_NM_L.npy')


# np.save('YMU_MU_S1.npy',mu_s1)
# np.save('YMU_MU_S2.npy',mu_s2)
# np.save('YMU_NM_S1.npy',nm_s1)
# np.save('YMU_NM_S2.npy',nm_s2)


for i in range(np.size(nm,0)):
    for j in range(np.size(mu_fm,0)):
        tempMu = mu_fm[j,:,:,:]
        mu_ip.append(tempMu)
        tempNm = nm[i,:,:,:]
        nm_ip.append(tempNm)
        tempMuOp = mu_fm[i,:,:,:]
        mu_op.append(tempMuOp)
    for j in range(np.size(mu_fm,0)):
        tempMu = mu_em[j,:,:,:]
        mu_ip.append(tempMu)
        tempNm = nm[i,:,:,:]
        nm_ip.append(tempNm)
        tempMuOp = mu_em[i,:,:,:]
        mu_op.append(tempMuOp)
    for j in range(np.size(mu_fm,0)):
        tempMu = mu_ls[j,:,:,:]
        mu_ip.append(tempMu)
        tempNm = nm[i,:,:,:]
        nm_ip.append(tempNm)
        tempMuOp = mu_ls[i,:,:,:]
        mu_op.append(tempMuOp)

# mu_ip = np.asarray(mu_ip)
# nm_ip = np.asarray(nm_ip)
# mu_op = np.asarray(mu_op)


np.save('MU_IP_L.npy',mu_ip)
np.save('MU_OP_L.npy',mu_op)
np.save('NM_IP_L.npy',nm_ip)