import numpy as np
from System import *
import torch
from test_const import TEST_CONST
import torch.nn as nn
DEVICE = "cuda:0" if torch.cuda.is_available else "cpu"

uplink_system = UplinkSystem(TEST_CONST)
Precoders = []
for k in range(uplink_system.K):
    Lk = uplink_system.L[k]
    NT_k = uplink_system.NT[k]
    dk_k = uplink_system.UserSystem.dk[k]
    precoder = torch.rand(Lk, NT_k,dk_k, device = DEVICE, requires_grad = True, dtype = torch.float32)
    Precoders.append(precoder)

class LagrangianLoss(nn.Module):
    def __init__(self):
        super(LagrangianLoss, self).__init__()
    def forward(self, uplink_system):
        
for epoch in (25):
for k in range(uplink_system.K):
    I = np.identity(uplink_system.UserSystem.NR[k])
    Tk = uplink_system.T[k]
    Hk = uplink_system.H[k] #(L,Nr,Nt)
    Hk_h_transpose = np.conj(Hk).transpose(0,2,1) #(L,Nt,Nr)
    
    Fk = uplink_system.F[k] #(L,NT,dk)
    Fk_h_transpose = np.conj(Fk).transpose(0,2,1) #(L,dk,Nt)
    
    Ak = Hk@Fk@Fk_h_transpose@Hk_h_transpose #(L,Nr,Nr)
    
    sigma2_k = uplink_system.sigma2[k] #scalar
    epsilon_k = uplink_system.epsilon[k] #scalar
    
    Ck = np.log2(np.linalg.det(I + (Ak)/sigma2_k)) # (L,)
    Vk = 0.5*(np.trace(I - (I + (Ak)/sigma2_k)**(-2) , axis1 = 1, axis2 = 2)) * (np.log2(np.e))**2
    R_fblk = Ck - np.sqrt(Vk/Tk) * Q_inv(epsilon_k)
    # R_fblk = 0
    
    uplink_system.C.append(Ck)
    uplink_system.V.append(Vk)
    uplink_system.R_fbl.append(R_fblk)

L = 