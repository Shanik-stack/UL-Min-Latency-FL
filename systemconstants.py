import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
""" Power constraints and precoder not yet implemented"""

class ChannelConstants:
    def __init__(self, system_constants: dict = {}):
        
        self.K, self.NR, self.NT, self.n, self.L,self.T, self.SNR_DB, self.desired_CNR, self.fs, self.B = system_constants["K"], system_constants["NR"], system_constants["NT"], system_constants["n"], system_constants["L"], system_constants["T"], system_constants["snr_db"], system_constants["desired_CNR"], system_constants["fs"], system_constants["B"]
        
        self.H = []
        self.N = []
        
        # Generate K-user channel realizations
        for k in range(self.K):
            if(len(self.desired_CNR) == 0):
                H_real = np.random.normal(0, 1/np.sqrt(2), (self.L[k], self.NR[k], self.NT[k]))
                H_imag = np.random.normal(0, 1/np.sqrt(2), (self.L[k], self.NR[k], self.NT[k]))

                self.H.append(H_real + 1j * H_imag) # shape: (K,L,Nr,Nt)
            
            else:
                """ To generate channels with CNR imabalance in the future"""
                pass


class UserConstants:
    def __init__(self, system_constants: dict = {}):
        
        self.K, self.Pt, self.NR, self.NT, self.n, self.L,self.T, = system_constants["K"], system_constants["Pt"], system_constants["NR"], system_constants["NT"], system_constants["n"], system_constants["L"], system_constants["T"]
        self.dk = system_constants["dk"] #No. of tranmsission stream
        
        #Define x_k and F_k (user message & precoder)
        self.X = []
        self.F = []
        for k in range(self.K):
            X_real_k = np.random.normal(0, 1/np.sqrt(2), (self.L[k], self.dk[k], self.T[k]))
            X_imag_k = np.random.normal(0, 1/np.sqrt(2), (self.L[k], self.dk[k], self.T[k]))
            
            F_real_k = np.random.normal(0, 1/np.sqrt(2), (self.L[k], self.NT[k], self.dk[k]))
            F_imag_k = np.random.normal(0, 1/np.sqrt(2), (self.L[k], self.NT[k], self.dk[k]))
            
            X_k = X_real_k + 1j*X_imag_k
            F_k = F_real_k + 1j*F_imag_k
            
            
            # Blockwise Power Constraint
            for l in range(self.L[k]):
                norm_factor = np.linalg.norm(F_k[l,:,:], 'fro')
                F_k[l,:,:] = np.sqrt(self.Pt[k]) * F_k[l,:,:] / norm_factor  # satisfies tr(F F^H) = Pt
           
            # User Power Constraint
            # total_power = np.sum([np.linalg.norm(F_k[l,:,:], 'fro')**2 for l in range(self.L[k])])
            # F_k = np.sqrt(self.Pt[k]) * F_k / np.sqrt(total_power)
            
            self.X.append(X_k) # X shape: (K,L,dk,T)
            self.F.append(F_k) # F shape: (K,L,NT,dk)



if __name__ == "__main__":
    pass
    # ch = ChannelConstants(N = [8, 16, 32])
    # usr = UserConstants(N = [8, 16, 32])
    # print(usr.X)
    # ch.plot_magnitude_per_block()
    