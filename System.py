import numpy as np
import matplotlib.pyplot as plt
from system_const import *
from simulation_params import *
from scipy.stats import norm

RNG = np.random.default_rng(seed)

def Q_inv(x):
    return norm.ppf(1-x)

class UplinkSystem():
    def __init__(self,system_constants: dict):
        self.system_constants = system_constants
        
        self.K = self.system_constants["K"]
        self.Pt, self.NR, self.NT, self.n, self.L,self.T, self.SNR_DB, self.desired_CNR, self.latency, self.B, self.epsilon = self.system_constants["Pt"], self.system_constants["NR"], self.system_constants["NT"],self.system_constants["n"], self.system_constants["L"], self.system_constants["T"], self.system_constants["snr_db"], self.system_constants["desired_CNR"], self.system_constants["latency"], self.system_constants["B"], self.system_constants["epsilon"]
        self.dk = self.system_constants["dk"]
        self.ChannelSystem = ChannelConstants(system_constants = self.system_constants)
        self.UserSystem = UserConstants(system_constants = self.system_constants)     
        
        self.H = self.ChannelSystem.H
        self.N = []
        self.X = self.UserSystem.X
        self.F = self.UserSystem.F
        self.Y = []
        
        self.CNR_linear = []
        self.CNR_db = []
        self.SNR_linear = []
        self.SNR_db = []
        self.sigma2 = [] #noise variance 
        
        for k in range(self.K):
            Y_k = (self.H[k] @ (self.F[k] @ self.X[k])).reshape(-1)
            P_signal_user = np.mean(np.abs(Y_k)**2)
            gamma_lin = 10**(self.SNR_DB[k] / 10)
            sigma2_k = P_signal_user / gamma_lin
            self.sigma2.append(sigma2_k)
            sigma_k = np.sqrt(sigma2_k)

            # SNR
            snr_linear_k = P_signal_user / sigma2_k
            snr_db_k = 10 * np.log10(snr_linear_k)
            self.SNR_linear.append(snr_linear_k)
            self.SNR_db.append(snr_db_k)

            # CNR
            H_power = np.mean([np.linalg.norm(self.H[k][l], 'fro')**2 for l in range(self.H[k].shape[0])])
            cnr_linear = H_power / sigma2_k
            cnr_db = 10 * np.log10(cnr_linear)
            self.CNR_linear.append(cnr_linear)
            self.CNR_db.append(cnr_db)

            # Noise
            N_real = RNG.normal(0, 1/np.sqrt(2), (self.L[k], self.NR[k], self.T[k]))
            N_imag = RNG.normal(0, 1/np.sqrt(2), (self.L[k], self.NR[k], self.T[k]))
            N_k = sigma_k * (N_real + 1j*N_imag)
            self.N.append(N_k)
            
            Y_k_noisy = self.H[k] @ (self.F[k] @ self.X[k]) + N_k
            self.Y.append(Y_k_noisy)
            
        # Using Einsum might be faster
        # self.Y = np.einsum('klij,kljt->klit', self.H, self.X) + self.N
        
        '''Define C_k(F_(k,l), H_k, sigma_k) -> user capcity at coh interval "l" (this is the isntantaneous capacity)
           Define V_k(F_(k,l))
           Define R^*(T_k, F_(k,l)) '''
        self.C = [] #(K,L)
        self.V = [] #(K,L)
        self.R_fbl = [] #(K,L)
        self.usr_avg_C = [] #(K,)
        
        self.update_system(self.F, self.n)
    
    
    def update_system(self, F, n):
        self.F = F
        self.n = n
        self.L = self.n // self.T
        for k in range(self.K):
            Tk = self.T[k]
            
            Hk = self.H[k] #(L,Nr,Nt)
            Lk = self.L[k]
            I = np.eye(self.NR[k])[None, :, :]  # shape (1, Nr, Nr)
            I = np.repeat(I, Lk, axis=0) # shape (L, Nr, Nr)
            Hk_h_transpose = np.conj(Hk).transpose(0,2,1) #(L,Nt,Nr)
            
            Fk = self.F[k] #(L,NT,dk)
            Fk_h_transpose = np.conj(Fk).transpose(0,2,1) #(L,dk,Nt)
            
            Ak = Hk@Fk@Fk_h_transpose@Hk_h_transpose #(L,Nr,Nr)
            
            sigma2_k = self.sigma2[k] #scalar
            epsilon_k = self.epsilon[k] #scalar
            
            Ck = np.log2(np.linalg.det(I + (Ak)/sigma2_k)) # (L,)
            LOG2E_SQ = (np.log2(np.e))**2
            
            # Suppose A is (..., N, N), I is identity (..., N, N)
            M = I + Ak / sigma2_k          # M

            # Compute matrix inverse for each batch
            M_inv = np.linalg.inv(M)   # shape (batch, N, N)
            M_inv2 = np.matmul(M_inv, M_inv)  # M^-2

            Vk = 0.5 * np.trace(I - M_inv2, axis1=1, axis2=2) * LOG2E_SQ


            R_fblk = Ck - np.sqrt(Vk/Lk) * Q_inv(epsilon_k)
            # R_fblk = 0
            
            self.C.append(Ck)
            self.V.append(Vk)
            self.R_fbl.append(R_fblk)
            
            self.usr_avg_C.append(np.mean(Ck, axis = 0))  
            
            
    def get_SNR(self):
        """
        Compute SNR per user using current channel, transmitted signal, and actual noise power from self.N.
        Returns:
            SNR_linear: list of linear SNR per user
            SNR_dB: list of SNR in dB per user
        """
        import numpy as np

        SNR_linear = []
        SNR_dB = []

        for k in range(self.K):
            # noiseless received signal
            Y_signal = self.H[k] @ (self.F[k] @ self.X[k])  # shape: (L, Nr, T)
            P_signal_user = np.mean(np.abs(Y_signal)**2)

            # actual noise power from self.N
            N_k = self.N[k]
            sigma2_k = np.mean(np.abs(N_k)**2)  # average noise power

            # linear and dB SNR
            snr_lin = P_signal_user / sigma2_k
            snr_db = 10 * np.log10(snr_lin)

            SNR_linear.append(snr_lin)
            SNR_dB.append(snr_db)

        return SNR_linear, SNR_dB


    def get_CNR(self):
        """
        Compute CNR per user using channel power and actual noise power from self.N.
        Returns:
            CNR_linear: list of linear CNR per user
            CNR_dB: list of CNR in dB per user
        """
        import numpy as np

        CNR_linear = []
        CNR_dB = []

        for k in range(self.K):
            # channel power per user (average Frobenius norm squared over L)
            H_power = np.mean([np.linalg.norm(self.H[k][l], 'fro')**2 for l in range(self.H[k].shape[0])])

            # actual noise power from self.N
            N_k = self.N[k]
            sigma2_k = np.mean(np.abs(N_k)**2)

            # linear and dB CNR
            cnr_lin = H_power / sigma2_k
            cnr_db = 10 * np.log10(cnr_lin)

            CNR_linear.append(cnr_lin)
            CNR_dB.append(cnr_db)

        return CNR_linear, CNR_dB
 
        
if __name__ == "__main__":   
    System = UplinkSystem(SYSTEM_TEST_PARAMS )

    #-----------------Sanity Check System--------------------$
    test_usr, test_block = 0,0
    # print("H")
    # print(System.H[test_usr][test_block])
    # print("X")
    # print(System.X[test_usr][test_block])
    # print("Y")
    # print(System.Y[test_usr][test_block])

    # print("Check Dims")
    # print(f" Y = {System.Y[test_usr].shape}, H = {System.H[test_usr].shape} , F = {System.F[test_usr].shape}, X = {System.X[test_usr].shape}\n")
    #--------------------------------------------------$
    
    print("|------------ System Constants ------------|")
    print(System.system_constants)
    print()
    
    print(System.R_fbl[test_usr][test_block].real)
    print(System.C[test_usr][test_block].real)
    print(System.F[test_usr][test_block])

    # System.check_SNR_user()
    # # System.check_SNR_block()
    # System.check_latency_users()
    
    # System.ChannelSystem.plot_magnitude_per_block()
    # System.ChannelSystem.plot_magnitude_over_blocklength()
    
    # for usr in range(test_k):
    #     print(f"Capacity per user per coh block: User {usr} :", System.C[usr])
    #     print(f"Ergodic capacity (avg across coh block) per user: User {usr}: ", System.ergodic_C[usr])
     
    # "Print Rate"
    # for usr in range(System.K):
    #     print(f"Rate_fbl per user : User {usr} :", System.R_fbl[usr])
    # print(System.sigma2[0])

