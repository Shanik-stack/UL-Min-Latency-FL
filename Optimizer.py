from System import *
from helper_functions import *
import torch.nn as nn
import torch
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from test_const import TEST_CONST

LOG2E_SQ = (np.log2(np.e))**2

def Q_inv(epsilon :float, device = None, dtype = torch.float32):
    val = norm.ppf(1 - epsilon)
    return torch.tensor(val, device = device, dtype = dtype)


@dataclass
class Config:
    B: float
    n: int
    P: float
    T: float
    H: torch.Tensor      # shape (Nr, Nt)
    Nr: int
    Nt: int
    dk: int
    sigma2: float
    epsilon: float
    
#------------------Precoder Neural Network------------------

class UplinkMLP_PrecoderNet(nn.Module):
    def __init__(self, config: Config,  hidden = 1024):
        super().__init__()
        Nr = config.Nr
        Nt = config.Nt
        dk = config.dk
        in_dim = 2 * Nr * Nt #(H_real, H_imag)
        out_dim = 2 * Nt * dk
        hidden = 4 * (4 * out_dim) # i want the last layer to be 4*out_dim -> out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, out_dim)
        )
    def forward(self, H):
        # H: complex64 (Nr, Nt)
        H_flat = H.view(1,-1)
        H_real = H_flat.real
        H_imag = H_flat.imag
        H_combined = torch.cat((H_real, H_imag), dim = 1) #(1, 2 * Nr * Nt)
        out = self.net(H_combined)
        
        return out
    
    
#----------------------Lagrangian Loss--------------------

class LagrangianLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
    
    def get_R_fbl(self, F: torch.Tensor, lambda_rate, lambda_power):
        #F: complex64 (Nt, dk)
        H = self.config.H  # (Nr, Nt)
        sigma2 = self.config.sigma2  # scalar
        epsilon = self.config.epsilon  # scalar
        
        self.lambda_rate = lambda_rate
        self.lambda_power = lambda_power
        
        I = torch.eye(self.config.Nr, dtype = torch.complex64, device=F.device)
        H_h_transpose = H.conj().transpose(1, 0)  # (Nt, Nr)
        F_h_transpose = F.conj().transpose(1, 0)  # (dk, Nt)
        
        A = H @ F @ F_h_transpose @ H_h_transpose  # (Nr, Nr)
        
        #Calculate CApacity
        sign, logdet = torch.linalg.slogdet(I + A / sigma2)
        C = sign * logdet / np.log(2)  # convert from ln to log2 # scalar
        
        #CAlculate Variance
        M = I + A / sigma2
        M_inv = torch.linalg.solve(M, I) #M^-1
        M_inv2 = M_inv @ M_inv # M^{-2}
        V = 0.5 * torch.trace(I - M_inv2) * LOG2E_SQ
        
        #Real value rate
        R_fbl = torch.real(C - torch.sqrt(V / self.config.T) * Q_inv(epsilon, device=F.device))
        
        return R_fbl
    
    def forward(self, F, lambda_rate, lambda_power):
        cfg = self.config
        R_fbl = self.get_R_fbl(F, lambda_rate, lambda_power)
        
        pre_rate_constraint = (R_fbl - cfg.B / cfg.n)
        pre_power_constraint = (cfg.P - torch.linalg.norm(F, ord='fro')**2)
        # print("TEST: ", R_fbl , cfg.B , cfg.n) 
        
        rate_constraint = self.lambda_rate * pre_rate_constraint
        power_constraint = self.lambda_power *  pre_power_constraint
        
        return -(R_fbl + rate_constraint + power_constraint)/3, R_fbl, pre_rate_constraint, pre_power_constraint

#------------------------- Utility ----------------------------

def out_to_precoder(F_out: torch.Tensor, Nt, dk):
    F_reshaped = F_out.view(2, Nt, dk) 
    F_complex = (F_reshaped[0] + 1j*F_reshaped[1]).to(torch.complex64)
    return F_complex
    
def update_lambdas(lambda_rate:float, lambda_power:float, pre_rate_constraint, pre_power_constraint, lr_rate_constraint:float, lr_power_constraint: float):
    #Detatch the contraints to prevent generatino of gradients
    pre_rate_constraint = pre_rate_constraint.detach().item()
    pre_power_constraint = pre_power_constraint.detach().item()

    lambda_rate = lambda_rate + lr_rate_constraint * pre_rate_constraint
    lambda_power = lambda_power + lr_power_constraint * pre_power_constraint
    
    # lambda_rate = max(0, lambda_rate + lr_rate_constraint * pre_rate_constraint)
    # lambda_power = max(0, lambda_power + lr_power_constraint * pre_power_constraint)
    return lambda_rate, lambda_power

#--------------------- Precoder Neural Net Optimization Loop ----------------------

def optimize_precoder(precoder_net,  F: torch.Tensor, config:Config, epochs: int,
                      lambda_rate: float, lambda_power: float,
                      lr_net, lr_rate_constraint, lr_power_constraint):

    loss_fn = LagrangianLoss(config)
    optimizer = torch.optim.Adam(precoder_net.parameters(), lr = lr_net)
    
    for epoch in range(epochs):
        F_out = precoder_net(config.H)
        F = out_to_precoder(F_out[0], config.Nt, config.dk)
        
        optimizer.zero_grad()
        loss, R_fbl, pre_rate_constraint, pre_power_constraint = loss_fn(F, lambda_rate, lambda_power)
        loss.backward()
        optimizer.step()
        
        # lambda_rate, lambda_power = update_lambdas(lambda_rate, lambda_power, pre_rate_constraint, pre_power_constraint, lr_rate_constraint, lr_power_constraint)
        print(f'''
Epoch: {epoch}
Loss: {loss.item()}, lambda_rate={lambda_rate}, lambda_power={lambda_power}
R_fbl: {R_fbl}, pre_rate_constraint: {pre_rate_constraint}, pre_power_constraint: {pre_power_constraint}
''')
    
    F_out = precoder_net(config.H)
    F_final = out_to_precoder(F_out.view(-1), config.Nt, config.dk)
    return F_final

def optimize_blocklength(n:int, config:Config):
    config.n = n
    return config


if __name__ == "__main__":
    user = 0
    block = 0
    # test_system_constants = initialize_const()
    uplinksystem = UplinkSystem(TEST_CONST)
    config = Config(
        B = uplinksystem.B[user],
        n = uplinksystem.n[user],
        P = uplinksystem.Pt[user],
        T = uplinksystem.T[user],
        H = torch.tensor(uplinksystem.H[user][block], dtype = torch.complex64), #block is the same as L
        Nr = uplinksystem.NR[user],
        Nt = uplinksystem.NT[user],
        dk = uplinksystem.dk[user],
        sigma2 = uplinksystem.sigma2[user],
        epsilon = 0.0001,
    )
    cfg = config
    F_initial = uplinksystem.F[user][block]
    
    lambda_rate = 6
    lambda_power = 5
    epochs = 20
    lr_net = 0.001
    lr_rate_constraint = 0.01
    lr_power_constraint = 0.01
    precoder_net =  UplinkMLP_PrecoderNet(config = config)
    
    for n in [100, 80, 50]:
        cfg = optimize_blocklength(n, cfg)
        F_final = optimize_precoder(precoder_net, F_initial, config = cfg, epochs = epochs, lambda_rate = lambda_rate, lambda_power = lambda_power, 
                                    lr_net = lr_net, lr_rate_constraint = lr_rate_constraint, lr_power_constraint = lr_power_constraint)
        print(f''' --------------- Real Rate (B/n) = {cfg.B/cfg.n} -------------''')
    initial_Rfbl = uplinksystem.R_fbl[user][block].real
    pre_rate_constraint = (initial_Rfbl - cfg.B / cfg.n)
    pre_power_constraint = (cfg.P - np.linalg.norm(uplinksystem.F[user][block], ord='fro')**2)
    
    print(f''' 
User: {user}, block: {block} 
Initial R_fbl: {initial_Rfbl}
Initial pre rate contraints: {pre_rate_constraint}
Initial pre_power_constraint: {pre_power_constraint} ''')
# Initial F: {uplinksystem.F[user][block]}''')
