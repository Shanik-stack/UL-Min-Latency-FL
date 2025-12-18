from System import *
from utils import *
from plotting import *
import torch
import torch.nn as nn
import torch.nn.functional as TF

import numpy as np
from scipy.stats import norm
from simulation_params import *

LOG2E_SQ = (np.log2(np.e))**2

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def Q_inv(epsilon :float, device = None, dtype = torch.float32):
    val = norm.ppf(1 - epsilon)
    return torch.tensor(val, device = device, dtype = dtype)

from dataclasses import dataclass

#----------------- Config Definitions --------------------
@dataclass
class UserSystemConfig:
    B: float
    n: int
    P: float
    T: int
    L: int
    H: torch.Tensor    # shape (Nr, Nt)
    Nr: int
    Nt: int
    dk: int
    sigma2: float
    epsilon: float
    F_initial: torch.Tensor
    
@dataclass
class SimulationConfig:
    initial_lambda_rate_constraint: float 
    initial_lambda_power_constraint: float
    epochs_per_n: int
    lr_net: float
    lr_rate_constraint: float
    lr_power_constraint: float
    n_min:int
    n_max:int
    n_step: int

#------------------------- Utility in Optmizer ----------------------------


def out_to_precoder(F_out: torch.Tensor, Nt:int , dk: int):
    F_reshaped = F_out.view(2, Nt, dk) 
    F_complex = (F_reshaped[0] + 1j*F_reshaped[1]).to(torch.complex64)
    return F_complex
    
def update_lambdas(lambda_rate:float, lambda_power:float, rate_constraint, power_constraint, lr_rate_constraint:float, lr_power_constraint: float):
    #Detatch the contraints to prevent generatino of gradients
    rate_constraint = rate_constraint.detach().item()
    power_constraint = power_constraint.detach().item()

    lambda_rate = lambda_rate + lr_rate_constraint * rate_constraint
    lambda_power = lambda_power + lr_power_constraint * power_constraint
    
    lambda_rate = max(0, lambda_rate + lr_rate_constraint * rate_constraint)
    lambda_power = max(0, lambda_power + lr_power_constraint * power_constraint)
    return lambda_rate, lambda_power

def create_user_system_config(user:int, block:int, uplinksystem: UplinkSystem) -> UserSystemConfig:
    system_config = UserSystemConfig(
        B = uplinksystem.B[user],
        n = uplinksystem.n[user],
        P = uplinksystem.Pt[user],
        T = uplinksystem.T[user],
        L = uplinksystem.L[user],
        H = torch.tensor(uplinksystem.H[user][block], dtype = torch.complex64, device = DEVICE), #block is the same as L
        Nr = uplinksystem.NR[user],
        Nt = uplinksystem.NT[user],
        dk = uplinksystem.dk[user],
        sigma2 = uplinksystem.sigma2[user],
        epsilon = uplinksystem.epsilon[user],
        F_initial = uplinksystem.F[user][block],
    )
    return system_config

def create_simulation_config(
    initial_lambda_rate_constraint: float,
    initial_lambda_power_constraint: float,
    epochs_per_n: int,
    lr_net: float,
    lr_rate_constraint: float,
    lr_power_constraint: float,
    n_min:int,
    n_max:int,
    n_step: int,
    ) -> SimulationConfig:
    
    simulation_config = SimulationConfig(initial_lambda_rate_constraint, initial_lambda_power_constraint, epochs_per_n, lr_net, lr_rate_constraint, lr_power_constraint, n_min,n_max,n_step)
    
    return simulation_config
    

def print_result(result, s:str):
    print(f"\nResult {s}: ")
    for i in result[::-1]:   
        print(i[s], end=", ")
    print()

    

    
#------------------Precoder Neural Network------------------

class UplinkMLP_PrecoderNet(nn.Module):
    def __init__(self, system_config: UserSystemConfig,  hidden = 1024):
        super().__init__()
        Nr = system_config.Nr
        Nt = system_config.Nt
        dk = system_config.dk
        in_dim = 2 * Nr * Nt #(H_real, H_imag)
        out_dim = 2 * Nt * dk
        hidden = 4 * (4 * out_dim) # i want the last layer to be 4*out_dim -> out_dim
        target_power = 0.95 * system_config.P  # Target 95% of budget
        expected_power = out_dim / 2

        init_scale = np.sqrt(target_power / expected_power)
        
        self.log_scale = nn.Parameter(torch.tensor(0.0, dtype = torch.float32))  # log(1)

        # self.F_scale = nn.Parameter((torch.tensor(1.0, dtype = torch.float32))
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, out_dim),
            nn.Tanh()                          # bounded output

        )
        
    def forward(self, H):
        # H: complex64 (Nr, Nt)
        H_flat = H.view(1,-1)
        H_real = H_flat.real
        H_imag = H_flat.imag
        H_combined = torch.cat((H_real, H_imag), dim = 1) #(1, 2 * Nr * Nt)
        F_out = torch.exp(self.log_scale) * self.net(H_combined)
        
        return F_out
    
    
#---------------------- Lagrangian Loss --------------------

class LagrangianLoss(nn.Module):
    def __init__(self, system_config: UserSystemConfig):
        super().__init__()
        self.system_config = system_config
    
    def get_R_fbl(self, F: torch.Tensor, lambda_rate, lambda_power):
        #Get blocklength n from system config
        n = self.system_config.n
        T = self.system_config.T
        L = n//T
        
        #F: complex64 (Nt, dk)
        H = self.system_config.H  # (Nr, Nt)
        sigma2 = self.system_config.sigma2  # scalar
        epsilon = self.system_config.epsilon  # scalar
        
        self.lambda_rate = lambda_rate
        self.lambda_power = lambda_power
        
        I = torch.eye(self.system_config.Nr, dtype = torch.complex64, device=H.device)
        H_h_transpose = H.conj().transpose(1, 0)  # (Nt, Nr)
        F_h_transpose = F.conj().transpose(1, 0)  # (dk, Nt)
        
        A = H @ F @ F_h_transpose @ H_h_transpose  # (Nr, Nr)
        
        #Calculate CApacity
        sign, logdet = torch.linalg.slogdet(I + A / sigma2)
        C = sign * logdet / np.log(2)  # convert from ln to log2 # scalar
        
        #CAlculate Variance
        M = I + A / sigma2
        if torch.isnan(M).any() or torch.isinf(M).any():
            print("NaN/Inf detected in M before inversion.")
            print("Max:", M.abs().max())
            print("Min:", M.abs().min())
            print("sigma2:", sigma2)
            print("H max:", H.abs().max())
            print("F max:", F.abs().max())
        try:
            M_inv = M_inv = torch.linalg.solve(M + 1e-8*torch.eye(M.shape[0], device=M.device, dtype=M.dtype), I) #M^-1
            M_inv2 = M_inv @ M_inv # M^{-2}
            V = 0.5 * torch.trace(I - M_inv2) * LOG2E_SQ
            
            # formula is changed to C - \sqrt{V/L} * Q^{-1}(\epsilon) for AWGN 
            # (and also it doesnt make sense to use T because T is constant in a block and whats tht point of the optimization proccess then)
            R_fbl_awgn = torch.real(C - torch.sqrt(V / n) * Q_inv(epsilon, device=F.device))
        except:
            print("Min eigenvalue of M:", torch.linalg.eigvals(M).real.min())
        
        return R_fbl_awgn
    
    def forward(self, F, lambda_rate, lambda_power):
        cfg = self.system_config
        R_fbl = self.get_R_fbl(F, lambda_rate, lambda_power)
        F_power = torch.linalg.norm(F, ord='fro')**2
        rate_constraint = cfg.B / cfg.n - R_fbl
        power_constraint =  F_power - cfg.P
        
        # Soft penalties (positive only if violated)
        rate_constraint = TF.leaky_relu(rate_constraint)   # R_fbl < target
        power_constraint = TF.relu(power_constraint)  # power > limit
        
        
        weighted_rate_constraint = self.lambda_rate * rate_constraint
        weighted_power_constraint = self.lambda_power *  power_constraint
        # To counter the case when power drops too low
        # max_power_constraint = cfg.P/F_power
        # max_power_constrain = 0.0
        return -R_fbl + weighted_rate_constraint + weighted_power_constraint , R_fbl, F_power, rate_constraint, power_constraint



#--------------------- Precoder Neural Net Optimization Loop ----------------------

def optimize_precoder(precoder_net,  F: torch.Tensor, system_config:UserSystemConfig, epochs: int,
                      lambda_rate: float, lambda_power: float,
                      lr_net, lr_rate_constraint, lr_power_constraint, optimizer = None):
    
    precoder_net = precoder_net.to(DEVICE)
    loss_fn = LagrangianLoss(system_config)
    if(optimizer == None):
        optimizer = torch.optim.Adam(precoder_net.parameters(), lr = lr_net)
    losses = []
    
    for epoch in range(epochs):
        H_in = system_config.H.to(DEVICE)
        F_out = precoder_net(H_in)
        F = out_to_precoder(F_out[0], system_config.Nt, system_config.dk)
        
        optimizer.zero_grad()
        loss, R_fbl, F_power, rate_constraint, power_constraint = loss_fn(F, lambda_rate, lambda_power)
       
        # early stoppping
        if(rate_constraint<=0 and power_constraint<=0 and epoch>=20):
            break
       
        lambda_rate, lambda_power = update_lambdas(lambda_rate, lambda_power, rate_constraint, power_constraint, lr_rate_constraint, lr_power_constraint)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())        
        
        # print(f'''
# Epoch: {epoch}
# Loss: {loss.item()}, lambda_rate={lambda_rate}, lambda_power={lambda_power}
# R_fbl: {R_fbl}, rate_constraint: {rate_constraint}, power_constraint: {power_constraint}
# ''')
    
    F_out = precoder_net(system_config.H)
    F_final = out_to_precoder(F_out[0], system_config.Nt, system_config.dk)
    return F_final.detach(),optimizer, lambda_rate, lambda_power, R_fbl.detach(), F_power.detach(), rate_constraint.detach(), power_constraint.detach(), losses



# ---------------------- Precoder-BLocklength Optimization Loop per block ------------------------
def optimize_precoder_blocklength_block(system_config:UserSystemConfig, simulation_config: SimulationConfig):
    
    F = system_config.F_initial
    
    initial_lambda_rate_constraint = simulation_config.initial_lambda_rate_constraint
    initial_lambda_power_constraint = simulation_config.initial_lambda_power_constraint
    lr_rate_constraint = simulation_config.lr_rate_constraint
    lr_power_constraint = simulation_config.lr_power_constraint
    epochs_per_n = simulation_config.epochs_per_n
    lr_net = simulation_config.lr_net
    n_max = simulation_config.n_max
    n_min = simulation_config.n_min
    n_step = simulation_config.n_step
    
    cfg = system_config
    
    result = []
    
    R_fbl = torch.tensor(0.0) 
    stop_count = 1
    count = 0
    
    # initialize only once (fixed)
    precoder_net = UplinkMLP_PrecoderNet(system_config = system_config).to(DEVICE)
    optimizer = torch.optim.Adam(precoder_net.parameters(), lr = lr_net)

    lambda_rate_constraint = initial_lambda_rate_constraint
    lambda_power_constraint = initial_lambda_power_constraint
    for n in n_max - np.arange(n_min, n_max - n_min , n_step):
        
        cfg.n = n
        cfg.L = cfg.n // cfg.T
        
        if(n == n_max - n_min):
            epochs = 25000
        else:
            epochs = epochs_per_n
        # Always optimize for this n first (fixed)
        F_final, optimizer, lambda_rate,lambda_power, R_fbl,F_power, true_rate_constraint, true_power_constraint, loss = optimize_precoder(
            precoder_net, F,
            system_config = cfg,
            epochs = epochs,
            lambda_rate = lambda_rate_constraint,
            lambda_power = lambda_power_constraint,
            lr_net = lr_net,
            lr_rate_constraint = lr_rate_constraint,
            lr_power_constraint = lr_power_constraint,
            # optimizer = None,
            optimizer = optimizer
        )
        
        #Update lambda
        lambda_rate_constraint = lambda_rate
        lambda_power_constraint = lambda_power
        
        # Check feasibility on updated R_fbl and updated F_final 
        if true_rate_constraint <= 0 and true_power_constraint <= 0:
            result.append({
                "n": cfg.n,
                "F": F_final,
                "optimizer": optimizer,
                "lambda_rate": lambda_rate_constraint,
                "lambda_power": lambda_power_constraint,
                "Bits per Blocklength": cfg.B/cfg.n,
                "R_fbl": R_fbl.item(),
                "F_power": F_power,
                "loss": loss
            })
            print(f" --------------- Possible Bits per Blocklength (B/n) = {cfg.B/cfg.n} -------------")
        else:
            count += 1
            print("COUNT: ", count)
            print(f"---- Not Possible Bits per Blocklength (B/n) {cfg.B/cfg.n} -------")

        # Stop early if too many failures
        if count >= stop_count:
            break
    print_result(result, "n")
    print_result(result, "R_fbl")
    print_result(result, "F_power")
    print_result(result, "lambda_rate")
    print_result(result, "lambda_power")

        
    # plot_optimization_result(result
    
    return result


            

def optimize_blocklength_user(user:int, uplinksystem: UplinkSystem, simulation_cfg: SimulationConfig, per_block = True):
    L = uplinksystem.L[user]
    B = uplinksystem.B[user]
    T = uplinksystem.T[user]
    result_R_fbl = []
    result_n = []
    User_Result = []
    for block in range(L):
        print(f"|------------------- Block: {block} ----------------|")
        system_cfg = create_user_system_config(user, block, uplinksystem)
        result = optimize_precoder_blocklength_block(system_config= system_cfg, simulation_config = simulation_cfg)
        User_Result.append(result)
        largest_R_fbl = result[-1]["R_fbl"]
        smallest_n = result[-1]["n"]
        
        result_R_fbl.append(largest_R_fbl)
        result_n.append(smallest_n)
    
    avg_R_fbl = np.mean(result_R_fbl)
    
    
    new_n = int(B/avg_R_fbl)
    if (not per_block):
        new_L = new_n//T
    else:
        new_L = L
    
    check_n_validity = 0
    
    #Check if rates in each block agrees with avg rate (Dont know if needed)
    # count = 0
    # while(not check_n_validity):
    #     for block in range(new_L):
    #         if result_R_fbl[block] < avg_R_fbl:
    #             break
    #         if(block == new_L - 1):
    #             check_n_validity = 1
    #     if(not check_n_validity):
    #         count+=1
    #         new_n =  new_n + simulation_cfg.n_step
    #         new_n = result[-1-count]["n"]
    #         print("new_n from step", new_n)
    #         new_L = new_n//T
    #         avg_R_fbl = np.mean(result[-1-count])
    
    return new_n, new_L, User_Result
    


if __name__ == "__main__":
    user = 0
    # test_system_constants = initialize_const()
    uplinksystem = UplinkSystem(SYSTEM_TEST_PARAMS)

    simulation_cfg = create_simulation_config(*SIMULATION_TEST_PARAMS.values())
    # system_cfg = create_user_system_config(user, block, uplinksystem)

    # precoder_net =  UplinkMLP_PrecoderNet(system_config = system_cfg)
    # F = torch.tensor(uplinksystem.F[user][block])
    # print("Power", torch.linalg.norm(F, ord='fro')**2)
    
    new_n = optimize_blocklength_user(user, uplinksystem, simulation_cfg)
    print(new_n)
    
    










    #Plot loss curves
    # n_values = n_max - np.arange(n_min, n_max - n_min, n_step)

    # plt.figure(figsize=(10,6))

    # for i, n in enumerate(n_values):
    #     loss_curve = loss_over_n[i]
    #     x_values = np.arange(1, len(loss_curve)+1)  # x-axis for this curve
    #     plt.plot(x_values, loss_curve, label=f'n={n}')

    # plt.xlabel('Epoch (or step)')
    # plt.ylabel('Loss')
    # plt.title('Nested Loss Curves for Different Blocklengths n')
    # plt.grid(True)
    # plt.legend(title='Blocklength n')
    # plt.show()


