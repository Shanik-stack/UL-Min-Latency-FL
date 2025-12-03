from System import *
from systemconstants import *
from helper_functions import *
import torch.nn as nn
import torch

class UplinkMLP_PrecoderNet(nn.Module):
    def __init__(self, Nr, Nt, dk,  hidden = 1024):
        super().__init__()
        self.Nr = Nr
        self.Nt = Nt
        self.dk = dk
        in_dim = 2 * Nr * Nt #(H_real, H_imag)
        out_dim = 2 * Nt * dk
        hidden = 4 * (4 * out_dim) # i want the last layer to have 4*out_dim neurons
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
        H = H.view(-1)
        H_real = torch.tensor(H.real, dtype = torch.float32)
        H_imag = torch.tensor(H.imag, dtype = torch.float32)
        H = torch.cat((H_real, H_imag), dim = 0) #(2 * Nr * Nt)
        print("H:", H)
        out = self.net(H)
        
        return out
        
class LagrangianLoss(nn.Module):

def optimize_precoder(F):
    Loss 

def optimize_blocklength(n):
    pass


if __name__ == "__main__":
    Nr = 4
    Nt = 2
    dk = 2
    H = torch.rand(Nr, Nt, dtype = torch.complex32)
    print(H)
    uplink_MLP_net = UplinkMLP_PrecoderNet(Nr, Nt, dk)
    x = uplink_MLP_net(H)
    x_real = x[:x.shape[0]//2]
    x_imag = x[x.shape[0]//2:]
    x = x_real + 1j *x_imag
    print(x, x.shape)
    print(x.view(Nt, dk))
    
