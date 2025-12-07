from system_const import ChannelConstants
from System import UplinkSystem
import matplotlib.pyplot as plt
import numpy as np
from simulation_params import SYSTEM_TEST_PARAMS 

# ------------------------- Plotting Channel Constansts/channel system --------------------

def plot_channel_magnitude1(channelsystem, users: list[int] = []):
    if(len(users) == 0 ): users = list(range(0,channelsystem.K))

    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):            
        H_mag_k = np.abs(channelsystem.H[k]) # shape: (L, NR, NT) 

        # Plot each Tx-Rx pair across coherence blocks
        for rx in range(channelsystem.NR[k]):
            for tx in range(channelsystem.NT[k]):
                axes[idx].plot(range(channelsystem.L[k]), H_mag_k[:, rx, tx], label=f'Rx{rx+1}-Tx{tx+1}')

        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Coherence block index (l)')
        axes[idx].set_ylabel('|H| magnitude')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')

    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_channel_magnitude2(channelsystem, users: list[int] = []):
    if(len(users) == 0 ): users = list(range(0,channelsystem.K))
    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):            
        H_mag_k = np.abs(channelsystem.H[k]) # shape: (L, NR, NT) 
        H_plot_k = []
        for l in range(len(H_mag_k)):
            H_plot_k.append([H_mag_k[l]]*channelsystem.T[k])
            
        H_plot_k = np.array(H_plot_k).reshape(-1, channelsystem.NR[k], channelsystem.NT[k])
        # Plot each Tx-Rx pair across coherence blocks
        for rx in range(channelsystem.NR[k]):
            for tx in range(channelsystem.NT[k]):
                axes[idx].plot(range(channelsystem.n[k]), H_plot_k[:, rx, tx], label=f'Rx{rx+1}-Tx{tx+1}')

        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('|H| magnitude')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')

    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot__channel_magnitude_over_time(channelsystem, user: list[int] = []):
    pass



def check_SNR_user(system: UplinkSystem, users: list[int] = []):
    #SNR at Receiver per user is calculated
    
    if(len(users) == 0): users = list(range(0,system.K))
    Y_users = []
    N_users = []
    SNR_users_db = []
    for usr in users:
        Y_users.append((system.H[usr] @( system.F[usr] @ system.X[usr])).reshape(-1))  # shape: (K, L*Nt*T)
        N_users.append(system.N[usr].reshape(-1))  # shape: (K, L*Nr*T)
        P_signal_user = np.mean(np.abs(Y_users[usr])**2)
        P_noise_user  = np.mean(np.abs(N_users[usr])**2)
        SNR_user = P_signal_user / P_noise_user
        SNR_users_db.append(10 * np.log10(SNR_user))
    
    print("\n------SNR per User------")
    print(SNR_users_db)
    
def check_SNR_block(system: UplinkSystem, users: list[int] = []):
    #SNR at Receiver per user per block is calculated
    
    if(len(users) == 0): users = list(range(0,system.K))
    
    Y_users = []
    N_users = []
    SNR_users_db = []
    
    print("\n------SNR per User per BLock------\n")
    for usr in users:
        Y_users.append((system.H[usr] @ system.X[usr]).reshape(system.L[usr],  -1)) # shape: (K, L*Nt*T)
        N_users.append(system.N[usr].reshape(system.L[usr], -1))  # shape: (K, L*Nr*T)
    
        P_signal_users = np.mean(np.abs(Y_users[usr])**2, axis = -1)
        P_noise_users  = np.mean(np.abs(N_users[usr])**2, axis = -1)
        SNR_user = P_signal_users / P_noise_users
        SNR_users_db.append(10 * np.log10(SNR_user))
        print(f"SNR per BLockl for User {usr}: ")
        print(SNR_user)
    """This code currently assumes that Nt and Nr are the same for all users"""

    
def check_latency_users(uplinksystem: UplinkSystem, users: list[int]=  []):
    if(len(users) == 0): users = list(range(0,uplinksystem.K))

    print("\n----------Latency of Users {users}-------------")
    print(system.latency[users])

# -------------------------- Plot Capacity ---------------------

def plot_capacity1(uplinksystem: UplinkSystem, users: list[int] = []):
    if(len(users) == 0): users = list(range(0,system.K))
    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):
        Ck = system.C[k]
        axes[idx].plot(range(uplinksystem.L[k]), Ck)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Capacity')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')
    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_capacity2(uplinksystem: UplinkSystem, users: list[int] = []):
    if(len(users) == 0): users = list(range(0,system.K))
    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):
        Ck = np.repeat(uplinksystem.C[k], uplinksystem.T[k], axis = 0)
        axes[idx].plot(range(uplinksystem.n[k]), Ck)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Capacity')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')
    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
# --------------------------- Plot Dispersion ----------------------

def plot_dispersion1(uplinksystem: UplinkSystem, users: list[int] = []):
    if(len(users) == 0): users = list(range(0,system.K))
    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):
        Vk = system.V[k]
        axes[idx].plot(range(uplinksystem.L[k]), Vk)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Dispersion')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')
    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_dispersion2(uplinksystem: UplinkSystem, users: list[int] = []):
    if(len(users) == 0): users = list(range(0,system.K))
    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):
        Vk = np.repeat(uplinksystem.V[k], uplinksystem.T[k], axis = 0)
        axes[idx].plot(range(uplinksystem.n[k]), Vk)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Dispersion')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')
    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# ----------------------- Plot Rate ---------------------
 
def plot_rate_fbl1(uplinksystem: UplinkSystem, users: list[int] = []):
    if(len(users) == 0): users = list(range(0,system.K))
    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):
        R_fbl_k = system.R_fbl[k]
        axes[idx].plot(range(uplinksystem.L[k]), R_fbl_k)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Rate fbl')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')
    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_rate_fbl2(uplinksystem: UplinkSystem, users: list[int] = []):
    if(len(users) == 0): users = list(range(0,system.K))
    K = len(users)
    nrows = (K + 1) // 2  # two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()  # flatten in case K is odd

    for idx, k in enumerate(users):
        R_fbl_k = np.repeat(uplinksystem.R_fbl[k], uplinksystem.T[k], axis = 0)
        axes[idx].plot(range(uplinksystem.n[k]), R_fbl_k)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Rate')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')
    # Hide extra subplots if K is odd
    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.ioff()
    plt.show()

def plot_2(uplinksystem):
    plt.ion()
    plot_capacity2(uplinksystem)
    plot_dispersion2(uplinksystem)
    plot_rate_fbl2(uplinksystem)

# ---------------------------- Plot Optimization ----------------------
def plot_optimization_result(result: list, user: int, T):

    # result is a list of dicts like:
    # {"n": ..., "Real Rate (B/n)": ..., "R_fbl": ..., "F": ....}

    # Extract arrays
    n_vals = [d["n"] for d in result]
    real_rates = [d["Real Rate (B/n)"] for d in result]
    r_fbl_vals = [d["R_fbl"] for d in result]

    # Plot
    plt.figure()
    plt.plot(n_vals, real_rates, marker='o', label='Real Rate (B/n) (bits/blocklength)')
    plt.plot(n_vals, r_fbl_vals, marker='s', label='R_fbl')

    plt.xlabel('Blocklength n')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)

    plt.gca().invert_xaxis()      # ← makes x-axis go high → low
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test_system_constants = SYSTEM_TEST_PARAMS 
    channelsystem = ChannelConstants(test_system_constants)
    system = UplinkSystem(test_system_constants)
    # check_latency_users(system)
    plot_2(system)
    # plot_magnitude_per_block(channelsystem)
    # plot_magnitude_over_blocklength(channelsystem) 
    