from utils import *

#-------------------- Test Params -------------------------
test_k = 2 # Number of users
test_Nr = [16]*test_k  #Number of receive antenna per user at BS
test_Nt = [4]*test_k #Numver of transmit antennaa per user

test_n = [600, 200] # Initial Blocklength for each user
test_t = [20, 10]  

test_snr_db = [5.0, 10.0]
test_Pt = [30.0, 10.0]
test_fs = [2] * test_k
test_B = [2000, 2000]
test_epsilon = [1e-10, 1e-5]

seed = 42

#------------------- Simulation Params --------------------
initial_lambda_rate_constraint =  0.1
initial_lambda_power_constraint =  0.1
epochs_per_n =  1000
lr_net =  0.001
lr_rate_constraint =  0.01
lr_power_constraint =  0.01
n_min =  100
n_max = 600
n_step =  5


SYSTEM_TEST_PARAMS = initialize_system_params(B = test_B, Pt = test_Pt, fs = test_fs, snr_db = test_snr_db, desired_CNR = [], Nt = test_Nt, Nr = test_Nr, K = test_k, N = test_n, T = test_t, epsilon = test_epsilon)
SIMULATION_TEST_PARAMS = { "initial_lambda_rate_constraint": initial_lambda_rate_constraint,
    "initial_lambda_power_constraint":initial_lambda_power_constraint ,
    "epochs_per_n": epochs_per_n,
    "lr_net":lr_net,
    "lr_rate_constraint": lr_rate_constraint ,
    "lr_power_constraint": lr_power_constraint,
    "n_min": n_min,
    "n_max": n_max,
    "n_step": n_step,}


if __name__ == "__main__":
    print(SYSTEM_TEST_PARAMS)
    print(SIMULATION_TEST_PARAMS)
    
    
    