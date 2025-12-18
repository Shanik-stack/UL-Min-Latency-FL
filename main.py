from System import *
from Optimizer_per_block import *
from simulation_params import SYSTEM_TEST_PARAMS 
import os
def update(user, user_result,new_n, new_L, iteration:int,  uplink_system:UplinkSystem):
    # Update n
    # Update L
    # Update F
    '''Need to update according to new L'''
    uplink_system.n[user] = new_n
    uplink_system.L[user] = new_L
    uplink_system.F[user] = np.array(F_block[iteration]["F"] for F_block in user_result)
    # uplink_system.H[user] = np.array()
    print("Updated F: ", uplink_system.F[user].shape)
    return uplink_system

def clear_figs(path = "figs"):
    path = os.getcwd() + f"\{path}"
    for filename in os.listdir(path):
        if "png" in filename:
            os.remove(fr"{path}\{filename}")
    
if __name__ == "__main__":
    
    clear_figs()
    
    all_user_results = []
    uplinksystem = UplinkSystem(SYSTEM_TEST_PARAMS)
    
    simulation_cfg = create_simulation_config(*SIMULATION_TEST_PARAMS.values())
    
    for user in range(1,2):
        print(f" |----------------- User: {user} -----------------|")
                
        new_n, new_L, user_result = optimize_blocklength_user(user, uplinksystem, simulation_cfg)
        all_user_results.append(user_result)
        update(user, user_result, new_n, new_L, iteration = -1, uplink_system = uplinksystem)
    
    for user_idx, user_result in enumerate(all_user_results):
        for block_idx, block_result in enumerate(user_result):
            result = block_result
            plot_optimization_result(block_result, user_idx, block_idx)
            
    for user_idx, user_result in enumerate(all_user_results):
        plot_SNR_result(user_result, uplinksystem, user_idx)
