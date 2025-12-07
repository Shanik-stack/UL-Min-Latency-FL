from System import *
from Optimizer import *
from simulation_params import SYSTEM_TEST_PARAMS 


if __name__ == "__main__":
    all_results = []    
    uplinksystem = UplinkSystem(SYSTEM_TEST_PARAMS)
    
    for user in range(uplinksystem.K):
        for block in range(uplinksystem.L[user]):
            system_cfg = create_user_system_config(user, block, uplinksystem)
            simulation_cfg = create_simulation_config(*SIMULATION_TEST_PARAMS.values())
            result = optimize_precoder_blocklength(system_config= system_cfg, simulation_config = simulation_cfg)
            
            all_results.append(result)
            
for result in all_results[:2]:
    plt.ion()
    plot_optimization_result(result)