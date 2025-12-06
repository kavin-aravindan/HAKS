import torch
import numpy as np
import copy
from datetime import datetime
import pytz

from config import CONFIG
from hedge import run_adaptive_bo, run_single_kernel_bo
from plotting import plot_seed_dashboard, plot_combined_log_regret
from utils import save_data, load_data, get_kernel_name, calculate_log_regret_stats, get_dashboard_path

def main():
    torch.set_default_dtype(torch.float64)
    loss_functions = CONFIG['loss_functions']
    seeds = CONFIG['seed_list']
    lrs = CONFIG['learning_rates']
    test_functions = CONFIG['test_functions']
    
    for test_func in test_functions:
        print(f"\n{'='*40}\nFunction: {test_func.name} (Dim: {test_func.dim})\n{'='*40}")
        
        n_init = test_func.dim * CONFIG['n_init_factor']
        
        # --- 1. Setup Kernels ---
        kernels = []
        kernel_names = []
        for k in CONFIG['candidate_kernels']:
            k_copy = copy.deepcopy(k)
            if hasattr(k_copy.base_kernel, 'ard_num_dims'):
                k_copy.base_kernel.ard_num_dims = test_func.dim
            kernels.append(k_copy)
            kernel_names.append(get_kernel_name(k_copy))

        # --- 2. Run/Load Baselines ---
        print(">> Processing Baselines...")
        baseline_stats_map = {} # For Aggregated Plot
        baseline_raw_map = {}   # For Per-Seed Dashboards: {kernel_name: [list of arrays per seed]}
        
        for i, kernel_name in enumerate(kernel_names):
            baseline_id = f"Baseline_{kernel_name}"
            baseline_data = load_data(CONFIG, test_func, baseline_id)
            
            if baseline_data is None:
                print(f"   Running Baseline: {kernel_name}")
                baseline_data = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    init_x = torch.rand(n_init, test_func.dim) * (test_func.bounds[1] - test_func.bounds[0]) + test_func.bounds[0]
                    init_y = test_func.true_function(init_x).squeeze()
                    
                    hist = run_single_kernel_bo(
                        kernels[i], test_func, n_init, init_x, init_y, 
                        CONFIG['acq_func'], CONFIG['ucb_beta'], CONFIG['num_bo_steps']
                    )
                    baseline_data.append(hist)
                save_data(baseline_data, CONFIG, test_func, baseline_id)
            
            # Store data in memory
            baseline_raw_map[kernel_name] = baseline_data
            
            # Compute stats
            mean, lower, upper = calculate_log_regret_stats(baseline_data, test_func.optimum_value)
            baseline_stats_map[kernel_name] = {'mean': mean, 'lower': lower, 'upper': upper}

        # --- 3. Run/Load Adaptive Methods ---
        print("\n>> Processing Adaptive Methods...")
        
        for lr in lrs:
            adaptive_stats_map = {}
            
            for loss_id in loss_functions:
                identifier = f"Adaptive_L{loss_id}_lr{lr}"
                run_results = load_data(CONFIG, test_func, identifier)
                
                # If data missing, run experiments
                if run_results is None:
                    print(f"   Running: LR={lr}, LossID={loss_id}")
                    run_results = {'best_y': [], 'prob': [], 'loss': []}
                    
                    for s_idx, seed in enumerate(seeds):
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        init_x = torch.rand(n_init, test_func.dim) * (test_func.bounds[1] - test_func.bounds[0]) + test_func.bounds[0]
                        init_y = test_func.true_function(init_x).squeeze()
                        
                        data = run_adaptive_bo(
                            kernels, test_func, n_init, init_x, init_y,
                            CONFIG['acq_func'], loss_id, lr, CONFIG['ucb_beta'], CONFIG['num_bo_steps']
                        )
                        
                        run_results['best_y'].append(data['best_y'])
                        run_results['prob'].append(data['prob_history'])
                        run_results['loss'].append(data['losses'])
                        
                        # --- GENERATE DASHBOARD FOR THIS SEED ---
                        # 1. Extract baseline traces for this specific seed index
                        current_seed_baselines = {
                            k_name: baseline_raw_map[k_name][s_idx] 
                            for k_name in kernel_names
                        }
                        
                        # 2. Get path
                        dash_path = get_dashboard_path(CONFIG['out_dir'], test_func.name, test_func.dim, loss_id, lr, seed)
                        
                        # 3. Plot
                        plot_seed_dashboard(
                            adaptive_data=data,
                            baseline_map=current_seed_baselines,
                            kernel_names=kernel_names,
                            optimum=test_func.optimum_value,
                            path=dash_path,
                            func_name=test_func.name,
                            dim=test_func.dim,
                            loss_id=loss_id,
                            lr=lr,
                            seed_idx=seed
                        )
                    
                    save_data(run_results, CONFIG, test_func, identifier)
                
                # If data was loaded from cache, we might want to regenerate plots?
                # For now, we assume if cache exists, we just calculate stats. 
                # (If you need to regenerate plots from cache, you'd iterate `run_results` here)
                
                # Compute Stats
                mean, lower, upper = calculate_log_regret_stats(run_results['best_y'], test_func.optimum_value)
                adaptive_stats_map[loss_id] = {'mean': mean, 'lower': lower, 'upper': upper}

            # --- 4. Combined Plot (Aggregated) ---
            plot_combined_log_regret(adaptive_stats_map, baseline_stats_map, 
                                   test_func.name, test_func.dim, lr, CONFIG['out_dir'])

if __name__ == "__main__":
    ist = pytz.timezone('Asia/Kolkata')
    start = datetime.now(ist)
    print(f"Started at: {start.strftime('%H:%M:%S')}")
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    print(f"Ended at: {datetime.now(ist).strftime('%H:%M:%S')}")