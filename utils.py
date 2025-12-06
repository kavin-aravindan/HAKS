import os
import pickle
import hashlib
import numpy as np
import scipy.stats as stats

def get_kernel_name(kernel):
    if hasattr(kernel, 'base_kernel'):
        return kernel.base_kernel.__class__.__name__
    return kernel.__class__.__name__

def calculate_log_regret_stats(best_y_history_list, optimum_value):
    best_y = np.array(best_y_history_list)
    epsilon = 1e-10
    simple_regret = np.maximum(optimum_value - best_y, epsilon)
    log_regret = np.log10(simple_regret)
    
    mean = np.mean(log_regret, axis=0)
    std_err = stats.sem(log_regret, axis=0)
    h = std_err * stats.t.ppf((1 + 0.95) / 2., len(log_regret) - 1)
    
    return mean, mean - h, mean + h

def generate_filename(config, test_func, identifier_string):
    unique_str = f"{test_func.name}_{test_func.dim}_{config['num_bo_steps']}_{config['acq_func']}_{config['mixture']}"
    if "Kernel" not in identifier_string: 
        kernel_names = [get_kernel_name(k) for k in config['candidate_kernels']]
        unique_str += "_".join(kernel_names)
    run_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]
    return f"{test_func.name}_d{test_func.dim}_{identifier_string}_{run_hash}.pkl"

def save_data(data, config, test_func, identifier_string):
    directory = os.path.join("results", config["out_dir"], "data_cache")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, generate_filename(config, test_func, identifier_string))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"  [Data Saved] {identifier_string}")

def load_data(config, test_func, identifier_string):
    directory = os.path.join("results", config["out_dir"], "data_cache")
    filepath = os.path.join(directory, generate_filename(config, test_func, identifier_string))
    if os.path.exists(filepath) and not config.get('force_rerun', False):
        print(f"  [Data Loaded] {identifier_string}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def get_dashboard_path(out_dir, func_name, dim, loss_id, lr, seed):
    """
    Returns path: results/out_dir/plots/Func_dDim/Dashboards/L{loss}_lr{lr}/Seed_{seed}.png
    """
    subfolder = f"L{loss_id}_lr{lr}"
    folder = os.path.join("results", out_dir, "plots", f"{func_name}_d{dim}", "Dashboards", subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"Seed_{seed}.png")