import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

CB_COLORS = sns.color_palette("colorblind")
MARKERS = ['o', 's', 'X', 'D', 'P', '^', 'v', '*']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

LOSS_MAP = {
    1: "L1 Error",
    2: "ExpProb",
    3: "NLL",
    7: "Equal Wts",
    8: "Random",
    9: "CRPS",
    10: "Brier Score",
    11: "EI Calibration",
    12: "BR + CRPS",
    13: "BR + NLL"  
}

def plot_seed_dashboard(adaptive_data, baseline_map, kernel_names, 
                        optimum, path, func_name, dim, loss_id, lr, seed_idx):
    """
    1x3 Grid: [Log Regret] [Probabilities] [Losses]
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    loss_name = LOSS_MAP.get(loss_id, f"ID:{loss_id}")
    iterations = range(len(adaptive_data['best_y']))
    
    # --- 1. Log Regret ---
    ax = axes[0]
    for i, name in enumerate(kernel_names):
        if name in baseline_map:
            base_y = baseline_map[name]
            regret = np.maximum(optimum - np.array(base_y), 1e-10)
            log_regret = np.log10(regret)
            ax.plot(log_regret, label=name, color=CB_COLORS[i % len(CB_COLORS)],
                    linestyle='--', alpha=0.6, linewidth=1.5)

    adap_regret = np.maximum(optimum - np.array(adaptive_data['best_y']), 1e-10)
    ax.plot(np.log10(adap_regret), label=f"Adaptive ({loss_name})", 
            color='black', linewidth=2.5, marker='o', markersize=3, markevery=5)
    
    ax.set_title("Log Regret")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log10 Regret")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize='x-small', loc='upper right')

    # --- 2. Probabilities ---
    ax = axes[1]
    probs = np.array(adaptive_data['prob_history'])
    for i, name in enumerate(kernel_names):
        ax.plot(probs[:, i], label=name, 
                color=CB_COLORS[i % len(CB_COLORS)], 
                linewidth=2, alpha=0.9)
    ax.set_title(f"Kernel Probabilities")
    ax.set_xlabel("Iteration")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- 3. Step Losses ---
    ax = axes[2]
    losses = np.array(adaptive_data['losses'])
    for i, name in enumerate(kernel_names):
        ax.plot(losses[:, i], label=name,
                color=CB_COLORS[i % len(CB_COLORS)],
                linewidth=1.5, alpha=0.8)
    ax.set_title(f"Step Losses")
    ax.set_xlabel("Iteration")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.suptitle(f"Seed {seed_idx} | {func_name} ({dim}D) | LR: {lr} | Loss: {loss_name}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def plot_combined_log_regret(adaptive_stats_map, baseline_stats_map, func_name, dim, lr, out_dir):
    plt.figure(figsize=(12, 8))
    
    for i, (loss_id, stats) in enumerate(adaptive_stats_map.items()):
        color = CB_COLORS[i % len(CB_COLORS)]
        iterations = range(len(stats['mean']))
        loss_name = LOSS_MAP.get(loss_id, str(loss_id))
        
        plt.plot(iterations, stats['mean'], color=color, linewidth=2.5, 
                 linestyle='-', marker='o', markersize=4, markevery=5,
                 label=f'Adaptive ({loss_name})')
        plt.fill_between(iterations, stats['lower'], stats['upper'], color=color, alpha=0.15)

    offset = len(adaptive_stats_map)
    for i, (kernel_name, stats) in enumerate(baseline_stats_map.items()):
        color = CB_COLORS[(i + offset) % len(CB_COLORS)]
        marker = MARKERS[(i + offset) % len(MARKERS)]
        iterations = range(len(stats['mean']))
        
        plt.plot(iterations, stats['mean'], color=color, linewidth=1.5, 
                 linestyle='--', marker=marker, markersize=4, markevery=5,
                 label=f'{kernel_name}', alpha=0.8)
        plt.fill_between(iterations, stats['lower'], stats['upper'], color=color, alpha=0.1)

    plt.title(f"Aggregated Log Regret | {func_name} ({dim}D) | LR: {lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Log10 Regret (Mean ± 95% CI)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.6, linestyle=':')
    plt.tight_layout()
    
    folder = os.path.join("results", out_dir, "plots", f"{func_name}_d{dim}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"Comparison_LogRegret_lr{lr}.png")
    plt.savefig(path)
    plt.close()