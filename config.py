import gpytorch
import numpy as np
from test_functions import (
    Ackley, Eggholder, Levy, Rastrigin, Griewank, Branin, SineCosine, 
    Sphere, Zakharov, DixonPrice, Rosenbrock, Michalewicz, Linear, 
    Periodic, Hartman3D, Hartman6D
)

# Reproducibility
rng = np.random.Generator(np.random.PCG64(42))
seeds = rng.integers(0, 1000, size=10)

CONFIG = {
    # --- Functions to Test ---
    "test_functions": [
        # Ackley(dim=2, maximize=True),
        # Eggholder(maximize=True),
        # Levy(dim=3, maximize=True),
        # Rastrigin(maximize=True, dim=4),
        # Griewank(maximize=True, dim=5),
        # Rosenbrock(dim=2, maximize=True),
        
        #high dim test
        Ackley(maximize=True, dim=8),
        Levy(maximize=True, dim=10)

    ],

    # --- Kernels ---
    "candidate_kernels": [
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)),
        # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()),
        # gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()),
    ],

    # --- Experiment Settings ---
    "seed_list": seeds.tolist(),
    "learning_rates": [1.0],
    
    # Loss ID Mapping: 
    # 1: Close (L1), 3: NLL, 7: Equal, 8: Random, 9: CRPS, 10: Brier, 11: EI Calib
    "loss_functions": [12, 3, 7, 9, 10, 11, 13], 

    "mixture": False,        # True = Convex Mixture, False = Thompson Sampling
    "scale_losses": True,    # Keep scaling logic for numerical stability

    # --- BO Parameters ---
    "num_bo_steps": 100,
    "n_init_factor": 3,
    "acq_func": "ei",
    "ucb_beta": 1.0,

    # --- System ---
    "out_dir": "experiments_final_v1_thompson",
    "curr_iter": 0,
    "force_rerun": False 
}