import torch
import numpy as np
from tqdm import tqdm
from onlinekernel import OnlineKernelSelector
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions.errors import ModelFittingError

from config import CONFIG

def run_adaptive_bo(kernels, test_function, n_init, init_x, init_y, acq_func, loss_function, lr, beta, num_steps):
    selector = OnlineKernelSelector(
        kernels=kernels,
        bounds=test_function.bounds,
        learning_rate=lr,
        n_init=n_init,
        init_x=init_x,
        init_y=init_y,
        acq=acq_func,
        beta=beta
    )

    prob_history = [selector.probabilities.copy()]
    best_y_history = [selector._y_observed.max().item()]
    losses_history = []
    
    # Early elimination threshold from PDF
    THRESHOLD = 0.99

    for i in tqdm(range(num_steps), desc="Adaptive BO", leave=False):
        # 1. Select Point
        if CONFIG["mixture"]:
            x_new = selector.select_next_point_mixed(test_function.bounds)
        else:
            x_new = selector.select_next_point(test_function.bounds)
            
        # 2. Query Objective
        y_new = test_function.true_function(x_new).view(-1)
        
        # 3. Process & Update
        losses = selector.process_new_data_point(x_new, y_new, loss_function)
        
        # Recording
        prob_history.append(selector.probabilities.copy())
        best_y_history.append(selector._y_observed.max().item())
        losses_history.append(losses)
        
        # Early Elimination Rule (PDF Algorithm 1, Line 17)
        if len(selector.kernels) > 1:
            max_p = np.max(selector.probabilities)
            if max_p > THRESHOLD:
                # Force weight of dominant kernel to 1, others to 0
                max_idx = np.argmax(selector.probabilities)
                new_weights = np.zeros_like(selector.weights)
                new_weights[max_idx] = 1.0
                selector.weights = new_weights
                selector.probabilities = new_weights
                # In strict implementation, we might physically remove other kernels to save compute, 
                # but setting weights to 0 achieves the mathematical result.

    return {
        'prob_history': prob_history,
        'best_y': best_y_history,
        'losses': losses_history
    }

def run_single_kernel_bo(kernel, test_function, n_init, init_x, init_y, acq_func, beta, num_steps):
    """Vanilla BO baseline."""
    X = init_x.clone()
    Y = init_y.clone().unsqueeze(-1)
    
    best_y_history = [Y.max().item()]
    
    for _ in tqdm(range(num_steps), desc="Single Kernel BO", leave=False):
        model = SingleTaskGP(X, Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError:
            mll = mll
        
        if acq_func == 'ei':
            acq = ExpectedImprovement(model, best_f=Y.max())
        else:
            acq = UpperConfidenceBound(model, beta=beta)
            
        try:
            candidate, _ = optimize_acqf(acq, test_function.bounds, q=1, num_restarts=5, raw_samples=100)
        except Exception:
            candidate = test_function.bounds[0] + (test_function.bounds[1] - test_function.bounds[0]) * torch.rand(1, test_function.dim)

        y_new = test_function.true_function(candidate).view(-1, 1)
        X = torch.cat([X, candidate])
        Y = torch.cat([Y, y_new])
        best_y_history.append(Y.max().item())
        
    return best_y_history