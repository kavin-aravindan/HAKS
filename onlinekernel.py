import numpy as np
import torch
import gpytorch
import warnings
import copy
import math

from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.acquisition import (
    UpperConfidenceBound,
    ExpectedImprovement,
    LogExpectedImprovement,
    AnalyticAcquisitionFunction,
)
from linear_operator.utils.errors import NotPSDError
from config import CONFIG

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


class GPModel(gpytorch.models.ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self._num_outputs = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MixedAcq(AnalyticAcquisitionFunction):
    def __init__(self, models, weights, acq_type="ucb", beta=2.0, y_obs=None):
        # y_obs is assumed to be in the same (normalized) scale as models' outputs
        super().__init__(models[0])
        self.models = models
        self.weights = weights
        self.acq_type = acq_type
        self.beta = beta
        self._y_obs = y_obs  # normalized best-so-far

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        acq_values = torch.zeros(X.shape[:-2].numel(), device=X.device, dtype=X.dtype)

        if self._y_obs is None or self._y_obs.numel() == 0:
            best_f = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        else:
            best_f = self._y_obs.max()

        for i, model in enumerate(self.models):
            if self.acq_type == "logEI":
                acq = LogExpectedImprovement(model, best_f=best_f)
            elif self.acq_type == "ei":
                acq = ExpectedImprovement(model, best_f=best_f)
            else:
                acq = UpperConfidenceBound(model, beta=self.beta)
            acq_values += self.weights[i] * acq(X)

        return acq_values


class OnlineKernelSelector:
    def __init__(
        self,
        kernels,
        bounds,
        learning_rate=0.5,
        n_init=0,
        init_x=None,
        init_y=None,
        acq="ucb",
        beta=2.0,
    ):
        self.kernels = copy.deepcopy(kernels)
        self.num_kernels = len(self.kernels)
        self.learning_rate = learning_rate
        self.bounds = bounds
        self.bounds_min = self.bounds[0]
        self.bounds_range = self.bounds[1] - self.bounds[0]

        self.models = []
        self.likelihoods = []
        self.weights = np.ones(self.num_kernels)
        self.probabilities = self.weights / np.sum(self.weights)
        self.beta = beta
        self.acq = acq

        # Output normalization (consistent across training, acquisitions, losses)
        self.y_mean = None
        self.y_std = None

        # Global scaling constants for bounded losses
        self.abs_err_max = 3.0        # |error| >= 3σ is "maximal" for AE loss (normalized units)
        self.nll_min = 0.0            # NLL clipping range (normalized y)
        self.nll_max = 10.0
        self.crps_max = 2.0           # CRPS clipping (normalized y)
        self.ei_max = 3.0             # improvement (normalized units)
        self.crps_brier_alpha = 0.5   # convex combo weight for CRPS–Brier loss

        if n_init > 0 and init_x is not None:
            self._X_observed = init_x.clone()
            self._y_observed = init_y.clone()
            self._initialize_models()
            self._fit_all_models(full_retrain=True)
        else:
            self._X_observed = torch.empty((0, bounds.shape[1]))
            self._y_observed = torch.empty((0,))

    # ----------------- Normalization helpers -----------------

    def _normalize_x(self, x):
        return (x - self.bounds_min) / self.bounds_range

    def _unnormalize_x(self, x_norm):
        return x_norm * self.bounds_range + self.bounds_min

    # ----------------- GP setup & training -----------------

    def _initialize_models(self):
        self.models = []
        self.likelihoods = []
        for kernel in self.kernels:
            k = copy.deepcopy(kernel)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = GPModel(self._X_observed, self._y_observed, likelihood, k)
            self.models.append(model)
            self.likelihoods.append(likelihood)

    def _train_model(self, model, likelihood, training_iter=50):
        # Assumes model.train_targets are already normalized
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            try:
                output = model(model.train_inputs[0])
                loss = -mll(output, model.train_targets)
                loss.backward()
                optimizer.step()
            except (NotPSDError, RuntimeError):
                break

    def _fit_all_models(self, full_retrain=False):
        # Normalize X
        train_x_norm = self._normalize_x(self._X_observed)

        # Compute and store y-normalization
        if self._y_observed.numel() > 0:
            y_mean = self._y_observed.mean()
            y_std = self._y_observed.std()
            if y_std <= 1e-6:
                y_std = torch.tensor(1.0, dtype=y_mean.dtype, device=y_mean.device)
            else:
                y_std = y_std.to(y_mean.device)
        else:
            # No data yet
            y_mean = torch.tensor(0.0, dtype=train_x_norm.dtype, device=train_x_norm.device)
            y_std = torch.tensor(1.0, dtype=train_x_norm.dtype, device=train_x_norm.device)

        self.y_mean = y_mean
        self.y_std = y_std

        if self._y_observed.numel() > 0:
            train_y_norm = (self._y_observed - self.y_mean) / self.y_std
        else:
            train_y_norm = self._y_observed  # empty

        for model, likelihood in zip(self.models, self.likelihoods):
            model.set_train_data(train_x_norm, train_y_norm, strict=False)
            if full_retrain:
                self._train_model(model, likelihood)

    # ----------------- Acquisition selection -----------------

    def select_next_point(self, bounds):
        """Select next point using single-kernel acquisition."""
        if self.num_kernels == 0:
            raise RuntimeError("No kernels/models initialized.")

        chosen_idx = np.random.choice(self.num_kernels, p=self.probabilities)
        model = self.models[chosen_idx]

        dim = bounds.shape[1]
        unit_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(bounds.device)

        if self.acq == "ei":
            # best_f must be in normalized y space
            if self._y_observed.numel() == 0 or self.y_mean is None or self.y_std is None:
                best_f = 0.0
            else:
                best_raw = self._y_observed.max()
                best_f = ((best_raw - self.y_mean) / self.y_std).item()
            acq = ExpectedImprovement(model, best_f=best_f)
        elif self.acq == "logEI":
            if self._y_observed.numel() == 0 or self.y_mean is None or self.y_std is None:
                best_f = 0.0
            else:
                best_raw = self._y_observed.max()
                best_f = ((best_raw - self.y_mean) / self.y_std).item()
            acq = LogExpectedImprovement(model, best_f=best_f)
        else:
            acq = UpperConfidenceBound(model, beta=self.beta)

        return self._optimize_acq(acq, unit_bounds)

    def select_next_point_mixed(self, bounds):
        """Select next point using a convex mixture of acquisitions over kernels."""
        if self.num_kernels == 0:
            raise RuntimeError("No kernels/models initialized.")

        weights_t = torch.tensor(self.probabilities, dtype=torch.float64)
        dim = bounds.shape[1]
        unit_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(bounds.device)

        # y_obs for acquisitions must be in normalized scale
        if self._y_observed.numel() > 0 and self.y_mean is not None and self.y_std is not None:
            y_obs_norm = (self._y_observed - self.y_mean) / self.y_std
        else:
            y_obs_norm = torch.empty_like(self._y_observed)

        mixed_acq = MixedAcq(self.models, weights_t, acq_type=self.acq, beta=self.beta, y_obs=y_obs_norm)
        return self._optimize_acq(mixed_acq, unit_bounds)

    def _optimize_acq(self, acq_func, bounds):
        try:
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=100,
            )
        except Exception:
            candidate = torch.rand(1, bounds.shape[1])

        return self._unnormalize_x(candidate).detach()

    # ----------------- Online update -----------------

    def process_new_data_point(self, x_new, y_new, loss_function=1):
        # Calculate losses (before adding the new point)
        losses = self._calculate_losses(x_new, y_new, loss_function)

        # Update weights
        self._update_weights(losses)

        # Append new data
        self._X_observed = torch.cat([self._X_observed, x_new.reshape(1, -1)])
        self._y_observed = torch.cat([self._y_observed, y_new.reshape(1,)])

        # Retrain models on updated dataset (with normalization)
        self._fit_all_models(full_retrain=True)
        return losses

    # ----------------- Loss computation -----------------

    def _calculate_losses(self, x_new, y_new, loss_function):
        losses = []

        # If no data or no models yet, return zeros → equal weights
        if self._y_observed.numel() == 0 or len(self.models) == 0:
            return np.zeros(self.num_kernels, dtype=float)

        # Use the same normalization as training
        if self.y_mean is None or self.y_std is None:
            y_mean = self._y_observed.mean()
            y_std = self._y_observed.std()
            if y_std <= 1e-6:
                y_std = torch.tensor(1.0, dtype=y_mean.dtype, device=y_mean.device)
        else:
            y_mean = self.y_mean
            y_std = self.y_std

        # Normalize new observation
        y_new_normalized = (y_new - y_mean) / y_std

        # Normalize input
        x_new_norm = self._normalize_x(x_new)

        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(x_new_norm))

                # ---- LOSS 1: Absolute Error (in normalized units, bounded) ----
                if loss_function == 1:
                    # model outputs normalized y; error in normalized space is |y_new_norm - mu|
                    mu_norm = pred_dist.mean
                    abs_err_norm = torch.abs(y_new_normalized - mu_norm)
                    abs_err_norm = torch.nan_to_num(
                        abs_err_norm, nan=self.abs_err_max,
                        posinf=self.abs_err_max, neginf=self.abs_err_max
                    )
                    abs_err_norm = torch.clamp(abs_err_norm, max=self.abs_err_max)
                    loss_t = abs_err_norm / self.abs_err_max
                    loss = float(loss_t.item())

                # ---- LOSS 3: NLL (on normalized scale, clipped and scaled) ----
                elif loss_function == 3:
                    raw_nll = -pred_dist.log_prob(y_new_normalized)
                    raw_nll = torch.nan_to_num(
                        raw_nll,
                        nan=self.nll_max,
                        posinf=self.nll_max,
                        neginf=self.nll_min,
                    )
                    raw_nll = torch.clamp(raw_nll, min=self.nll_min, max=self.nll_max)
                    loss_t = (raw_nll - self.nll_min) / (self.nll_max - self.nll_min)
                    loss = float(loss_t.item())

                # ---- LOSS 7: Equal Weights (No Op) ----
                elif loss_function == 7:
                    loss = 0.0

                # ---- LOSS 8: Random ----
                elif loss_function == 8:
                    loss = float(np.random.rand())

                # ---- LOSS 9: CRPS (on normalized scale, bounded & scaled) ----
                elif loss_function == 9:
                    mean = pred_dist.mean
                    std = pred_dist.stddev if hasattr(pred_dist, "stddev") else pred_dist.variance.sqrt()
                    std = torch.clamp(std, min=1e-6)

                    z = (y_new_normalized - mean) / std
                    normal0 = torch.distributions.Normal(
                        loc=torch.zeros_like(z),
                        scale=torch.ones_like(z),
                    )
                    phi = torch.exp(normal0.log_prob(z))
                    Phi = normal0.cdf(z)

                    crps = std * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
                    crps = torch.nan_to_num(
                        crps,
                        nan=self.crps_max,
                        posinf=self.crps_max,
                        neginf=0.0,
                    )
                    crps = torch.clamp(crps, min=0.0, max=self.crps_max)
                    loss_t = crps / self.crps_max
                    loss = float(loss_t.item())

                # ---- LOSS 10: Brier (probability of improvement) ----
                elif loss_function == 10:
                    if self._y_observed.numel() == 0:
                        loss = 0.0
                    else:
                        best_observed = self._y_observed.max()
                        best_f_norm = (best_observed - y_mean) / y_std

                        # Event: did we improve?
                        event = 1.0 if float(y_new) >= float(best_observed) else 0.0

                        mean = pred_dist.mean
                        std = pred_dist.stddev if hasattr(pred_dist, "stddev") else pred_dist.variance.sqrt()
                        std = torch.clamp(std, min=1e-6)

                        normal_pred = torch.distributions.Normal(mean, std)
                        threshold = torch.as_tensor(best_f_norm, dtype=mean.dtype, device=mean.device)
                        threshold = threshold.expand_as(mean)
                        q = 1.0 - normal_pred.cdf(threshold)
                        q = torch.nan_to_num(q, nan=0.5, posinf=1.0, neginf=0.0)
                        q = torch.clamp(q, 0.0, 1.0)

                        event_t = torch.tensor(event, dtype=q.dtype, device=q.device)
                        loss_t = (event_t - q) ** 2
                        loss_t = torch.nan_to_num(loss_t, nan=1.0, posinf=1.0, neginf=1.0)
                        loss_t = torch.clamp(loss_t, 0.0, 1.0)
                        loss = float(loss_t.item())

                # ---- LOSS 11: EI Calibration (improvement magnitude, bounded & scaled) ----
                elif loss_function == 11:
                    if self._y_observed.numel() == 0:
                        loss = 0.0
                    else:
                        best_observed = self._y_observed.max()
                        best_f_norm = (best_observed - y_mean) / y_std

                        # Actual improvement in normalized units:
                        # (y_new_norm - best_f_norm) = (y_new - best_observed) / y_std
                        imp_actual = y_new_normalized - best_f_norm
                        imp_actual = torch.clamp(imp_actual, min=0.0)
                        imp_actual = torch.nan_to_num(
                            imp_actual,
                            nan=self.ei_max,
                            posinf=self.ei_max,
                            neginf=0.0,
                        )
                        imp_actual = torch.clamp(imp_actual, max=self.ei_max)

                        mean = pred_dist.mean
                        std = pred_dist.stddev if hasattr(pred_dist, "stddev") else pred_dist.variance.sqrt()
                        std = torch.clamp(std, min=1e-6)

                        tau = torch.as_tensor(best_f_norm, dtype=mean.dtype, device=mean.device)
                        tau = tau.expand_as(mean)
                        z = (mean - tau) / std

                        normal0 = torch.distributions.Normal(
                            loc=torch.zeros_like(z),
                            scale=torch.ones_like(z),
                        )
                        phi = torch.exp(normal0.log_prob(z))
                        Phi = normal0.cdf(z)

                        ei = (mean - tau) * Phi + std * phi  # normalized EI
                        ei = torch.nan_to_num(ei, nan=self.ei_max, posinf=self.ei_max, neginf=0.0)
                        ei = torch.clamp(ei, min=0.0, max=self.ei_max)

                        imp_actual_t = imp_actual.to(ei.dtype).to(ei.device).expand_as(ei)
                        sq_err = (imp_actual_t - ei) ** 2
                        sq_err = torch.nan_to_num(sq_err, nan=self.ei_max**2, posinf=self.ei_max**2, neginf=0.0)
                        sq_err = torch.clamp(sq_err, 0.0, self.ei_max**2)
                        loss_t = sq_err / (self.ei_max**2)
                        loss = float(loss_t.item())

                # ---- LOSS 12: Convex combination of CRPS and Brier ----
                elif loss_function == 12:
                    if self._y_observed.numel() == 0:
                        loss = 0.0
                    else:
                        # CRPS part
                        mean = pred_dist.mean
                        std = pred_dist.stddev if hasattr(pred_dist, "stddev") else pred_dist.variance.sqrt()
                        std = torch.clamp(std, min=1e-6)

                        z = (y_new_normalized - mean) / std
                        normal0 = torch.distributions.Normal(
                            loc=torch.zeros_like(z),
                            scale=torch.ones_like(z),
                        )
                        phi = torch.exp(normal0.log_prob(z))
                        Phi = normal0.cdf(z)

                        crps = std * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
                        crps = torch.nan_to_num(
                            crps,
                            nan=self.crps_max,
                            posinf=self.crps_max,
                            neginf=0.0,
                        )
                        crps = torch.clamp(crps, min=0.0, max=self.crps_max)
                        crps_scaled = crps / self.crps_max

                        # Brier part
                        best_observed = self._y_observed.max()
                        best_f_norm = (best_observed - y_mean) / y_std
                        event = 1.0 if float(y_new) >= float(best_observed) else 0.0

                        normal_pred = torch.distributions.Normal(mean, std)
                        threshold = torch.as_tensor(best_f_norm, dtype=mean.dtype, device=mean.device)
                        threshold = threshold.expand_as(mean)
                        q = 1.0 - normal_pred.cdf(threshold)
                        q = torch.nan_to_num(q, nan=0.5, posinf=1.0, neginf=0.0)
                        q = torch.clamp(q, 0.0, 1.0)

                        event_t = torch.tensor(event, dtype=q.dtype, device=q.device)
                        brier = (event_t - q) ** 2
                        brier = torch.nan_to_num(brier, nan=1.0, posinf=1.0, neginf=1.0)
                        brier = torch.clamp(brier, 0.0, 1.0)

                        alpha = self.crps_brier_alpha
                        loss_t = alpha * crps_scaled + (1.0 - alpha) * brier
                        loss = float(loss_t.item())

                elif loss_function == 13:
                    if self._y_observed.numel() == 0:
                        loss = 0.0
                    else:
                        # NLL part
                        raw_nll = -pred_dist.log_prob(y_new_normalized)
                        raw_nll = torch.nan_to_num(
                            raw_nll,
                            nan=self.nll_max,
                            posinf=self.nll_max,
                            neginf=self.nll_min,
                        )
                        raw_nll = torch.clamp(raw_nll, min=self.nll_min, max=self.nll_max)
                        loss_t = (raw_nll - self.nll_min) / (self.nll_max - self.nll_min)
                        nll_loss = float(loss_t.item())

                        # Brier part
                        mean = pred_dist.mean
                        std = pred_dist.stddev if hasattr(pred_dist, "stddev") else pred_dist.variance.sqrt()
                        std = torch.clamp(std, min=1e-6)
                        best_observed = self._y_observed.max()
                        best_f_norm = (best_observed - y_mean) / y_std
                        event = 1.0 if float(y_new) >= float(best_observed) else 0.0

                        normal_pred = torch.distributions.Normal(mean, std)
                        threshold = torch.as_tensor(best_f_norm, dtype=mean.dtype, device=mean.device)
                        threshold = threshold.expand_as(mean)
                        q = 1.0 - normal_pred.cdf(threshold)
                        q = torch.nan_to_num(q, nan=0.5, posinf=1.0, neginf=0.0)
                        q = torch.clamp(q, 0.0, 1.0)

                        event_t = torch.tensor(event, dtype=q.dtype, device=q.device)
                        brier = (event_t - q) ** 2
                        brier = torch.nan_to_num(brier, nan=1.0, posinf=1.0, neginf=1.0)
                        brier = torch.clamp(brier, 0.0, 1.0)

                        alpha = self.crps_brier_alpha
                        loss_t = alpha * nll_loss + (1.0 - alpha) * brier
                        loss = float(loss_t.item())
                        
                else:
                    # Fallback: neutral
                    loss = 0.0

                losses.append(loss)

        losses = np.array(losses, dtype=float)
        losses = np.nan_to_num(losses, nan=1.0, posinf=1.0, neginf=1.0)

        # No per-step min-max scaling; each branch already maps into [0,1].
        return losses

    # ----------------- Hedge weight update -----------------

    def _update_weights(self, losses):
        self.weights *= np.exp(-self.learning_rate * losses)

        total_weight = np.sum(self.weights)
        if total_weight <= 1e-10 or not np.isfinite(total_weight):
            self.weights = np.ones(self.num_kernels)
            total_weight = self.num_kernels

        self.probabilities = self.weights / total_weight
