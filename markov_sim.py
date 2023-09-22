import numpy as np
import pandas as pd
import argparse

from lmfit import Parameter


class MarkovSimulator:
    def __init__(self, param_configs, cov_matrix = None):
        """
        param_configs: Dictionary of parameter configurations
        e.g.
        {
            'param_name1': {'value': x, 'min': a, 'max': b, 'vary': True},
            'param_name2': {'value': y, 'vary': False, 'expr': 'param_name1*2'},
            ...
        }
        """
        self.parameters = {}
        if cov_matrix is None:
            cov_matrix = np.identity(len(param_configs))
        self.cov_matrix = cov_matrix
        # Creating the lmfit Parameter objects based on the configurations
        for name, config in param_configs.items():
            self.parameters[name] = Parameter(name=name, **config)
            
        #self.cov_matrix = cov_matrix
        self.history = []


    def likelihood(self, params, fn = None, X=None, y=None):
        """Computes the likelihood of the parameters."""
        if fn is None:
            diff = params - np.array([p.value for p in self.parameters.values()])
            likelihood = np.exp(-0.5 * np.dot(diff, np.linalg.inv(self.cov_matrix).dot(diff)))
        else:
            likelihood = fn(params, X,y)
        print(likelihood)
        return likelihood
    def update_param(self, param):
        """Updates the parameter based on a random walk."""
        step = np.random.normal(0, 0.1)
        param.value = min(max(param.value + step, param.min), param.max)
    def run_simulation(self, n_steps, fn=None, X=None, y=None):
        accepted = 0
        current_log_lh = self.likelihood([p.value for p in self.parameters.values()], fn=fn, X=X, y=y)
        
        for _ in range(n_steps):
            proposed_params = [p.value + np.random.normal(0, 0.1) for p in self.parameters.values()]
            proposed_log_lh = self.likelihood(proposed_params, fn=fn, X=X, y=y)
            
            alpha = min(0, proposed_log_lh - current_log_lh)  # Since we're working with logs
            if np.log(np.random.rand()) < alpha:
                accepted += 1
                for name, param in zip(self.parameters.keys(), proposed_params):
                    self.parameters[name].value = param
                current_log_lh = proposed_log_lh
                self.history.append({name: param for name, param in zip(self.parameters.keys(), proposed_params)})
        
        print(f"Accepted Steps: {accepted}/{n_steps}")
        return pd.DataFrame(self.history)

    def propose_step(self):
        """Proposes a step based on the covariance matrix."""
        return np.random.multivariate_normal(self.params, self.cov_matrix)

    def metropolis_step(self):
        """Performs one Metropolis-Hastings step."""
        proposed_params = self.propose_step()
        alpha = min(1, self.likelihood(proposed_params) / self.likelihood(self.params))
        if np.random.rand() < alpha:
            self.params = proposed_params
            
import emcee

class MarkovSimulatorEmcee:
    def __init__(self, param_configs):
        self.parameters = {name: Parameter(name=name, **config) for name, config in param_configs.items()}
        self.ndim = len(self.parameters)

    def log_prior(self, params):
        for param, value in zip(self.parameters.values(), params):
            if not param.min <= value <= param.max:
                return -np.inf  # Log of 0
        return 0  # Log of 1

    def log_likelihood(self, params, fn, X, y):
        if fn is None:
            diff = params - np.array([p.value for p in self.parameters.values()])
            return -0.5 * np.dot(diff, diff)
        else:
            return fn(params, X, y)

    def log_posterior(self, params, fn, X, y):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, fn, X, y)

    def run_simulation(self, n_steps, n_walkers, fn, X, y):
        initial_positions = [np.array([p.value for p in self.parameters.values()]) + 1e-4*np.random.randn(self.ndim) for i in range(n_walkers)]
        sampler = emcee.EnsembleSampler(n_walkers, self.ndim, self.log_posterior, args=[fn, X, y])
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        return sampler.chain

# Assuming you've already defined the Parameter class and MarkovSimulator from your previous code...

# Generate some sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2.5 * x + 1.5 + np.random.normal(0, 2, 100)

# Initialize parameter configurations
param_configs = {
    'm': {'value': 1.0, 'min': -10, 'max': 10, 'vary': True},
    'c': {'value': 0.0, 'min': -10, 'max': 10, 'vary': True},
    'sigma': {'value': 2.0, 'min': 0.1, 'max': 10, 'vary': True}
}
# When running the simulation, make sure to pass the correct likelihood function and data

def linear_regression_likelihood(params, X, y):
    if len(params) != 3:
        raise ValueError(f"Expected 3 parameters for linear regression (intercept, slope, sigma), but got {len(params)}")
    
    intercept, slope, sigma = params
    y_pred = intercept + X * slope
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    
    # Assuming Gaussian errors
    n = len(y)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2) - rss / (2 * sigma**2)
    return log_likelihood
simulator = MarkovSimulator(param_configs)

history = simulator.run_simulation(100000, fn=linear_regression_likelihood, X=x, y=y)



##emcee example

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee

# Generate some synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 3 * x + 2 + np.random.normal(0, 2, size=len(x))

param_configs = {
    'intercept': {'value': 1.0, 'min': -10, 'max': 10},
    'slope': {'value': 1.0, 'min': 0, 'max': 5},
    'sigma': {'value': 1.5, 'min': 0.1, 'max': 5}
}

simulator = MarkovSimulatorEmcee(param_configs)
chain = simulator.run_simulation(n_steps=5000, n_walkers=32, fn=linear_regression_likelihood, X=x, y=y)

# Taking the last 100 steps of each walker
samples = chain[:, -100:, :].reshape((-1, simulator.ndim))

# Plotting the samples
plt.figure(figsize=(10, 6))
for i, param_name in enumerate(simulator.parameters):
    plt.subplot(simulator.ndim, 1, i+1)
    plt.hist(samples[:, i], bins=50, alpha=0.6)
    plt.title(f"Distribution for {param_name}")
plt.tight_layout()
plt.show()




















def leaky_box_model(params, dt, M_prev, MZ_prev):
    """Simulate the leaky box model given a set of parameters."""
    M_galaxy, M_in, SFR, Z_in, M_out = params
    dM = M_in - SFR - M_out
    Z_prev = MZ_prev / M_prev
    dMZ = Z_in * M_in - Z_prev * SFR
    return M_prev + dM*dt, MZ_prev + dMZ*dt

# Example Usage
param_configs = {
    'log_M_total': {'value': np.log10(1e10), 'min': np.log10(1e9), 'max': np.log10(1e11), 'vary': True, 'expr': 'log10(10**log_M_stellar + 10**log_M_gas + 10**log_M_in - 10**log_M_out)'},
    'log_M_stellar': {'value': np.log10(1e9), 'min': np.log10(1e8), 'max': np.log10(1e10), 'vary': True, 'expr': 'log10(10**log_SFR * dt)'},
    'log_M_gas': {'value': np.log10(9e9), 'min': np.log10(1e8), 'max': np.log10(1e10), 'vary': True, 'expr': 'log10(10**log_M_in - 10**log_M_out - 10**log_SFR * dt)'},
    'log_M_in': {'value': np.log10(1e9), 'min': np.log10(1e8), 'max': np.log10(1e10), 'vary': True},
    'log_SFR': {'value': np.log10(1), 'min': np.log10(1e-4), 'max': np.log10(1e4), 'vary': True, 'expr': ''} ,
    'Z_in': {'value': np.log10(0.01), 'min': -6, 'max': 0, 'vary': False},
    'log_M_out': {'value': np.log10(1e9), 'min': np.log10(1e8), 'max': np.log10(1e10), 'vary': True},

}
param_configs.update({
    'morphology': {'value': 0.5, 'min': 0, 'max': 1, 'vary': True},
    'P_merger': {'value': 0.01, 'min': 0, 'max': 1, 'vary': False},
    'AGN_activity': {'value': 0.5, 'min': 0, 'max': 1, 'vary': False},
    'log_M_out_agn': {'value': np.log10(1e8), 'min': np.log10(1e7), 'max': np.log10(1e9), 'vary': True, 'expr': 'log10(beta * AGN_activity)'},
    'log_M_out_stellar': {'value': np.log10(1e8), 'min': np.log10(1e7), 'max': np.log10(1e9), 'vary': True, 'expr': 'log10(gamma * log_SFR)'},
    'log_M_out': {'value': np.log10(2e8), 'min': np.log10(1e7), 'max': np.log10(2e9), 'vary': True, 'expr': 'log10(10**log_M_out_agn + 10**log_M_out_stellar)'
                  
                  },
    'log_sigma': {'value': 2, 'min': 1, 'max': 3, 'vary': True, 'expr': 'log_sigma + alpha*log_M_out'},
    'log_SFE': {'value': -9, 'min': -12, 'max': -6, 'vary': True, 'expr': 'log_SFR - log_M_gas'}
    
    
    # Just an example range, adjust as needed
})

# Modify the existing log_SFR expression to incorporate efficiency and morphology
threshold = 2.5  # This is an example threshold for the velocity dispersion's impact on efficiency. Adjust as needed.
efficiency_expr = f"log(exp(-log_sigma / {threshold}))"
morphology_factor_expr = "0.5 + 0.5 * morphology"
new_SFR_expr = f"({param_configs['log_SFR']['expr']} + {efficiency_expr}) * {morphology_factor_expr}"

param_configs['log_SFR']['expr'] = new_SFR_expr

simulator = MarkovSimulator(param_configs)

# Assuming X as time intervals (like [0.1, 0.1, ...] for time steps of 0.1 Gyr) and 
# y as observed data ([M_obs, MZ_obs]) for each time step.

history = simulator.run_simulation(100000)

import matplotlib.pyplot as plt
plt.ion()
def plot2dhist(x, y, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.hist2d(x, y, **kwargs)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_aspect('equal')
    return ax
