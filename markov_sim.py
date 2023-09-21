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

