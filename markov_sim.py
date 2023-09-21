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


    def likelihood(self, params):
        """Computes the likelihood of the parameters."""
        diff = params - np.array([p.value for p in self.parameters.values()])
        return np.exp(-0.5 * np.dot(diff, np.linalg.inv(self.cov_matrix).dot(diff)))

    def update_param(self, param):
        """Updates the parameter based on a random walk."""
        step = np.random.normal(0, 0.1)
        param.value = min(max(param.value + step, param.min), param.max)
    def run_simulation(self, n_steps):
        """Runs the Markov chain simulation for a given number of steps."""
        for _ in range(n_steps):
            current_params = [p.value for p in self.parameters.values()]
            if not self.likelihood(current_params):
                continue

            for param in self.parameters.values():
                if param.vary:
                    self.update_param(param)
            
            self.history.append({name: param.value for name, param in self.parameters.items()})

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
            
