#%%[markdown]
# # pieinsky Example
# John Kucharski | johnkucharski@gmail.com | 25 August 2021
#
# This file provides a conceptual model for the *optimal allocation of random samples from a stratified population with hetergenous sampling costs across the strata*.
# The intended use case is sampling trace data from exploratory scenarios with variable computing costs across the scenarios. 
#
# # TODO:
# (1) Some shortcuts and wierdness in the threshold calculations and allocation of traces versus samples.
# (2) General checking and testing needs to be done, there could be errors (even fundemental ones). Compare actual results to theoretical (difference could be easily be initial sampling if its not large).
# (3) How does this translate in a multi-objective problem. Might want to deal with each threshold indepenently (but then what to do with time constraint). 
#

#%%
import math
import scipy.stats
import numpy as np
from scipy.stats import stats

import time
import typing


#%%[markdown]
# ## Distributions
# 8 normal distributions with a different means and variances are generated below. 
# Actually, there are 4 distribution each of which is duplicated one time so that the cost can vary across the duplicates.
# The variability in the means should have no impact on the sample allocation but better represents the use case.
# Sampling from these distributions simulates the retrieval of a performance metric from simulation model forced with a trace from an exploratory scenario. 
#%%
mu, sigma = [0.50, 0.50, 0.25, 0.25], [0.10, 0.20] 
strata = [] #lolo x 2, lohi x 2, hilo x2 , hihi x 2
for i in range(len(mu)):
    for j in range(len(sigma)):
        strata.append(scipy.stats.norm(mu[i], sigma[j]))

# %%[markdown]
# ## Allocation Function
# Optimal allocation (across h strata) without consideration of cost follows
# (1) $n_h = n\cdot\frac{N_h\sqrt{s_h}}{\sum_{h=1}^{k}N_h\sqrt{s_h}}$
# where $n$ is the total sample size, $n_h$ is the allocation of sample observations to strata $h$, $N_h$ is the population size of strata $h$, and $s_h$ is the variance in strata $h$.
# Therefore the allocation of observations to each strata $h$ is in propotion to the total expected variance contained in that strata.
# 
# The total cost of sampling is given by
# (2) $C = c_o + \sum_{h=1}^{k}c_hn_h$
# where $C$ is the total cost, $c_o$ if a fixed cost and $c_h$ is the cost per observation in strata $h$.
#
# For a fixed sampling budget, and variable cost within stata the optimal allocation across strata is
# (3) $n_h = \frac{(C - c_o)N_h \frac{\sqrt{s_h}}{\sqrt{c_h}}} {\sum_{h=1}^{k}(N_h\sqrt{s_h}\sqrt{c_h})}$
# Therefore the allocaiton of observations to each strata $h$ is in portion to the total experted variance *per unit of cost* contained in that strata.
#

# %%[markdown]
# ## Example set up
# c_o: fixed cost = 0 seconds
# N: sample size = 8 strata (i.e. scenarios) containing 100 traces. Each trace contains 100 years produces 100 metrics of interest, so N = 80,000. 
# strata_n: strata size = there is an equal number of traces per strata (i.e. $\frac{N}{8} = 100$metric observations) per strata.
# c: strata (computational) cost = it is assumed that even numbered strata take 0.01 seconds per trace (or simulation) to run, odd numbered stata take double that time: 0.2 seconds per trace to run. *Note*, the simulation runs 100 years (1 trace = 100 years) at a time.
# budget = 100 seconds. 
# n: total sample size (years to run) = the sample size, $n \in [5,000, 10,000]$ (i.e. $n_{min} = 5,000 = \frac{100}{1}\frac{1}{0.2}\frac{100}{1}$ [s(trace/s)(yr/trace)] and $n_{max} = 10,000 = \frac{100}{1}\frac{1}{0.1}\frac{100}{1}$ [s/(trace/s)(yr/trace)]).
# %%
c_o = 0
N = 8 * 100 
strata_N, c = [N/8 for i in range(len(strata))], [0.1 if i % 2 else 0.2 for i in range(len(strata))]
total_cost = c_o + sum([c[h] * strata_N[h] for h in range(len(strata))])

# %%[markdown]
# ## Simulation Simulation
# The function below samples 100 numbers at a time and takes $n_h$ seconds to run (simulating the computational cost of a simulation model).
# %%
def simulation_simulation(h: int, strata: typing.List = strata, c: typing.List[float] = c) -> typing.List:
    start = time.time()
    x = strata[h].rvs(size=100)
    time.sleep(c[h] - (time.time() - start))
    return x.tolist()

# %% [markdown]
# ## Wrappper Example
#
# %%
class Wrapper:
    def __init__(self, strata: typing.List, budget: float, metric_threshold: float, 
                 count_strata: typing.List[int] = strata_N, updates: int = 10,
                 cost_init: float = c_o, cost_strata: typing.List[float] = c):
        self.budget = budget
        self.strata = strata
        self.strata_N = count_strata
        self.updates = updates
        self.cost_init = cost_init
        self.cost_strata = cost_strata
        self.cost_total = self.total_cost()
        self.threshold = metric_threshold
    
    def total_cost(self):
        '''
        Computes the expected total compute time of sampling the entire population.
        '''
        return self.cost_init + sum([self.cost_strata[h] * self.strata_N[h] for h in range(len(self.strata))]) 
    
    def failures(self, sample: typing.List[float]) -> typing.List[int]:
        '''
        Replaces strata_metrics with 1s where threshold < strata_metric and 0s where threshold >= strata_metric. 
        '''
        # assumes below threshold is failure above threshold is success
        return [1 if i < self.threshold else 0 for i in sample]
    
    def allocate(self, samples: typing.List[typing.List[float]], budget: float) -> typing.List[int]:
        '''
        Returns a list of integers representing the optimal allocation of the sample based on a fixed maximum sampling cost.
        '''
        def n_h(h) -> float:
            return ((budget - self.cost_init) * self.strata_N[h] * np.std(self.failures(samples[h])) / np.sqrt(self.cost_strata[h])) / sum([self.strata_N[i] * np.std(self.failures(samples[i])) for i in range(len(self.strata))]) 
        return [np.ceil(n_h(h)) for h in range(len(self.strata))]
    
    def pieinsky(self, simulation: typing.Callable):
        '''
        Samples from the strata based on optimal allocation with variable cost.
        '''
        clock, start, updates = 0, time.time(), self.updates
        print(f'{updates} updates remaining.')
        samples = [simulation(h, self.strata) for h in range(len(self.strata))]
        clock += time.time() - start
        print(f'--initial sample retreived in {clock} seconds.')
        updates -= 1
        allocation = self.allocate(samples, (self.budget - clock) / updates)
        
        while updates > 0:
            print(f'{updates} updates remaining, {int(self.budget - clock)} seconds remaining.')
            start = time.time()
            for h in range(len(allocation)):
                if len(samples[h]) < int(allocation[h] * 100):
                    print(f'strata {h}: contains {len(samples[h])} of {int(allocation[h] * 100)} samples needed.')
                    while len(samples[h]) < allocation[h] * 100:
                        samples[h] += simulation(h, strata)
                        print(f'--{len(samples[h])} samples retreived.') 
            clock += time.time() - start
            allocation = self.allocate(samples, (self.budget - clock) / updates)
            updates -= 1
        return samples
        
# %%
wrapper = Wrapper(strata, budget = 100, metric_threshold = 0.10)
x = wrapper.pieinsky(simulation_simulation)
