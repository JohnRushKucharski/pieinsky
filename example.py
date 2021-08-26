#%%[markdown]
# # pieinsky Example
# John Kucharski | johnkucharski@gmail.com | 25 August 2021
#
# This file provides a conceptual model for the *optimal allocation of random samples from a stratified population with hetergenous sampling costs across the strata*.
# The intended use case is sampling trace data from exploratory scenarios with variable computing costs across the scenarios. 
#
# # TODO:
# (1) This is bascially sampling around the mean, the % below a threshold per simulation_simulation() should be calculated and each simulation_simulation() time should be included as the cost constraint (not time per year)
# (2) General testing. Compare actual results to theoretical (difference could be easily be initial sampling if its not large).
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
# N: sample size = 8 strata (i.e. scenarios) containing 100 traces of 100 years each (i.e. 10,000 years per scenario) or 80,000 years. *Note: its assumed that one performance measurement produced each year.
# strata_n: strata size = an assumed equal number of elements per strata (i.e. $\frac{N}{8} = 10,000$years) per strata.
# c: strata (computational) cost = it is assumed that even numbered strata take 0.01 seconds *per year* (or 1 seconds per trace or simulation) to run, odd numbered stata take double that time: 0.02 seconds per year (or 2 seconds per trace) to run. *Note*, the simulation runs 100 years (1 trace = 100 years) at a time.
# budget = 100 seconds. 
# n: total sample size (in yrs) = the sample size, $n \in [5,000, 10,000]$ (i.e. $n_{min} = 5,000 = \frac{100}{0.02}$ [s/(s/yr)] and $n_{max} = 10,000 = \frac{100}{0.01}$ [s/(s/yr)]).
# %%
c_o = 0
N = 8 * 100 * 100 
strata_N, c = [N/8 for i in range(len(strata))], [0.01 if i % 2 else 0.02 for i in range(len(strata))]
total_cost = c_o + sum([c[h] * strata_N[h] for h in range(len(strata))])

# %%[markdown]
# ## Simulation Simulation
# The function below samples 100 numbers at a time and takes $n_h$ seconds to run (simulating the computational cost of a simulation model).
# %%
def simulation_simulation(h: int, strata: typing.List = strata, c: typing.List[float] = c) -> typing.List:
    start = time.time()
    x = strata[h].rvs(size=100)
    time.sleep(c[h] * 100 - (time.time() - start))
    return x.tolist()

# %% [markdown]
# ## Wrappper Example
#
# %%
def target_n(data: typing.List[typing.List[float]], budget: float, init_cost: float = c_o,
             cost: typing.List[float] = c, strata_count: typing.List[int] = strata_N) -> typing.List[int]:
    n = []
    for h in range(len(data)):
        num = (budget - init_cost) * strata_count[h] * np.std(data[h]) / np.sqrt(cost[h])
        den = sum([strata_count[h] * np.std(data[h]) * np.sqrt(cost[h]) for h in range(len(data))])
        n.append(num / den)
    return n
    
def pieinsky_allocation(sim: typing.Callable, strata: typing.List, budget: float) -> typing.List[typing.List[float]]:
    x = [sim(h, strata) for h in range(len(strata))] 
    n = target_n(x, budget)
    for h in range(len(strata)):
        print(f'strata {h}: {n[h]} samples needed')
        while len(x[h]) < n[h]:
            x[h] = x[h] + sim(h, strata)
            print(f'--strata {h}: {len(x[h])} samples collected')
    return x
    
    #x: typing.List[typing.List[float]] = []
    # while tics < budget:
    #     if tics == 0:
    #         x = [sim(h, strata) for h in range(len(strata))]
    #         # for h in range(len(strata)):
    #         #     x.append(sim(h, strata))
    #     else:
    #         n = target_n(x)
    #         for h in range(len(strata)):
    #             i, n_traces = 0, np.ceil((n[h] - len(x[h])) / 100)
    #             while i < n_traces:
    #                 x[h] += sim(h, strata)
    # return x
        
# %%
budget = 100
samples = pieinsky_allocation(sim=simulation_simulation, strata=strata, budget=budget)
