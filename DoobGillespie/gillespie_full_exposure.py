import numpy as np
from multiprocessing import Pool
import random
import os 
from numba import jit
import shutil

@jit(nopython=True, cache=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


# Gillespie version:
@jit(nopython=True, cache=True)
def gillespie_simulation_T(beta, sigma, gamma, m, N, T_max, threshold):
    # Initialize state variables
    # Should initialize at steady-state values
    if sigma < 1:
        R1_frac_initial = min(gamma/((1-sigma) * beta), 1)
    else:
        R1_frac_initial = 1
    R1, I1, I2, R2, I12 = int(N * R1_frac_initial), N - int(N * R1_frac_initial), 0, 0, 0
    t = 0
    t_emergs = [] # List to hold emergence times
    
    rates = np.zeros(8, dtype=np.float64)
    ratenames = ['R1_to_I1', 'R1_to_I2', 'R2_to_I2', 'R2_to_I12', 'I1_to_R1', 'I2_to_R2', 'I12_to_R2', 'I1_to_I2_mutation']
    ratenumbers = np.arange(0,8,1,dtype=np.int32)
    
    while t < T_max:
        if t > 100:
            m_eff = m
        else:
            m_eff = 0
        # Rate overview:
        # 0: R1_to_I1
        # 1: R1_to_I2
        # 2: R2_to_I2
        # 3: R2_to_I12
        # 4: I1_to_R1
        # 5: I2_to_R2
        # 6: I12_to_R2
        # 7: I1_to_I2_mutation
        rates[0] = beta * (1 - sigma) * R1 * (I1+I12) / N
        rates[1] = beta * R1 * I2 / N
        rates[2] = beta * (1 - sigma) * R2 * I2 / N
        rates[3] =  beta * (1 - sigma) * R2 * (I1+I12) / N
        rates[4] = gamma * I1
        rates[5] = gamma * I2
        rates[6] = gamma * I12
        rates[7] = m_eff * (I1+I12)

        total_rate = np.sum(rates)
        if total_rate == 0:
            break

        # Time until next event
        dt = -np.log(random.uniform(0, 1)) / total_rate
        t += dt

        # Select event
        p = rates
        p /= np.sum(p)
        event = rand_choice_nb(ratenumbers, p)

        # Update states
        if event == 0:
            R1 -= 1
            I1 += 1
        elif event == 1:
            R1 -= 1
            I2 += 1
        elif event == 2:
            R2 -= 1
            I2 += 1
        elif event == 3:
            R2 -= 1
            I12 += 1
        elif event == 4:
            I1 -= 1
            R1 += 1
        elif event == 5:
            I2 -= 1
            R2 += 1
        elif event == 6:
            I12 -= 1
            R2 += 1
        elif event == 7:
            I1 -= 1
            I2 += 1

        # Check if threshold for I2 is reached
        if I2 >= threshold:
            t_emergs.append(t)
            I2 = 0 # Reset number of mutants
            I1 += I12
            I12 = 0
    
    return len(t_emergs)


def single_realization(_):
    return gillespie_simulation_T(beta, sigma, gamma, m, N, T_max, threshold)

def multiple_realizations_mean_T_parallel(beta, sigma, gamma, m, N, T_max, threshold, num_realizations):
    with Pool() as pool:
        n_emergs = pool.map(single_realization, range(num_realizations))
        # T_values should then be a list of lists.
        # Each of the inner lists contain the emergence times recorded in a single simulation
        # Flatten T_values:
    n_emergs_mean = np.mean(n_emergs)
    print(f"{n_emergs_mean} emergences at beta={beta}, sigma={sigma}.")
    return n_emergs_mean

# Parameters for the heatmap
N = 5000
T_max = 500
threshold = 100
sigmas = np.arange(0, 1.01, 0.02)
#sigmas = np.array([0.5])
betas = np.arange(0.5, 3.0, 0.1)
#betas = np.arange(1.0, 2.0, 0.1)
#betas = np.arange(3.0, 6.01, 0.2)
#betas = np.arange(0.5, 1.2, 0.1)
gamma = 0.5
m = 1e-4

num_realizations = 100  # Number of realizations for each parameter combination

n_parvalues = len(betas) * len(sigmas)
n_run = 0

outdir = "full_exposure/data/"
shutil.copy(__file__, f"{outdir}script.py")

# Run simulations and fill heatmap array
for i, beta in enumerate(betas):
    for j, sigma in enumerate(sigmas):
        beta = np.round(beta,4)
        sigma = np.round(sigma,4)
        n_run += 1
        print(f"Running {n_run} of {n_parvalues}")
        outfilepath = f"{outdir}result_beta_{beta}_sigma_{sigma}.txt"
        fileexists = os.path.exists(outfilepath)
        if not fileexists:
            T = multiple_realizations_mean_T_parallel(beta, sigma, gamma, m, N, T_max, threshold, num_realizations)
            f = open(outfilepath, "a")
            f.write(f"{T}")
            f.close()
        else:
            print(f"File {outfilepath} already exists.")