import numpy as np
from multiprocessing import Pool
import random
import os 
from numba import jit
import shutil
import uuid

@jit(nopython=True, cache=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


# Gillespie version:
#@jit(nopython=True, cache=True)
def gillespie_simulation_T(beta0, dbeta, sigma, gamma, delta, m, N, T_max, threshold):
    # Level of cross immunity:
    sigma_c = 0.55
    # How often to save data:
    dt_measure = 1 # Once per day
    # Initialize state variables
    ts_out = []
    Ss_out, I1s_out, I2s_out, I12s_out, I21s_out, R1s_out, R2s_out, Rs_out = [], [], [], [], [], [], [], []
    
    # Fraction initially infected:
    I1_frac_initial = 0.01
    S, I1, I2, I12, I21, R1, R2, R = N - int(N * I1_frac_initial), int(N * I1_frac_initial), 0, 0, 0, 0, 0, 0
    t = 0
    ts_out.append(t)
    Ss_out.append(S)
    I1s_out.append(I1)
    I2s_out.append(I2)
    I12s_out.append(I12)
    I21s_out.append(I21)
    R1s_out.append(R1)
    R2s_out.append(R2)
    Rs_out.append(R)
    
    t_emergs = [] # List to hold emergence times
    emerg_flag = False
    t_emerg = 0
    
    rates = np.zeros(16, dtype=np.float64)
    ratenumbers = np.arange(0,16,1,dtype=np.int32)
    while t < T_max:
        if t > 500:
            m_eff = m
        else:
            m_eff = 0
        # Time-varying beta:
        beta = beta0 + dbeta * np.cos(2 * np.pi * t / 365) 
        # Full equations (17)-(24) of manuscript
        # Rates:
        rates[0] = beta * S * (I1+I12) / N # S to I1
        rates[1] = beta * (1 - sigma) * R1 * (I1+I12) / N # R1 to I1
        rates[2] = beta * (1 - sigma_c) * R2 * (I1+I12) / N # R2 to I12
        rates[3] = beta * (1 - sigma) * R * (I1+I12) / N # R to I12
        rates[4] = beta * S * (I2+I21) / N  # S to I2
        rates[5] = beta * (1 - sigma) * R2 * (I2+I21) / N # R2 to I2
        rates[6] = beta * (1 - sigma_c) * R1 * (I2+I21) / N # R1 to I21
        rates[7] = beta * (1 - sigma) * R * (I2+I21) / N # R to I21
        rates[8] = gamma * I1 # I1 to R1 (recovery)
        rates[9] = gamma * I2 # I2 to R2 (recovery)
        rates[10] = gamma * I12 # I12 to R (recovery)
        rates[11] = gamma * I21 # I21 to R (recovery)
        rates[12] = delta * R1 # R1 to S (waning) 
        rates[13] = delta * R2 # R2 to S (waning)
        rates[14] = delta * R # R to S (waning)
        rates[15] = m_eff * (I1+I12) # I1 to I2 mutation
        
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
        
        # Rules (update states):
        if event == 0: # S to I1
            S -= 1
            I1 += 1
        elif event == 1: # R1 to I1
            R1 -= 1
            I1 += 1
        elif event == 2: # R2 to I12
            R2 -= 1
            I12 += 1
        elif event == 3: # R to I12
            R -= 1
            I12 += 1
        elif event == 4: # S to I2
            S -= 1
            I2 += 1
        elif event == 5: # R2 to I2
            R2 -= 1
            I2 += 1
        elif event == 6: # R1 to I21
            R1 -= 1
            I21 += 1
        elif event == 7: # R to I21
            R -= 1
            I21 += 1
        elif event == 8: # I1 to R1 (recovery)
            # Do not allow epidemic to die out completely:
            if I1+I12 > 1:
                I1 -= 1
                R1 += 1
        elif event == 9: # I2 to R2 (recovery)
            I2 -= 1
            R1 += 1
        elif event == 10: # I12 to R (recovery)
            # Do not allow epidemic to die out completely:
            if I1+I12 > 1:
                I12 -= 1
                R += 1
        elif event == 11: # I21 to R (recovery)
            I21 -= 1
            R += 1
        elif event == 12: # R1 to S (waning) 
            R1 -= 1
            S += 1
        elif event == 13: # R2 to S (waning)
            R2 -= 1
            S += 1
        elif event == 14: # R to S (waning)
            R -= 1
            S += 1
        elif event == 15: # I1 to I2 mutation
            # Do not allow epidemic to die out completely:
            if I1+I12 > 1:
                I1 -= 1
                I2 += 1

        if S < 0:
            print("Error! S went negative at:")
            print(f"t={t}, S={S}, I1={I1}, I2={I2}, I12={I12}, I21={I21}, R1={R1}, R2={R2}, R={R}")
        
        if t-ts_out[-1] >= dt_measure:
            if int(t) % 100 == 0:
                print(f"t = {t}")
            ts_out.append(t)
            Ss_out.append(S)
            I1s_out.append(I1)
            I2s_out.append(I2)
            I12s_out.append(I12)
            I21s_out.append(I21)
            R1s_out.append(R1)
            R2s_out.append(R2)
            Rs_out.append(R)

        # Check if variant has appeared, sub-threshold:
        if I2+I21 > 0 and not emerg_flag:
            emerg_flag = True
            t_emerg = t
        # Check if variant has dies out:
        if emerg_flag and I2+I21 == 0:
            emerg_flag = False
            t_emerg = 0
        # Check if threshold for I2 is reached
        if I2+I21 >= threshold:
            t_emergs.append(t_emerg)
            t_emerg = 0
            emerg_flag = False
            print(f"{t}, adding {I2 + I21} to S")
            # Reset number of mutants:
            S += I2 + I21
            I1 += I12
            I2 = 0
            I21 = 0
            I12 = 0
    
    return t_emergs, ts_out, Ss_out, I1s_out, I2s_out, I12s_out, I21s_out, R1s_out, R2s_out, Rs_out  # We must record the emergence times, and not just the number of emergences

# Parameters for the heatmap
N = 40000
T_max = 4000
threshold = 100
gamma = 0.2
m = 1e-4
dbeta = 0.2  # Amplitude of seasonal forcing (amplitude)
delta = 1/365  # Immune waning rate

beta = 2*gamma
sigma = 0.8

t_emergs, ts_out, Ss_out, I1s_out, I2s_out, I12s_out, I21s_out, R1s_out, R2s_out, Rs_out = gillespie_simulation_T(beta, dbeta, sigma, gamma, delta, m, N, T_max, threshold)

# Directory to save outputs
outdir = "seasonal/data/"
os.makedirs(outdir, exist_ok=True)

# Copy the script for posterity
script_path = __file__ if "__file__" in locals() else "gillespie_seasonal_single.py"  # Replace with actual script name if not running as a file
shutil.copy(script_path, os.path.join(outdir, "script.py"))

# Generate a unique file name with a random UUID
unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of the UUID for brevity
output_file = os.path.join(outdir, f"simulation_data_{unique_id}.csv")

# Data to save
data = {
    "t_emerg": t_emergs,
    "t": ts_out,
    "S": Ss_out,
    "I1": I1s_out,
    "I2": I2s_out,
    "I12": I12s_out,
    "I21": I21s_out,
    "R1": R1s_out,
    "R2": R2s_out,
    "R": Rs_out,
}

# Write data to file in a sequential, list-per-line format
with open(output_file, "w") as f:
    # Write a header
    f.write("# Simulation Output\n")
    f.write("# Each line represents a list of values for a specific variable.\n")
    f.write("# Order:\n")
    f.write("# t_emerg,t,S,I1,I2,I12,I21,R1,R2,R\n")
    f.write("\n")

    # Write each list on its own line with a label
    for key, values in data.items():
        print(f"Writing {key}.")
        #f.write(f"{key}:\n")
        f.write(",".join(map(str, values)) + "\n")

print(f"Data successfully saved in {output_file}")
