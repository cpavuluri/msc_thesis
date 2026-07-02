import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path 
from scipy.optimize import curve_fit 
from scipy.special import expit 

from fit_sig import *
from channel import *

_HERE = Path(__file__).parent 
_df_mcs = pd.read_csv(_HERE/"MCS_table.csv")

SINR_MIN = float(_df_mcs["SINR"].min())
SINR_MAX = float(_df_mcs["SINR"].max())
CQI_LEVELS = 29         # no. of CQI levels are same as no. of MCS indices
_BIN_WIDTH = (SINR_MAX - SINR_MIN) / CQI_LEVELS 
# print(SINR_MIN,SINR_MAX,_BIN_WIDTH) # -15.0 32.0 1.6206896551724137 

# Simulation parameters
BLER_TARGET = 0.1
DELTA_DN = 1
DELTA_UP = BLER_TARGET / (1.0 - BLER_TARGET) * DELTA_DN 
DELTA_MIN = -10
DELTA_MAX = 10
N_SLOTS =  10000

def quantize_cqi(sinr_db:float) -> int:
    sinr_clipped = np.clip(sinr_db, SINR_MIN, SINR_MAX) 
    q_index = int(np.floor((sinr_clipped - SINR_MIN) / _BIN_WIDTH)) 
    return min(q_index, CQI_LEVELS - 1) # sinr_db == SINR_MAX -> index 28 

def dequantize_cqi(q_index: int) -> float:
    return SINR_MIN + (q_index + 0.5) * _BIN_WIDTH 

def bler(sinr_db, mcs_idx) :
    c = SIGMOID_CENTER[mcs_idx] 
    b = SIGMOID_SCALE[mcs_idx] 
    return 1.0 / (1.0 + np.exp((sinr_db - c)/ b)) 

def select_mcs(sinr_eff_db, bler_target=BLER_TARGET):
    for mcs_idx in range(N_MCS - 1, -1, -1):
        if bler(sinr_eff_db, mcs_idx) <= bler_target:
            return mcs_idx
    return 0 

def olla_update(delta,ack,delta_min=DELTA_MIN, delta_max=DELTA_MAX):
    delta = delta + DELTA_UP if ack else delta - DELTA_DN 
    return float(np.clip(delta, delta_min, delta_max))
    
def olla_effective_sinr(sinr_estimated_db, delta):
    return sinr_estimated_db + delta

# sinr_est (quantize) -> (dequantize) effective_sinr -> select_mcs

## TRUE SINR 
n_slots = N_SLOTS
sinr_bounds = [10, 15]
#sinr_trace = generate_rect(n_slots, n_jumps=2, bounds=sinr_bounds) 
sinr_trace = generate_ar(n_slots, coef=0.99, std_noise=.1, bounds=sinr_bounds)
sinr_array = np.asarray(sinr_trace, dtype=float)

# f, a = plt.subplots(1, 1, figsize=(8, 4))
# a.plot(sinr_trace)
# a.grid()
# a.set_title('SINR evolution')
# a.set_xlabel('Slot')
# a.set_ylabel('SINR [dB]')
# plt.show()

rng  = np.random.default_rng(seed=42)

delta_hist = np.zeros(n_slots)
mcs_hist = np.zeros(n_slots, dtype=int)
ack_hist = np.zeros(n_slots, dtype=int)
bler_hist = np.zeros(n_slots) 
se_hist = np.zeros(n_slots)
sinr_true_hist = np.zeros(n_slots)
sinr_reported_hist = np.zeros(n_slots)
sinr_eff_hist = np.zeros(n_slots)

delta = 0.0 

for t in range(n_slots):
    # for each TTI

    # 1. true sinr
    sinr_true = sinr_array[t]

    delta_hist[t] = delta

    # 2.quantize - dequantize sinr
    sinr_reported = dequantize_cqi(quantize_cqi(sinr_true))

    # 3. apply olla offset to reported sinr
    sinr_eff = sinr_reported + delta
    
    # 4. select MCS on effective sinr
    mcs_idx = select_mcs(sinr_eff, bler_target=BLER_TARGET)

    # 5. BLER and outcome (ack/nack) at TRUE SINR 
    b = bler(sinr_true, mcs_idx)
    #b = bler(sinr_reported, mcs_idx)
    outcome = rng.binomial(1, 1.0 - b)

    # 6. update ack/ nack ( initial ack=1)
    delta = olla_update(delta,outcome)

    # Record
    mcs_hist[t]      = mcs_idx
    ack_hist[t]      = outcome
    delta_hist[t]     = delta
    bler_hist[t]     = b
    #se_hist[t]       = SPECTRAL_EFF[mcs_idx] * outcome
    sinr_true_hist[t]     = sinr_true
    sinr_reported_hist[t] = sinr_reported
    sinr_eff_hist[t] = sinr_eff

# print("sinr_true:",sinr_true_hist)
# print("sint_reported:",sinr_reported_hist)
# print("sinr_eff:", sinr_eff_hist) 
# print("mcs_index:",mcs_hist)
# print("ack/nack:", ack_hist)
# print("delta:",delta_hist)
# #print("spectral_efficieny:", se_hist)



# plots

# SINR: true / reported / effective
f, a = plt.subplots(1, 1, figsize=(8, 4))
a.plot(sinr_true_hist, label='true')
a.plot(sinr_reported_hist, label='reported')
#a.plot(sinr_eff_hist, label='effective')
a.grid()
a.legend()
a.set_title('SINR')
a.set_xlabel('Slot')
a.set_ylabel('SINR [dB]')
plt.savefig("SINR.png")
plt.show()

# # MCS index
# f, a = plt.subplots(1, 1, figsize=(8, 4))
# a.step(range(n_slots), mcs_hist, where='post')
# a.grid()
# a.set_title('Selected MCS')
# a.set_xlabel('Slot')
# a.set_ylabel('MCS index')
# plt.show()

# # BLER at true SINR
# f, a = plt.subplots(1, 1, figsize=(8, 4))
# a.plot(bler_hist)
# a.axhline(BLER_TARGET, color='r', linestyle='--', label='target')
# a.grid()
# a.legend()
# a.set_title('BLER')
# a.set_xlabel('Slot')
# a.set_ylabel('BLER')
# plt.show()


### revisit : average bler aver a window

W = 100  # window length in slots

nack = 1 - ack_hist
bler_avg = np.convolve(nack, np.ones(W) / W, mode='valid')

f, a = plt.subplots(1, 1, figsize=(8, 4))
a.plot(np.arange(W - 1, n_slots), bler_avg, label=f'avg BLER (window={W})')
a.axhline(BLER_TARGET, color='r', linestyle='--', label='target')
a.grid()
a.legend()
a.set_title('Sliding-window average BLER')
a.set_xlabel('Slot')
a.set_ylabel('BLER')
plt.savefig("sliding_window_bler.png")
plt.show()






