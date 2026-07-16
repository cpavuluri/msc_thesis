
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from pathlib import Path 
from channel import * 

# get_MCS()
_HERE = Path(__file__).parent 
_df_mcs = pd.read_csv(_HERE/"MCS_table.csv") 

# Parameters
bler_target = 0.1 
n_slots = 1000
sinr_bounds = [-5,20]
np.random.seed(42) 

delta_offset = 0.0
delta_down = 0.1
delta_up = (1 - bler_target) / bler_target * delta_down  # 0.9

# Channel
sinr_true = generate_ar(n_slots, coef=0.99, std_noise = 0.1, bounds = sinr_bounds, seed=42) 
# round the sinr values to one decimal
sinr_true = np.round(sinr_true,1) 

# def initialize_parameters()
olla_cqi_hist = np.full(n_slots, np.nan) 
olla_sinr_est_hist = np.full(n_slots, np.nan) 
olla_sinr_corrected_hist = np.full(n_slots, np.nan)
olla_ack_hist = np.full(n_slots, np.nan)
olla_mcs_hist = np.full(n_slots, np.nan) 
olla_tti_hist = range(0, n_slots) 
olla_bler_estimated_hist = np.full(n_slots, np.nan)
olla_bler_hist = np.full(n_slots, np.nan) 
olla_se_value_hist = np.full(n_slots, np.nan) 
olla_delta_hist = np.full(n_slots, np.nan)

def sinr_thresh_table():
    # cqi_table holds the sinr thresholds for that index
    cqi_table = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]

    return cqi_table


def quantize_cqi(sinr_true):
    # rename to quantize_sinr_to_cqi /  estimate_cqi_from_sinr(sinr)
    # cqi is an integer value, sinr is mapped to cqi 
    # -5 dB -> cqi 0, 20 dB -> cqi 25
    cqi = 0
    cqi_table = sinr_thresh_table()
    for cqi_idx, sinr_thresh in enumerate(cqi_table):
        
        if sinr_true >= sinr_thresh:
            cqi = cqi_idx

    return cqi

def dequantize_cqi(cqi):
    # rename to dequantize_cqi_to_sinr_est / estimate_sinr_from_cqi(cqi)
    cqi_table = sinr_thresh_table()

    for cqi_idx, sinr_thresh in enumerate(cqi_table):
        if cqi == cqi_idx:
            sinr_est = sinr_thresh

    return sinr_est 

def calculate_bler(sinr_true, current_mcs, _df_mcs):
    # this is calculated from the UE perspective
    # for the current MCS, find the estimated bler vaule at the sinr_true
    _df_mcs_verified = _df_mcs[(_df_mcs["SINR"]== sinr_true) & (_df_mcs["MCS_Index"]==current_mcs)]
    if _df_mcs_verified.empty:
        return 1.0
    estimated_bler = _df_mcs_verified["BLER"].iloc[0] 
    return estimated_bler

def update_delta_offset(ack,delta_offset):
    if ack:
        delta_offset -= delta_down
    else:
        delta_offset += delta_up
    return round(delta_offset,1)

def ack_nack_feedback(estimated_bler, actual_bler):
    rnd_number = np.random.uniform()
    if rnd_number < estimated_bler:
        ack = 0
    else:
        ack = 1
    return ack


def determine_mcs_from_sinr_corrected(sinr_corrected, _df_mcs):
    olla_mcs = 0
    olla_bler = 1
    # choose all MCS values that are equal to sinr
    _df_sinr_verified = _df_mcs[_df_mcs["SINR"]== sinr_corrected]
    # keep only the values with bler <= target_bler
    _df_bler_verified = _df_sinr_verified[_df_sinr_verified["BLER"] < bler_target]
    # check for spectral efficiency
    if _df_bler_verified.empty:
        # if there is no value below the target_bler defaault to MCS=0
        olla_mcs = 0
        olla_bler = 1
        se_value = 0.0
    else:
        # pick the value with highest spectral efficiency
        _df_se_verified = _df_bler_verified.loc[[_df_bler_verified["SE"].idxmax()]]
        olla_mcs = _df_se_verified["MCS_Index"].iloc[0] 
        olla_bler = _df_se_verified["BLER"].iloc[0]
        se_value = (1 - olla_bler) * (_df_se_verified["SE"].iloc[0])
    return olla_mcs, olla_bler , se_value 


print(f"bler target = {bler_target}")
# for the first iteration, you need inital delta_offset = 0.0, initial mcs =? initial ack = ?
for t in range(n_slots):

    # 1. quantize sinr_true to get cqi 
    olla_cqi_hist[t] = quantize_cqi(sinr_true[t])

    # 2. dequantize to sinr_est 
    olla_sinr_est_hist[t] = dequantize_cqi(olla_cqi_hist[t])

    # 3. Subtract the offset
    sinr_corrected_value = olla_sinr_est_hist[t] + delta_offset
    olla_sinr_corrected_hist[t] = np.round(sinr_corrected_value, 1)
    olla_delta_hist[t] = delta_offset

    # 4. Use the corrected sinr to find the olla MCS , bler and effective SE
    olla_mcs_hist[t], olla_bler_hist[t], olla_se_value_hist[t] = determine_mcs_from_sinr_corrected(olla_sinr_corrected_hist[t], _df_mcs)
    #print(f"sinr:{olla_sinr_corrected_hist[t]}, mcs:{olla_mcs_hist[t]}")

    # 5. Use the current MCS to find the calculated bler for the next tti
    olla_bler_estimated_hist[t] = calculate_bler(sinr_true[t], olla_mcs_hist[t], _df_mcs)

    # 6. Use the estimated_bler to get ack/nack
    olla_ack_hist[t] = ack_nack_feedback(olla_bler_estimated_hist[t], olla_bler_hist[t])

    # 7. Use the ack feedback to update the delta_offset    
    delta_offset = update_delta_offset(olla_ack_hist[t], delta_offset)


# print(f"\nsinr_true:{sinr_true}, \nolla_sinr_est_hist: {olla_sinr_est_hist}, \nolla_sinr_corrected_hist: {olla_sinr_corrected_hist} ")

# print(f"\nolla_bler_estimated_hist: {olla_bler_estimated_hist} ,\nolla_bler_hist: {olla_bler_hist}")

# print(f"\n olla_ack_hist: {olla_ack_hist} , \n olla_delta_hist: {olla_delta_hist}")

# print(f"\n olla_cqi_hist: {olla_cqi_hist} ,\n olla_mcs_hist: {olla_mcs_hist} , \nolla_se_value_hist: {olla_se_value_hist}")

fig, ax = plt.subplots() 

ax.plot(olla_tti_hist, sinr_true, label="SINR true")
ax.plot(olla_tti_hist, olla_sinr_est_hist, label="SINR eestimated")
ax.plot(olla_tti_hist, olla_sinr_corrected_hist, label="SINR corrected")

ax.grid()
ax.set_title("SINR true, estimated, corrected")
ax.set_xlabel("TTI")
ax.set_ylabel("SINR [dB]")
ax.legend()

plt.tight_layout()
plt.savefig("olla_sinr.png", bbox_inches="tight")
plt.show()

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)

ax1.plot(olla_tti_hist,olla_delta_hist)
ax1.grid()
ax1.set_title("offset delta values")

# box plot for mcs indices
ax2.boxplot(olla_mcs_hist)
ax2.grid()
ax2.set_title("box plot for selected MCS")

# ecdf SE
ax3.ecdf(olla_se_value_hist)
ax3.grid()
ax3.set_title("ecdf SE")

# ecdf BLER
ax4.ecdf(olla_bler_hist, label=f"bler_target=0.1")
ax4.grid()
ax4.set_title("ecdf bler")
ax4.legend()

plt.tight_layout()
plt.savefig("olla.png", bbox_inches="tight")
plt.show()






