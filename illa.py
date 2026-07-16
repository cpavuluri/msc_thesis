import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from pathlib import Path 
from channel import *

_HERE = Path(__file__).parent 
_df_mcs = pd.read_csv(_HERE/"MCS_table.csv") 

# Parameters
bler_target = 0.1 
n_slots = 1000
sinr_bounds = [-5,20] 
np.random.seed(42)

# channel
sinr_true = generate_ar(n_slots, coef=0.99, std_noise = 0.1, bounds = sinr_bounds, seed=42) 
# round of the sinr values to one decimal
sinr_true =  np.round(sinr_true,1)

cqi_hist = np.full(n_slots, np.nan)
sinr_eff_hist = np.full(n_slots, np.nan)
mcs_hist = np.full(n_slots, np.nan)
tti_hist = range(0,n_slots)
bler_hist = np.full( n_slots, np.nan)
se_value_hist = np.full( n_slots, np.nan)

def quantize_cqi(sinr_true):
    # cqi is an integer value, sinr is mapped to cqi
    # -5 dB -> cqi 0, 20dB -> cqi 25
    sinr_min = -5
    sinr_max = 20
    step_size = 1
    cqi = np.clip(round(sinr_true-sinr_min)/step_size, 0, 28)
    return cqi

def dequantize_cqi(cqi):
    sinr_min = -5
    sinr_max = 20
    step_size = 1
    # the decimal value of sinr is lost due to quantization
    sinr_eff = sinr_min + cqi * step_size 
    return sinr_eff


def calculate_bler(ack,nack,tti):
    pass 

for t in range(n_slots):

    # 1. qunatize sinr_true and get cqi
    cqi_hist[t] = quantize_cqi(sinr_true[t])
    # 2. dequantize to sinr_eff
    sinr_eff_hist[t] = dequantize_cqi(cqi_hist[t])
    # 3. choose all MCS values that are equal to sinr_eff
    _df_sinr_verified = _df_mcs[_df_mcs["SINR"]==sinr_eff_hist[t]]
    # 4. keep only the values with bler <= target_bler
    _df_bler_verified = _df_sinr_verified[_df_sinr_verified["BLER"] < bler_target]
    # 5. check for spectral efficiency
    if _df_bler_verified.empty:
        # if there is no value below the target BLER default to MCS =0
        mcs_hist[t] = 0 
        bler_hist[t] = 1
        se_value_hist[t] = 0.0
    else:
        # pick the value with highest spectral effiicency
        _df_se_verified = _df_bler_verified.loc[[_df_bler_verified["SE"].idxmax()]]
        mcs_hist[t] = _df_se_verified["MCS_Index"].iloc[0]
        bler_hist[t] = _df_se_verified["BLER"].iloc[0]
        #bler_hist[t] = 0
        # 6. calculate the SE wrt bler
        se_value_hist[t] = (1-bler_hist[t])*(_df_se_verified["SE"].iloc[0])

print(f"bler target ={bler_target}")

#print(f"\n sinr_true:{sinr_true},\n cqi:{cqi_hist},\n sinr_eff:{sinr_eff_hist},\n mcs:{mcs_hist},\n bler:{bler_hist},\n se_value: {se_value_hist}")

fig, (ax1, ax2) = plt.subplots(2,1)

# sinr_true plot
ax1.plot(tti_hist, sinr_true)
ax1.grid()
ax1.set_title("sinr true")

# sinr_eff plot
ax2.plot(tti_hist, sinr_eff_hist)
ax2.grid()
ax2.set_title("sinr effective")

plt.tight_layout()
plt.savefig("illa_sinr.png", bbox_inches="tight")
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

# cqi indices
ax1.plot(tti_hist, cqi_hist)
ax1.grid()
ax1.set_title("cqi indices")

# box plot for selected mcs indices
ax2.boxplot(mcs_hist)
ax2.grid()
ax2.set_title("box plot selected mcs")

# ecdf SE
ax3.ecdf(se_value_hist)
ax3.grid()
ax3.set_title("ecdf SE")

# ecdf bler
ax4.ecdf(bler_hist, label=f"bler_target=0.1")
ax4.grid()
ax4.set_title("ecdf bler")
ax4.legend()

plt.tight_layout()
plt.savefig("illa.png", bbox_inches="tight")
plt.show()



    
