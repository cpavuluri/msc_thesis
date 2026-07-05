
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from channel import *

_HERE = Path(__file__).parent 
_df_mcs = pd.read_csv(_HERE/"MCS_table.csv")

# Parameters 
bler_target_list = [0.1, 0.00001] # 1e-6  
b = len(bler_target_list)    # target BLER 0.000001
n_slots = 3         # total TTI
#sinr_true = [13.5]*n_slots
sinr_bounds = [24,26]
# get sinr via channel
sinr_true =generate_ar(n_slots, coef=0.99, std_noise=.1, bounds=sinr_bounds)
# round of the sinr values to one decimal
sinr_true =  np.round(sinr_true,1)

mcs_hist=np.full((b, n_slots), np.nan)
tti_hist=range(0,n_slots)

for b in range(len(bler_target_list)):
    bler_target = bler_target_list[b]
    print("bler_target:", bler_target)
    
    for t in range(n_slots):

        # 1. choose all the MCS values that are equal to the SINR
        _df_sinr_verified = _df_mcs[_df_mcs["SINR"].round(1)==sinr_true[t]]

        # 2. compare with target_bler, keep only the values with bler <= target_bler
        _df_bler_verified = _df_sinr_verified[_df_sinr_verified["BLER"] < bler_target]

        if _df_bler_verified.empty:
            # if there is no vlaue below the target BLER default to MCS = 0
            mcs_hist[b][t] = 0
            #print(f"tti:{tti_hist[t]}, sinr:{sinr_true[t]}, mcs:{mcs_hist[b][t]}")
        else:
            # 3. Pick the value with highest spectral efficency
            _df_se_verified = _df_bler_verified.loc[[_df_bler_verified["SE"].idxmax()]]
            mcs_hist[b][t] = _df_se_verified["MCS_Index"].iloc[0]
            #print(f"tti:{tti_hist[t]}, sinr:{sinr_true[t]}, mcs:{mcs_hist[b][t]}")
            #print("_df_bler_verified::",_df_bler_verified)
            #print("_df_se_verified::",_df_se_verified)

            #print(f'\nTTI:{t}, sinr:{target_row["SINR"]} , SE:{target_row["SE"]}, MO:{target_row["MO"]}, BLER:{target_row["BLER"]}, selected_mcs:{target_row["MCS_Index"]}')
        print(f"sinr:{sinr_true[t]}, mcs:{mcs_hist[b][t]}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18,6))
# sinr_true plot
ax1.plot(tti_hist, sinr_true)
ax1.grid()
ax1.set_title("sinr")

# selected mcs indices for different target_bler: plot
for b_idx, b in enumerate(bler_target_list):
    ax2.plot(tti_hist, mcs_hist[b_idx], label=f"bler_target={b}")

ax2.grid()
ax2.set_title("selected mcs")
ax2.set_xlabel("TTI")
ax2.legend()

plt.tight_layout()
plt.savefig("ground_truth.png")
plt.show()








