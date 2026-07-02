import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path 
from scipy.optimize import curve_fit 
from scipy.special import expit 


_HERE = Path(__file__).parent 
_df_mcs = pd.read_csv(_HERE/"MCS_table.csv")

def _fit_sigmoid(mcs_idx: int):
    sub = _df_mcs[_df_mcs["MCS_Index"] == mcs_idx]
    x = sub["SINR"].values
    y = sub["BLER"].values 
    c0 = float(x.mean())

    popt, _ = curve_fit(
        lambda s, c, b: expit(-(s-c) / b),
        x, y, p0=[c0,0.5], maxfev=10_000,
    )  
    return float(popt[0]), abs(float(popt[1]))

_fits = [_fit_sigmoid(i) for i in range(29)] 
_sigmoid_centers = np.array([c for c, _ in _fits])
_sigmoid_scales = np.array([s for _, s in _fits]) 

# One representative row per MCS (SE,MO constant for each MCS)
_mcs_rep = _df_mcs.groupby("MCS_Index", sort=True).first().reset_index()
_MO = _mcs_rep["MO"].values.astype(int) 
_SE = _mcs_rep["SE"].values 
# bps/Hz = code_rate x MO

# Assemble MCS_PARAMS - shape(29,4)
# Columns: [MP, SE, sigmoid_center_db, sigmoid_scale_db] 
MCS_PARAMS = np.column_stack([_MO, _SE, _sigmoid_centers, _sigmoid_scales]) 
N_MCS = len(MCS_PARAMS)         # 29 entries (MCS 0-28) 
MOD_ORDER = MCS_PARAMS[:,0].astype(int) # modulation oreder (bits/symbol) 
SPECTRAL_EFF = MCS_PARAMS[:,1]          # spectral efiiciency (bps/Hz) 
SIGMOID_CENTER = MCS_PARAMS[:,2] 
SIGMOID_SCALE = MCS_PARAMS[:,3]

# Save the values into a csv file
_out_df = pd.DataFrame({
    "MCS_Index": np.arange(N_MCS),
    "MOD_ORDER": MOD_ORDER,
    "SPECTRAL_EFF": SPECTRAL_EFF,
    "SIGMOID_CENTER": SIGMOID_CENTER, 
    "SIGMOID_SCALE": SIGMOID_SCALE, 
})
_out_df.to_csv(_HERE/"MCS_PARAMS.csv", index=False) 
