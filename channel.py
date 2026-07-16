
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(42)

def rescale(y, bounds, return_coeffs=False):
    M, m = max(y), min(y)
    if M != m:
        a = (bounds[1] - bounds[0]) / (M - m)
        b = bounds[0] - m * (bounds[1] - bounds[0]) / (M - m)
    else:
        a, b = 1, 0
    y = a * y + b
    if return_coeffs:
        return y, a, b
    else:
        return y


def generate_rect(n_samples, n_jumps, bounds):
    f = np.ones(n_samples)
    for t in range(n_jumps+1):
        f[t*2*n_samples//(n_jumps+1):(t*2+1)*n_samples//(n_jumps+1)] = 0
    f = rescale(f, bounds)
    return f


def generate_ar(n_samples, coef, std_noise, bounds, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(n_samples)
    for t in range(1, n_samples):
        #x[t] = coef * x[t-1] + np.random.randn() * std_noise
        x[t] = coef * x[t-1] + rng.standard_normal() * std_noise
    x = rescale(x, bounds)
    return x

if __name__ == "__main__":
    sinr_bounds = [10, 15]
    n_slots = 1000
    #sinr_true = generate_rect(n_slots, n_jumps=2, bounds=sinr_bounds)
    sinr_true = generate_ar(n_slots, coef=0.99, std_noise=.1, bounds=sinr_bounds)
    f, a = plt.subplots(1, 1, figsize=(8, 4))
    a.plot(sinr_true)
    a.grid()
    a.set_title('SINR evolution')
    a.set_xlabel('Slot')
    a.set_ylabel('SINR [dB]')
    plt.show()