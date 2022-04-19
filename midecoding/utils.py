# From https://tttapa.github.io/Pages/Mathematics/Systems-and-Control-Theory/Digital-filters/Simple%20Moving%20Average/Simple-Moving-Average.html

from scipy.optimize import newton
from scipy.signal import freqz, dimpulse, dstep
from math import sin, cos, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt

# Function for calculating the cut-off frequency of a moving average filter
def get_sma_cutoff(N, **kwargs):
    func = lambda w: sin(N*w/2) - N/sqrt(2) * sin(w/2)  # |H(e^j?)| = v2/2
    deriv = lambda w: cos(N*w/2) * N/2 - N/sqrt(2) * cos(w/2) / 2  # dfunc/dx
    omega_0 = pi/N  # Starting condition: halfway the first period of sin(N?/2)
    return newton(func, omega_0, deriv, **kwargs)

    
if__name__=="__main__":
    # Simple moving average design parameters
    f_s = 250
    N = 55

    # Find the cut-off frequency of the SMA
    w_c = get_sma_cutoff(N)
    f_c = w_c * f_s / (2 * pi)

    print(f"Cut-Off frequency for Sampling Rate {f_s}Hz and {N} samples({N/f_s}s): {f_c}Hz")