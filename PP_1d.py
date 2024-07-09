"""
Created on Tue Jul 09 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Generates 1-dimensional phase plates.

Methods:
    gs:
        use Gerchberg Saxton algorithm to iteratively find ideal phase plate phases
"""

import numpy as np
import matplotlib.pyplot as plt
from PP_Tools import ideal_beam_shape, modulation_beam, round_phase

def gs(n: int, amp: float, mod_amp: float, mod_freq: float, std: float,
       max_iter: int = 1000, plot: bool = False) -> np.ndarray:
    '''
    Gerchberg Saxton algorithm:
    Approximate plate phases iteratively

    Input intensity + phase -> FFT -> far field intensity + phase (discard intensity and use ideal)
    -> iFFT -> near field intensity + phase (discard intensity and use ideal) -> repeat

    Args:
        n: number of phase elements
        amp: amplitude of laser in J
        mod_amp: modulation amplitude in J
        mod_freq: modulation frequency in Hz
        std: standard deviation of super Gaussian beam
        max_iter: maximum number of iterations
        plot: whether to plot the input/output/ideal electric fields

    Returns:
        array of phases
    '''
    x = np.linspace(-std, std, n)

    # ideal beam
    ideal_beam = ideal_beam_shape(x, amp, std)

    # initial input beam
    theta_in = (np.pi/2)*np.random.randint(-2, 3, size = n) # random phases from -pi to pi
    original_beam_electric = np.abs(modulation_beam(x, amp, std, mod_amp, mod_freq, theta_in))

    for _ in range(max_iter):
        # initial intensity * phase from iFFT
        input_beam_electric = np.square(original_beam_electric) * np.exp(1j*theta_in)

        # FFT of beam
        beam_ft = np.fft.fft(input_beam_electric)
        beam_ft = beam_ft/max(beam_ft)*max(ideal_beam)
        theta_out = np.angle(beam_ft)  # far field phase

        # desired focal spot intensity * phase from FFT
        new_beam_ft = np.square(ideal_beam) * np.exp(1j*theta_out)

        # inverse FFT
        new_beam_electric = np.fft.ifft(new_beam_ft)
        theta_in = np.angle(new_beam_electric) # near field phase

    theta_in = round_phase(theta_in)
    np.savetxt("phase_plate.txt", X = theta_in,
               header = "Phase values [rad]")

    if plot:
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(x, original_beam_electric, label = 'Input beam')
        ax1.plot(x, ideal_beam, label = 'Ideal beam')
        ax2.plot(x, np.abs(beam_ft), label = 'Output beam')
        ax2.plot(x, ideal_beam, label = 'Ideal beam')
        ax1.legend()
        ax2.legend()
        plt.show()

    return theta_in

if __name__ == "__main__":
    # phase elements
    PHASE_ELEMENTS = 1000

    # laser beam parameters
    AMPLITUDE = 5 # in J
    STD_DEV = 3 # in micron (FWHM/2.35482 for Gaussian)
    MODULATION_AMPLITUDE = 0.01 # in J
    MODULATION_FREQUENCY = 10 # in micron^-1

    # Gerchberg Saxton algorithm
    gs(n = PHASE_ELEMENTS, amp = AMPLITUDE, std = STD_DEV, mod_amp = MODULATION_AMPLITUDE,
            mod_freq = MODULATION_FREQUENCY, max_iter = int(1e5), plot = True)
