"""
Created on Tue Jul 09 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Generates 2-dimensional phase plates.

Methods:
    gs:
        use Gerchberg Saxton algorithm to iteratively find ideal phase plate phases
"""

import numpy as np
import matplotlib.pyplot as plt
from PP_Tools import ideal_beam_shape, modulation_beam, round_phase

def gs_2d(n: int, amp: float, mod_amp: float, mod_freq: float, std: float,
       max_iter: int = 1000, plot: bool = False) -> np.ndarray:
    '''
    Gerchberg Saxton algorithm:
    Approximate 2-d plate phases iteratively

    Input intensity + phase -> FFT -> far field intensity + phase (discard intensity and use ideal)
    -> iFFT -> near field intensity + phase (discard intensity and use ideal) -> repeat

    Args:
        n: number of phase elements in square side
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
    xy = np.zeros((len(x), len(x), 2))
    for i, row in enumerate(xy):
        for j, _ in enumerate(row):
            xy[i, j] = [x[i], x[j]]

    # ideal beam
    ideal_beam = ideal_beam_shape(xy, amp, std)

    # initial input beam
    theta_in = (np.pi/2)*np.random.randint(-2, 3, size = np.shape(xy)[:-1]) # random phases
    original_beam_electric = np.abs(modulation_beam(xy, amp, std, mod_amp, mod_freq, theta_in))

    for _ in range(max_iter):
        # initial intensity * phase from iFFT
        input_beam_electric = np.square(original_beam_electric) * np.exp(1j*theta_in)

        # FFT of beam
        beam_ft = np.fft.fft(input_beam_electric)
        beam_ft = beam_ft/np.max(beam_ft)*np.max(ideal_beam)
        theta_out = np.angle(beam_ft)  # far field phase

        # desired focal spot intensity * phase from FFT
        new_beam_ft = np.square(ideal_beam) * np.exp(1j*theta_out)

        # inverse FFT
        new_beam_electric = np.fft.ifft(new_beam_ft)
        theta_in = np.angle(new_beam_electric) # near field phase
        print(_)

    theta_in = round_phase(theta_in)
    np.savetxt("phase_plate_2d.txt", X = theta_in,
               header = "Phase values [rad]")

    if plot:
        x = np.linspace(-std, std, n)
        x, y = np.meshgrid(x, x)
        fig = plt.figure()
        subpl = fig.add_subplot(111, projection = '3d')
        subpl.plot_surface(x, y, original_beam_electric)
        # subpl.plot_surface(x, y, ideal_beam)
        plt.show()

        x = np.linspace(-std, std, n)
        x, y = np.meshgrid(x, x)
        fig = plt.figure()
        subpl = fig.add_subplot(111, projection = '3d')
        subpl.plot_surface(x, y, np.abs(beam_ft))
        # subpl.plot_surface(x, y, ideal_beam)
        plt.show()

    return theta_in

def plot_phase_plate(thetas):
    '''
    Plot phase plates.
    
    Args:
        thetas: array of phase values
    '''
    plt.imshow(thetas, cmap = 'Greys')
    plt.show()

if __name__ == "__main__":
    # phase elements
    PHASE_ELEMENTS = 100

    # laser beam parameters
    AMPLITUDE = 5 # in J
    STD_DEV = 3 # in micron (FWHM/2.35482 for Gaussian)
    MOD_AMPLITUDE = 0.1 # in J
    MOD_FREQUENCY = 10 # in micron^-1

    # Gerchberg Saxton algorithm
    theta = gs_2d(n = PHASE_ELEMENTS, amp = AMPLITUDE, std = STD_DEV, mod_amp = MOD_AMPLITUDE,
            mod_freq = MOD_FREQUENCY, max_iter = int(1e4), plot = True)
    plot_phase_plate(theta)
