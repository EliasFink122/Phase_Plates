"""
Created on Tue Jul 09 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Generates 2-dimensional phase plates.

Methods:
    gs:
        use Gerchberg Saxton algorithm to iteratively find ideal phase plate phases
    plot_phase_plate:
        shows phase plate as black-white dots for phase
    circular_phase_plate:
        generates circular image of phase plate from square
"""

import numpy as np
import matplotlib.pyplot as plt
from PP_Tools import rms, ideal_beam_shape, modulation_beam, round_phase

def gs_2d(n: int, amp: float, mod_amp: float, mod_freq: float, std: float,
       max_iter: int = 1000, binarise: bool = True, plot: bool = False) -> np.ndarray:
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
        binarise: whether to binarise phase plate
        plot: whether to plot the input/output/ideal electric fields

    Returns:
        array of phases
    '''
    x = np.linspace(-std, std, n)
    xy = np.zeros((len(x), len(x), 2))
    for i, row in enumerate(xy):
        for j, _ in enumerate(row):
            xy[i, j] = [x[i], x[j]]
    i_arr = []
    convergence = []

    # ideal beam
    ideal_beam = ideal_beam_shape(xy, amp, std)

    # initial input beam
    theta_in = (np.pi/2)*np.random.randint(-2, 3, size = np.shape(xy)[:-1]) # random phases
    original_beam_electric = np.abs(modulation_beam(xy, amp, std, mod_amp, mod_freq, theta_in))

    for i in range(max_iter):
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

        # plotting convergence
        i_arr.append(i)
        convergence.append(np.sum(np.abs(np.abs(beam_ft) - ideal_beam))/np.sum(ideal_beam))

    print(f"Convergence accuracy: {100 - convergence[-1]*100:.2f} %")

    if binarise: # force binary phases of 0 or pi
        theta_in = round_phase(theta_in)

    np.savetxt("phase_plate_2d.txt", X = theta_in,
               header = "Phase values [pi rad]")
    print("Saved phase plate as txt file.")

    if plot:
        plt.figure()
        plt.title("Convergence of output to ideal")
        plt.plot(i_arr, np.array(convergence)*100)
        plt.xlabel("Steps")
        plt.ylabel("Difference [%]")
        plt.show()

        x = np.linspace(-std, std, n)
        x, y = np.meshgrid(x, x)
        fig = plt.figure()
        subpl = fig.add_subplot(111, projection = '3d')
        subpl.plot_surface(x, y, original_beam_electric)
        subpl.plot_surface(x, y, ideal_beam)
        subpl.set_xlabel("x [micron]")
        subpl.set_ylabel("y [micron]")
        subpl.set_zlabel("Energy [J]")
        plt.show()

        x = np.linspace(-std, std, n)
        x, y = np.meshgrid(x, x)
        fig = plt.figure()
        subpl = fig.add_subplot(111, projection = '3d')
        subpl.plot_surface(x, y, np.abs(beam_ft))
        subpl.plot_surface(x, y, ideal_beam)
        subpl.set_xlabel("x [micron]")
        subpl.set_ylabel("y [micron]")
        subpl.set_zlabel("Energy [J]")
        plt.show()

    return theta_in

def plot_phase_plate(thetas: np.ndarray):
    '''
    Plot phase plates.
    
    Args:
        thetas: array of phase values
    '''
    plt.imshow(thetas, cmap = 'Greys')
    plt.show()

def circular_phase_plate(thetas: np.ndarray) -> np.ndarray:
    '''
    Make phase plate circular
    
    Args:
        thetas: array of phase values

    Returns:
        circularised phase plate
    '''
    radius = np.round(len(thetas)/2)
    new_thetas = thetas
    element_count = 0
    for i, row in enumerate(thetas):
        for j, _ in enumerate(row):
            if np.linalg.norm([i - radius, j - radius]) > radius:
                new_thetas[i, j] = 0
            else:
                element_count += 1
    np.savetxt("phase_plate_circular_2d.txt", X = new_thetas,
               header = "Phase values [pi rad]")
    print(f"Number of circular plate phase elements: {element_count}")
    plt.imshow(new_thetas, cmap = 'Greys')
    plt.show()

    return new_thetas

if __name__ == "__main__":
    # phase elements
    PHASE_ELEMENTS = 100

    # laser beam parameters
    AMPLITUDE = 5 # in J
    STD_DEV = 3 # in micron (FWHM/2.35482 for Gaussian)
    MOD_AMPLITUDE = 0.1 # in J
    MOD_FREQUENCY = 10 # in micron^-1

    # Gerchberg Saxton algorithm
    print("--- Running Gerchberg Saxton algorithm ---")
    print(f"Total number of phase elements: {PHASE_ELEMENTS**2}")
    theta = gs_2d(n = PHASE_ELEMENTS, amp = AMPLITUDE, std = STD_DEV, mod_amp = MOD_AMPLITUDE,
            mod_freq = MOD_FREQUENCY, max_iter = int(1e4), plot = True)
    circular_phase_plate(theta)
