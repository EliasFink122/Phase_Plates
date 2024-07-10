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
from PP_Tools import ideal_beam_shape, modulation_beam, round_phase, circular_phase_plate, plot_phase_plate

def gs_2d(n: int, amp: float, std: float, mod_amp: float, mod_freq: float,
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
        if int(i/max_iter / 0.05) != int((i-1)/max_iter / 0.05):
            print(f"GS algorithm: {int((i/max_iter) * 100)} %")
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

    print(f"Continuous convergence accuracy: {100 - convergence[-1]*100:.2f} %")

    if binarise: # force binary phases of 0 or pi
        theta_in = round_phase(theta_in)
        bin_beam_electric = np.square(original_beam_electric) * np.exp(1j*theta_in*np.pi)
        bin_beam_ft = np.fft.fft(bin_beam_electric)
        bin_beam_ft = bin_beam_ft/np.max(bin_beam_ft)*np.max(ideal_beam)
        conv = np.sum(np.abs(np.abs(bin_beam_ft) - ideal_beam))/np.sum(ideal_beam)
        print(f"Binarised convergence accuracy: {100 - conv*100:.2f} %")

    np.savetxt("Outputs/phase_plate_2d.txt", X = theta_in,
               header = "Phase values [pi rad]")
    print("Saved phase plate as txt file.")

    if plot: # plot 3d graphs of all beams
        plt.figure()
        plt.title("Convergence of output to ideal")
        plt.plot(i_arr, np.array(convergence)*100)
        plt.xlabel("Steps")
        plt.ylabel("Difference [%]")
        plt.show()

        x = np.linspace(-std, std, n)
        x, y = np.meshgrid(x, x)

        fig = plt.figure()
        fig.suptitle("Beams")
        if binarise:
            ax1 = fig.add_subplot(2, 2, 1, projection = '3d')
        else:
            ax1 = fig.add_subplot(1, 3, 1, projection = '3d')
        ax1.set_title("Ideal beam")
        ax1.plot_surface(x, y, ideal_beam)
        ax1.set_xlabel("x [micron]")
        ax1.set_ylabel("y [micron]")
        ax1.set_zlabel("Energy [J]")

        if binarise:
            ax2 = fig.add_subplot(2, 2, 2, projection = '3d')
        else:
            ax2 = fig.add_subplot(1, 3, 2, projection = '3d')
        ax2.set_title("Original beam")
        ax2.plot_surface(x, y, original_beam_electric)
        ax2.set_xlabel("x [micron]")
        ax2.set_ylabel("y [micron]")
        ax2.set_zlabel("Energy [J]")

        if binarise:
            ax3 = fig.add_subplot(2, 2, 3, projection = '3d')
        else:
            ax3 = fig.add_subplot(1, 3, 3, projection = '3d')
        ax3.set_title("Continuous phase plate beam")
        ax3.plot_surface(x, y, np.abs(beam_ft))
        ax3.set_xlabel("x [micron]")
        ax3.set_ylabel("y [micron]")
        ax3.set_zlabel("Energy [J]")

        if binarise:
            bin_beam_electric = np.square(original_beam_electric) * np.exp(1j*theta_in*np.pi)
            bin_beam_ft = np.fft.fft(bin_beam_electric)
            bin_beam_ft = bin_beam_ft/np.max(bin_beam_ft)*np.max(ideal_beam)
            ax4 = fig.add_subplot(2, 2, 4, projection = '3d')
            ax4.set_title("Binarised phase plate beam")
            ax4.plot_surface(x, y, np.abs(bin_beam_ft))
            ax4.set_xlabel("x [micron]")
            ax4.set_ylabel("y [micron]")
            ax4.set_zlabel("Energy [J]")

        plt.show()

    return theta_in

if __name__ == "__main__":
    # INSTRUCTIONS:
    # Adjust all parameters of laser beam and phase plate
    # Optional: create new arbitrary noise pattern in PP_Tools

    # laser beam parameters
    AMPLITUDE = 5 # in J
    STD_DEV = 3 # in micron (FWHM/2.35482 for Gaussian)
    MOD_AMPLITUDE = 0.1 # in J
    MOD_FREQUENCY = 10 # in micron^-1

    # Phase plate parameters
    PHASE_ELEMENTS = 100
    MAX_ITER = 1e4
    BINARISE = True
    PLOT = True
    CIRCULARISE = True

    # Gerchberg Saxton algorithm
    print("--- Construction of 2-d phase plate ---")
    print(f"Total number of phase elements: {PHASE_ELEMENTS**2}")
    print("Running Gerchberg Saxton algorithm")
    theta = gs_2d(n = PHASE_ELEMENTS, amp = AMPLITUDE, std = STD_DEV, mod_amp = MOD_AMPLITUDE,
            mod_freq = MOD_FREQUENCY, max_iter = int(MAX_ITER), binarise = BINARISE, plot = PLOT)
    if CIRCULARISE:
        circular_phase_plate(theta)
    else:
        plot_phase_plate(theta)
