"""
Created on Tue Jul 09 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Generates continuous phase plates.

Methods:
    rms:
        calculates root mean square of array
    ideal_beam_shape:
        super Gaussian shape of ideal laser beam
    modulation_beam:
        random complex modulation added to approximate real laser beam
    round_phase:
        round all phases to multiples of pi
    gs:
        use Gerchberg Saxton algorithm to iteratively find ideal phase plate phases
"""

import numpy as np
import matplotlib.pyplot as plt

def rms(arr: list) -> float:
    '''
    Root mean square value of array

    Args:
        arr: any array of numbers

    Returns:
        root mean square of array
    '''
    arr = np.array(arr)
    return np.sqrt(np.mean(np.abs(arr)**2))

def ideal_beam_shape(x: float, amp: float, std: float) -> float:
    '''
    Super Gaussian function A*e^(-(x^2/(2*std^2))^5)
    
    Args:
        x: parameter
        amp: amplitude of beam
        std: standard deviation

    Returns:
        value of laser beam
    '''
    return amp*np.exp(-((x**2)/(2*std**2))**5)

def modulation_beam(x: float, amp: float, std: float, mod_amp: float,
                    mod_freq: float, phase: float) -> float:
    '''
    Adds modulation to the beam shape
    
    Args:
        x: independent variable
        mod_amp: amplitude of modulation
        mod_freq: frequency of modulation
        phase: complex phase of modulation

    Returns:
        value of modulation
    '''
    modulation = np.exp(mod_amp * np.sin(mod_freq*x)**2 * np.exp(1j*phase))
    return ideal_beam_shape(x, amp, std) * modulation

def round_phase(arr: list, tol: float = 1e-2) -> np.ndarray:
    '''
    Round phases to multiples of pi

    Args:
        arr: input phase list

    Returns:
        rounded list
    '''
    for i, theta in enumerate(arr):
        if np.isclose(theta, np.pi, atol = tol):
            arr[i] = np.pi
        elif np.isclose(theta, -np.pi, atol = tol):
            arr[i] = -np.pi
        else:
            arr[i] = 0
    return np.array(arr)

def gs(n: int, amp: float, mod_amp: float, mod_freq: float, std: float,
       max_iter: int = 1000, plot: bool = False) -> np.ndarray:
    '''
    Gerchberg Saxton algorithm:
    Approximate plate phases iteratively
    
    Args:
        n: number of phase elements
        amp: amplitude of laser in J
        mod_amp: modulation amplitude in J
        mod_freq: modulation frequency in Hz
        std: standard deviation of super Gaussian beam
        max_iter: maximum number of iterations
        plot: whether to plot the input/output/ideal electric fields
    '''
    x = np.linspace(-std, std, n)

    # ideal beam
    ideal_beam = ideal_beam_shape(x, amp, std)
    rms_ideal = rms(ideal_beam)

    # initial input beam
    theta_in = (np.pi/2)*np.random.randint(3, size = n) # random phases 0, pi/2, pi
    original_beam_electric = np.abs(modulation_beam(x, amp, std, mod_amp, mod_freq, theta_in))

    for _ in range(max_iter):
        input_beam_electric = np.abs(modulation_beam(x, amp, std, mod_amp, mod_freq, theta_in))

        rms_input = rms(input_beam_electric)

        if np.isclose(rms_input, rms_ideal):
            break

        # FFT of beam
        beam_ft = np.fft.fft(input_beam_electric)
        theta_out = np.angle(beam_ft)  # far field phase

        # discarding old amplitude and only using phase
        new_beam_ft = np.sqrt(ideal_beam) * (np.exp(1j*theta_out))

        # inverse FFT of beam
        new_beam_electric = np.fft.ifft(new_beam_ft)
        theta_in = np.angle(new_beam_electric) # near field phase

    # round phases to multiples of pi
    theta_in = round_phase(theta_in)

    np.savetxt("phase_plate.txt", X = theta_in,
               header = "Phase values [rad]")

    if plot:
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(x, original_beam_electric, label = 'Input beam')
        ax1.plot(x, ideal_beam, label = 'Ideal beam')
        ax2.plot(x, np.abs(input_beam_electric), label = 'Output beam')
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
