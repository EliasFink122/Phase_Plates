"""
Created on Tue Jul 09 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Generates continuous phase plates.

Methods:
    rms:
        calculates root mean square of array
"""

import numpy as np

def rms(arr: list) -> float:
    '''
    Root mean square value of array

    Args:
        arr: any array of numbers

    Returns:
        root mean square of array
    '''
    arr = np.array(arr)
    return np.sqrt(np.mean(arr*arr))

def iterate(n, amp, mod_amp, mod_freq, iter):
    '''
    Approximate value iteratively
    
    Args:
        n: number of phase elements
        amp: amplitude of laser in J
        mod_amp: modulation amplitude in J
        mod_freq: modulation frequency in Hz
        iter: number of iterations
    '''
    x = np.linspace(-3, 3, n)

    ideal_beam = amp*np.exp(-((x**2)/3)**5)

    rms_ideal = rms(ideal_beam)

    tol = 1.02*rms_ideal # 2% rms of ideal beam is considered smooth

    iter = 1000

    for _ in range(iter):

        input_beam_E_field = np.abs((amp * np.exp(mod_amp * (np.sin(mod_freq*x)**2) * (np.cos(theta_in) + np.imag(np.sin(theta_in)))) * np.exp(-((x**2)/3)**5) ))

        rms_input = rms(input_beam_E_field)

        if np.isclose(rms_input, rms_ideal):
            break

        G_out = np.fft.fft(input_beam_E_field)

        diff = np.sum(np.abs(G_out)**2 - ideal_beam)

        theta_out = np.angle(G_out)  #finds the far field phase

        g_new = np.sqrt(ideal_beam) * (np.cos(theta_out) + np.imag(np.sin(theta_out)))

        f_new = np.fft.ifft(g_new)

        theta_in = np.angle(f_new) #finds near field phase

        print(diff)
        print(theta_in)

if __name__ == "__main__":
    # initial phase elements
    PHASE_ELEMENTS = 1000
    theta_in = (np.pi/2)*np.random.randint(3, size = PHASE_ELEMENTS) # random phases 0, pi/2 or pi

    # laser beam parameters
    AMPLITUDE = 5 # in J
    MODULATION_AMPLITUDE = 0.6 # in J
    MODULATION_FREQUENCY = 10 # in Hz
    FWHM = 3 # in μm
    RADIUS_PARAMETER = FWHM/2.35482 # std dev for normal dist
