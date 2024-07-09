"""
Created on Tue Jul 09 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Tools for phase plate codes.

Methods:
    rms:
        calculates root mean square of array
    ideal_beam_shape:
        super Gaussian shape of ideal laser beam
    modulation_beam:
        random complex modulation added to approximate real laser beam
    round_phase:
        round all phases to multiples of pi
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
    if len(np.shape(x)) == 3: # for 2-dimensional
        x = np.linalg.norm(x, axis = 2)
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
    if len(np.shape(x)) == 3: # for 2-dimensional
        x = np.linalg.norm(x, axis = 2)
    modulation = np.exp(mod_amp * np.sin(mod_freq*x)**2 * np.exp(1j*phase))
    return ideal_beam_shape(x, amp, std) * modulation

def round_phase(arr: list) -> np.ndarray:
    '''
    Round phases to 0 or pi

    Args:
        arr: input phase list

    Returns:
        rounded list
    '''
    # 1-dimensional
    if len(np.shape(arr)) == 1:
        for i, theta in enumerate(arr):
            if np.abs(theta) >= np.pi/2:
                arr[i] = np.pi
            else:
                arr[i] = 0
    # 2-dimensional
    elif len(np.shape(arr)) == 2:
        for i, row in enumerate(arr):
            for j, theta in enumerate(row):
                if np.abs(theta) >= np.pi/2:
                    arr[i, j] = np.pi
                else:
                    arr[i, j] = 0
    return np.array(arr)
