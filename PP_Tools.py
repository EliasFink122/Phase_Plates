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
    arbitrary_noise:
        generate any arbitrary noise pattern on top of ideal beam
    round_phase:
        round all phases to multiples of pi
    read_in:
        read in phase plate data for analysis
    plot_phase_plate:
        shows phase plate as black-white dots for phase
    circular_phase_plate:
        generates circular image of phase plate from square
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
    if len(np.shape(x)) == 3: # for 2-dimensional
        x = np.linalg.norm(x, axis = 2)
    return amp*np.exp(-((x**2)/(2*std**2))**5)

def modulation_beam(x: float | np.ndarray, amp: float, std: float, mod_amp: float,
                    mod_freq: float, phase: float | np.ndarray, arb: bool = False) -> float:
    '''
    Adds modulation to the beam shape.
    Change this for more control about the initial beam shape.

    Args:
        x: independent variable
        mod_amp: amplitude of modulation
        mod_freq: frequency of modulation
        phase: complex phase of modulation
        arb: whether to use arbitrary noise

    Returns:
        value of modulation
    '''
    if arb: # arbitrary noise patterns
        modulation = arbitrary_noise(x)
    else: # test noise pattern
        if len(np.shape(x)) == 3: # for 2-dimensional
            x = np.linalg.norm(x, axis = 2)
        modulation = np.exp(mod_amp * np.sin(mod_freq*x)**2 * np.exp(1j*phase))
    return ideal_beam_shape(x, amp, std) * modulation

def arbitrary_noise(x: float | np.ndarray) -> float | np.ndarray:
    '''
    Add arbitrary noise patterns here.
    
    Args:
        x: independent variable(s)

    Returns:
        noise value (must be constant or have correct dimensions)
    '''
    print(f"Called arbitrary noise at position {x}")

    return 1

def round_phase(arr: list) -> np.ndarray:
    '''
    Round phases to 0 or pi (binarise)

    Args:
        arr: input phase list

    Returns:
        rounded list
    '''
    # 1-dimensional
    new_arr = np.array(arr)
    arr = np.array(np.abs(arr))
    thresh = np.median(arr)
    print(f"Binarising with threshold: {thresh/np.pi:.2f} pi")
    if len(np.shape(arr)) == 1:
        for i, theta in enumerate(arr):
            if theta >= thresh:
                new_arr[i] = np.pi
            else:
                new_arr[i] = 0
        loss = np.sum(np.abs(new_arr - arr)/np.pi)/len(arr)
    # 2-dimensional
    elif len(np.shape(arr)) == 2:
        for i, row in enumerate(arr):
            for j, theta in enumerate(row):
                if theta >= thresh:
                    new_arr[i, j] = np.pi
                else:
                    new_arr[i, j] = 0
        loss = np.sum(np.abs(new_arr - arr)/np.pi)/(len(arr)**2)
    print(f"Loss of binarisation: {loss*100:.2f} %")
    return new_arr

def read_in(path: str) -> np.ndarray:
    '''
    Read in phase plate data from txt file

    Args:
        path: absolute path to phase plate txt file

    Returns:
        phase information
    '''
    phase_information = np.loadtxt(path, delimiter = ' ', skiprows = 1)
    return phase_information

def plot_phase_plate(thetas: np.ndarray, plate_type: str):
    '''
    Plot phase plates.
    
    Args:
        thetas: array of phase values
        plate_type: type of plate (random or zonal)
    '''
    if plate_type == "random":
        plt.title("Random Phase Plate")
    elif plate_type == "zonal":
        plt.title("Phased Zonal Plate")
    plt.imshow(thetas, cmap = 'Greys')
    plt.show()

def circular_phase_plate(thetas: np.ndarray, plate_type: str) -> np.ndarray:
    '''
    Make phase plate circular
    
    Args:
        thetas: array of phase values
        plate_type: type of plate (random or zonal)

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
    print(f"Number of circular plate phase elements: {element_count}")
    if plate_type == "random":
        np.savetxt("Outputs/random_phase_plate_circular_2d.txt", X = new_thetas,
               header = "Phase values [rad]")
        plt.title("Random Phase Plate")
    elif plate_type == "zonal":
        np.savetxt("Outputs/phased_zonal_plate_circular_2d.txt", X = new_thetas,
               header = "Phase values [rad]")
        plt.title("Phased Zonal Plate")
    plt.imshow(new_thetas, cmap = 'Greys')
    plt.show()

    return new_thetas

def smooth(arr: list) -> np.ndarray:
    '''
    Smooth array of float values

    Args:
        arr: list to smooth

    Returns:
        smoothed numpy array
    '''
    smoothed_arr = np.zeros(np.shape(arr))
    if len(np.shape(arr)) == 1:
        for i, val in enumerate(arr):
            if i == 0:
                smoothed_arr[i] = (3*val + arr[i+1])/4
                continue
            if i == len(arr)-1:
                smoothed_arr[i] = (3*val + arr[i-1])/4
                continue
            smoothed_arr[i] = (2*val + arr[i+1] + arr[i-1])/4
    elif len(np.shape(arr)) == 2:
        for i, row in enumerate(arr):
            for j, val in enumerate(row):
                if i == 0:
                    if j == 0:
                        smoothed_arr[i, j] = (4*val + arr[i+1, j] + arr[i, j+1])/6
                        continue
                    if j == len(row)-1:
                        smoothed_arr[i, j] = (4*val + arr[i+1, j] + arr[i, j-1])/6
                        continue
                    smoothed_arr[i, j] = (3*val + arr[i+1, j] + arr[i, j+1] + arr[i, j-1])/6
                    continue
                if i == len(arr)-1:
                    if j == 0:
                        smoothed_arr[i, j] = (4*val + arr[i-1, j] + arr[i, j+1])/6
                        continue
                    if j == len(row)-1:
                        smoothed_arr[i, j] = (4*val + arr[i-1, j] + arr[i, j-1])/6
                        continue
                    smoothed_arr[i, j] = (3*val + arr[i-1, j] + arr[i, j+1] + arr[i, j-1])/6
                    continue
                if j == 0:
                    smoothed_arr[i, j] = (3*val + arr[i, j+1] + arr[i-1, j] + arr[i+1, j])/6
                    continue
                if j == len(row)-1:
                    smoothed_arr[i, j] = (3*val + arr[i, j-1] + arr[i-1, j] + arr[i+1, j])/6
                    continue
                smoothed_arr[i, j] = (2*val + arr[i+1, j] + arr[i-1, j] + arr[i, j+1] + arr[i, j-1])/6
    else:
        raise ValueError("Only 1 or 2 dimensional arrays allowed!")
    return smoothed_arr
