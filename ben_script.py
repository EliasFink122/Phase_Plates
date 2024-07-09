#phase plate - continuous

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci 
import math 
import cmath

#Function that Calculate Root Mean Square 
def rmsValue(arr, n):
    square = 0
    mean = 0.0
    root = 0.0
     
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
     
    #Calculate Mean 
    mean = (square / float(n))
     
    #Calculate Root
    root = math.sqrt(mean)
     
    return root

################################
# Define the inital phase mask
################################

length = 1000 #number of phase elements
phase_initial = np.random.randint(2, size = length) #randomly populate the array, number of integers defines the number of phase steps
theta_in = (np.pi/2)*phase_initial #give them a phase change between 0 and pi

print(phase_initial)

################################
# initial laser near field parameters - i.e input beam
################################

amplitude = 5 #joules
modulation_amplitude = 0.6
modulation_frequency = 10
FWHM = 3 #micron
radius_parameter = FWHM/2.35482

###########################
#start of iteration
###########################

x = np.linspace(-3,3,length)

ideal_beam = amplitude*np.exp(-((x**2)/3)**5)

rms_ideal = rmsValue(ideal_beam, length)

tol = 1.02*rms_ideal # 2% rms of ideal beam is considered smooth

iter = 1000

for i in range(0,iter,1):

    input_beam_E_field = np.abs((amplitude * np.exp(modulation_amplitude * (np.sin(modulation_frequency*x)**2) * (np.cos(theta_in) + np.imag(np.sin(theta_in)))) * np.exp(-((x**2)/3)**5) ))
    
    rms_input = rmsValue(input_beam_E_field, length)

    if rms_input == rms_ideal:
        break

    G_out = np.fft.fft(input_beam_E_field) 

    diff = np.sum(np.abs(G_out)**2 - ideal_beam)

    theta_out = np.angle(G_out)  #finds the far field phase

    g_new = np.sqrt(ideal_beam) * (np.cos(theta_out) + np.imag(np.sin(theta_out)))

    f_new = np.fft.ifft(g_new)

    theta_in = np.angle(f_new) #finds near field phase

    print(diff)
    print(theta_in)


    
#

plt.plot(x,np.real(input_beam_E_field))
plt.plot(x,ideal_beam)
plt.show()
#plt.plot(x,input_beam)
#plt.plot(x,ideal_beam)
#plt.show()

