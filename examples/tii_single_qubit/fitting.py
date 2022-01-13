import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def rabi_fit(dataset):
    pguess = [
        np.mean(dataset['y0'].values),
        np.max(dataset['y0'].values) - np.min(dataset['y0'].values),
        0.5/dataset['x0'].values[np.argmin(dataset['y0'].values)], 
        np.pi/2,
        0.1e-6
    ]
    popt, pcov = curve_fit(rabi, dataset['x0'].values, dataset['y0'].values, p0=pguess)
    smooth_dataset = rabi(dataset['x0'].values, *popt)
    pi_pulse_duration = np.abs((1.0 / popt[2]) / 2)
    rabi_oscillations_pi_pulse_min_voltage = smooth_dataset.min() * 1e6
    t1 = 1.0 / popt[4]
    return smooth_dataset, pi_pulse_duration, rabi_oscillations_pi_pulse_min_voltage, t1

def t1_fit(dataset):
    pguess = [
        max(dataset['y0'].values),
        (max(dataset['y0'].values) - min(dataset['y0'].values)),
        1/250
    ]
    popt, pcov = curve_fit(exp, dataset['x0'].values, dataset['y0'].values, p0=pguess)
    smooth_dataset = exp(dataset['x0'].values, *popt)
    t1 = abs(1/popt[2])
    return smooth_dataset, t1

def ramsey_fit(dataset):
    pguess = [
        np.mean(dataset['y0'].values),
        np.max(dataset['y0'].values) - np.min(dataset['y0'].values),
        0.5/dataset['x0'].values[np.argmin(dataset['y0'].values)], 
        np.pi/2,
        0.1e-6
    ]
    popt, pcov = curve_fit(ramsey, dataset['x0'].values, dataset['y0'].values, p0=pguess)
    smooth_dataset = ramsey(dataset['x0'].values, *popt)
    delta_frequency = popt[2]
    t2 = 1.0 / popt[4]
    return smooth_dataset, delta_frequency, t2
    
def rabi(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : 1/p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    #return p[0] + p[1] * np.sin(2 * np.pi / p[2] * x + p[3]) * np.exp(-x / p[4])
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(- x * p4)

def ramsey(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    #return p[0] + p[1] * np.sin(2 * np.pi / p[2] * x + p[3]) * np.exp(-x / p[4])
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(- x * p4)


def exp(x,*p) :
    return p[0] - p[1]*np.exp(-1 * x * p[2])    