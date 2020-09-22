import pandas
import random
import datetime

import classes
import IO
from interface import get_orbit_from, check_validity, plot_data

import numpy as np
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates

from prepare_data import get_data, display_data

import time

import multiprocessing

def gaussian(x,a,b,c):
    return a*np.exp( -1*((x-b)**2)/(2*c**2) )

def normalised_gaussian(x,b,c):
    a = 1/(c*np.sqrt(2*3.141592))
    return a*np.exp( -1*((x-b)**2)/(2*c**2) )
    
def gaussian_algorithm(dataset, index):
        a = curve_fit(gaussian, dataset.iloc[index].index, dataset.iloc[index])
        return {'A':a[0][0], 'B':a[0][1], 'C':a[0][2]}

def normalised_gaussian_algorithm(dataset, index):
    a = curve_fit( normalised_gaussian, dataset.iloc[index].index,
                                     dataset.iloc[index]/dataset.iloc[index].max()   )
    return {'B':a[0][0], 'C':a[0][1]}
    
def scale(trace):
    trace = np.array(trace)
    return (trace/max(trace))-min(trace)/max(trace)

def find_region(trace, threshold=0.1):
    start = 0
    end = len(trace)-1
    max_index = 0
    max_val = trace.max()

    for i in range(len(trace)):
        if trace[i] >= max_val:
            max_index = i
            break
            
    for i in range(len(trace)):
        if trace[i] >= max_val*threshold:
            if i != 0:
                start = i-1
            else:
                start = i
            break
    for i in range(len(trace)):
        if trace[len(trace)-i-1] >= max_val*threshold:
            if i != 0:
                end = (len(trace)-i-1)+1
            else:
                end = (len(trace)-i-1)
            break
            

    return (max_index, start, end)

def gaussian_approx(data, base_threshold = 0.1, depth = 10):
    model = {'max':list()}

    for t in range(depth):
        model['width_{}'.format(t)] = list()
        model['region_max_{}'.format(t)] = list()
        model['region_min_{}'.format(t)] = list()
        threshold = base_threshold*t

        for i in range(len(data)):

            (max_index, start, end) = find_region(data.iloc[i], threshold)

            if t == 0:
                model['max'].append(max_index)

            model['width_{}'.format(t)].append(end-start)
            model['region_max_{}'.format(t)].append(start)
            model['region_min_{}'.format(t)].append(end)
            


    return pandas.DataFrame(model, index=data.index)
    
def gaussian_model(data):
    model = {'a':list(),'b':list(),'c':list(), 'c^2':list()}

    for i in range(len(data)):
        coefficients = gaussian_algorithm(data, i)

        model['a'].append(coefficients['A'])

        model['b'].append(coefficients['B'])

        model['c'].append(coefficients['C'])

        model['c^2'].append(coefficients['C']**2)
        
    return pandas.DataFrame(model, index=data.index)

def normalised_gaussian_model(data):

    model = {'b':list(),'c':list(), 'c^2':list()}

    for i in range(len(data)):
        coefficients = normalised_gaussian_algorithm(data, i)

        model['b'].append(coefficients['B'])

        model['c'].append(coefficients['C'])

        model['c^2'].append(coefficients['C']**2)
        
    return pandas.DataFrame(model, index=data.index)

def predict(dataset, model, index):
    """
    Computes slice of temporal dataset, use to check and compare gaussian fit.

    Usage:
        P = predict(dataset['default'],product['gaussian'],0)
        P.plot(figsize=(20,10))
        dataset['default'].plot()
        plt.show()
    """
    data=dataset.iloc[index]

    if 'a' in model:
        prediction = list()
        for i in range(len(data)):
            prediction.append( gaussian(i, model['a'], model['b'], model['c']) )
        return pandas.DataFrame(prediction, columns=['gaussian'], index=data.index)
    else:
        prediction = list()
        for i in range(len(data)):
            prediction.append( normalised_gaussian(i, model['b'], model['c']) )
        return pandas.DataFrame(prediction, columns=['norm_gaussian'], index=data.index)
