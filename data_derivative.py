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
        try:
            a = curve_fit(gaussian, dataset.iloc[index].index, dataset.iloc[index])
            return {'A':a[0][0], 'B':a[0][1], 'C':a[0][2]}
        except:
            return {'A': None, 'B': None, 'C': None}

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

        try:
            model['c^2'].append(coefficients['C']**2)
        except:
            model['c^2'].append(None)
        
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

def constrain_time(dataset, start_hour, start_minute, end_hour, end_minute):
    old_date = dataset.index[0]
    hour = start_hour
    minute = start_minute

    if hour < old_date.hour:
        base_date = old_date + datetime.timedelta(days=1)
    else:
        base_date = old_date

    new_date = dict()
    new_date[0] = datetime.datetime(base_date.year, base_date.month, base_date.day, hour, minute)

    hour = end_hour
    minute = end_minute

    if hour < old_date.hour:
        base_date = old_date + datetime.timedelta(days=1)
    else:
        base_date = old_date

    new_date[1] = datetime.datetime(base_date.year, base_date.month, base_date.day, hour, minute)

    return constrain(dataset, new_date[0], new_date[1])

def find_common_edges(dataset, key):
    if 'IMA' in dataset:
        latest_start = dataset['IMA'][key].index[0]
        earliest_end = dataset['IMA'][key].index[-1]
    elif 'ELS' in dataset:
        latest_start = dataset['ELS'][key].index[0]
        earliest_end = dataset['ELS'][key].index[-1]
    elif 'MAG' in dataset:
        return (  dataset['MAG'][key].index[0],  dataset['MAG'][key].index[-1] )
    else:
        return (None, None)
        
    for instrument in ['ELS', 'MAG']:

        if instrument in dataset:
            
            if dataset[instrument][key].index[0] > latest_start:
                latest_start = dataset[instrument][key].index[0]

            if dataset[instrument][key].index[-1]< earliest_end:
                earliest_end = dataset[instrument][key].index[-1]
            
    return (latest_start, earliest_end)

def constrain(data, start_time, end_time):
    start_index = 0
    end_index = -1

    for i, time in enumerate(data.index):

        if time >= start_time:
            start_index = i
            break

    for i, time in enumerate(data.index):

        if time == end_time:
            break
        elif time > end_time:
            end_index = i
            break

    return data.iloc[start_index:end_index]

def enhance_dataset(dataset, key='default'):
    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()

    if 'IMA' in dataset:
        s = min_max_scaler.fit_transform(dataset['IMA'][key].T)
        dataset['IMA']['enhanced'] = pandas.DataFrame(s).T
        dataset['IMA']['enhanced'].index = dataset['IMA'][key].index

    if 'ELS' in dataset:
        s = min_max_scaler.fit_transform(dataset['ELS'][key].T)
        dataset['ELS']['enhanced'] = pandas.DataFrame(s).T
        dataset['ELS']['enhanced'].index = dataset['ELS'][key].index

    if 'MAG' in dataset:
        s = min_max_scaler.fit_transform(dataset['MAG'][key])
        dataset['MAG']['enhanced'] = pandas.DataFrame(s)
        dataset['MAG']['enhanced'].index = dataset['MAG'][key].index

    return dataset

def get_hourly_ticks(indices):
    last_hour = indices[0].hour
    hourly_ticks = list()

    for i, index in enumerate(indices):
        if index.hour != last_hour:
            hourly_ticks.append([i, index])
            last_hour = index.hour
    return hourly_ticks

def tri_plot(dataset, labels, figsize=(20,10)):
    image = dict()
    im = dict()
    ticks = dict()

    fig, ax = plt.subplots(3,1,figsize=figsize)

    latest_start, earliest_end = find_common_edges(dataset, labels[0])

    if 'MAG' in dataset:
        
        MAG_label=labels[2]
        MAG = constrain(dataset['MAG'][MAG_label], latest_start, earliest_end)
        
        im[2] = ax[2].plot(MAG)
        ticks['MAG'] = get_hourly_ticks(MAG.index)
        ax[2].set_xticks( [i[1] for i in ticks['MAG']] )
        ax[2].set_xticklabels(["{}h\n({})".format(Y[1].hour, Y[0]) for Y in ticks['MAG']])
        ax[2].set_ylabel('MAG\nmagnetometer\n({})'.format(MAG_label))
        ax[2].set_xlim(MAG.index[0], MAG.index[-1])


    if 'ELS' in dataset:
        
        ELS_label=labels[1]
        ELS = constrain(dataset['ELS'][ELS_label], latest_start, earliest_end)
        image['ELS'] = ELS.T
        
        im[1] = ax[1].imshow(image['ELS'], interpolation='nearest', aspect='auto')
        ticks['ELS'] = get_hourly_ticks(ELS.index)
        ax[1].set_xticks( [i[0] for i in ticks['ELS']] )
        ax[1].set_xticklabels(["{}h\n({})".format(Y[1].hour, Y[0]) for Y in ticks['ELS']])
        ax[1].set_ylabel('ELS\nelectron spectrometer\n({})'.format(ELS_label))


    if 'IMA' in dataset:
        
        IMA_label=labels[0]
        IMA = constrain(dataset['IMA'][IMA_label], latest_start, earliest_end)
        image['IMA'] = IMA.T
        
        im[0] = ax[0].imshow(image['IMA'], interpolation='nearest', aspect='auto')
        ticks['IMA'] = get_hourly_ticks(IMA.index)
        ax[0].set_xticks( [i[0] for i in ticks['IMA']] )
        ax[0].set_xticklabels(["{}h\n({})".format(Y[1].hour, Y[0]) for Y in ticks['IMA']])
        ax[0].set_ylabel('IMA\nion mass analyser\n({})'.format(IMA_label))


    plt.show()
