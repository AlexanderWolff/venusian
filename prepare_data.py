import pandas
import random
import datetime

import classes
import IO
from interface import get_orbit_from, check_validity, plot_data

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates


def produce_data(orbit, datapath, raw = False, preview = False, debug=False, verbose=False):

    dataset = dict()
    
    instruments = ["IMA", "ELS", "MAG"]
    
    for instrument in instruments:
    
        data = get_data( orbit, instrument, datapath, raw, preview, debug, verbose )
        
        if data != (None, None):
            dataset[instrument] = data
        else:
            print("Error: {} not found.".format(instrument))
    return dataset


def get_data(orbit, instrument, datapath, raw = False, preview = False, debug=False, verbose=False):
    if verbose:
        print('orbit :{}'.format(orbit))

    # pull data
    frame, t = get_orbit_from(path=datapath, orbit=orbit)

    if preview:
        plot_data(frame,t)

    data = (frame, t)

    try:
        dataset = dict()
    
        if instrument == 'IMA' or instrument == 'ELS':
            data_index, sample_rate = temporal_sections(data, instrument=instrument, sample_rate = 'common', max_skip = 5, verbose=verbose)

            dataframe = interpolate_sections(data, data_index, sample_rate = sample_rate, instrument=instrument, verbose=verbose)

            dataframe = regroup(dataframe, instrument=instrument, sample_rate = sample_rate,  max_loss = 100, verbose=verbose)
            dataset['sample_rate'] = sample_rate
        else:
            dataframe = pandas.DataFrame(frame[instrument], t[instrument])
            
            data_index, sample_rate = temporal_sections(data, instrument=instrument, sample_rate = 'common', max_skip = 5, verbose=verbose)
            
            dataset['sample_rate'] = sample_rate
            
        dataset['raw'] = dataframe
        
        dataset['default'] = dataset['raw']
        
        if not raw:
        
            if instrument == 'IMA':
                
                # there are times when the IMA data is not segmented into 16 periodic signals (eg. 2608)
                # detection of these not yet implemented
                dataset['default'] = combine_IMA(dataframe)
                dataset['streams'] = separate_IMA(dataframe)
                
        
            elif instrument == 'ELS':
                if find_if_split(dataframe):
                    if preview or verbose or debug:
                        print("Warning: Four Streams Detected, preparing mitigation.")
                        
                    dataset['streams'] = separate_ELS(dataframe)
                    
                    dataset['default'] = combine_ELS(dataframe)
                    
            elif instrument == 'MAG':
                # linear might not be the best method of interpolation
                dataset['default'] = dataset['raw'].interpolate(method = 'linear')
                
        return dataset
    except:
        if preview or verbose or debug:
            print("Error: Instrument not available.")
        return None, None

def find_if_split(dataframe):
    traces = list()
    for i in range(4):
        a = i*32
        b = ((i+1)*32)-1
        part = dataframe.T[a:b].T
        trace = part.mean().values.tolist()
        traces.append(trace)

    # normalise to 1
    for i, trace in enumerate(traces):
        traces[i] = np.array(traces[i])
        traces[i] = traces[i]/traces[i].max()

    # calculate mean squared error of all permutations
    mean_squared_error = list()

    for i in range(3):
        for j in range(3-i):
            k = 3-j
            if i == k:
                continue
            else:
                mean_squared_error.append( (traces[i]-traces[j]).mean()**2 )

    # find overall error
    E = sum(mean_squared_error)

    # threshold is a magic number tuned through trial and error:
    # when 4:
    # 2.495214912134985e-06
    # 1.2413893061441248e-06
    #
    # when 1:
    # 0.010172269857076926
    # 1.2742714511405433
    # 0.16337868220793833
    # 1.2132394469419747

    threshold = 1e-4
    if E > threshold:
        return False
    else:
        # four parallel streams detecteds
        return True
     

def temporal_sections( data, instrument, sample_rate = 'common', max_skip = 5, verbose = False ):
    frame, t = data

    if verbose:
        print('instrument :{}'.format(instrument))

    time_diff = pandas.Series([t[instrument][i+1]-t[instrument][i] for i in range(len(t[instrument])-1)])

    if verbose:
        print('\n')
    try:

        if verbose:
            print('Min Sample Delay:  {}'.format(time_diff.min().round('1s')))
            print('Mean Sample Delay: {}'.format(time_diff.mean().round('1s')))
            print('Max Sample Delay:  {}'.format(time_diff.max().round('1s')))
            print('\n')

        buckets = dict()
        for i in range(len(time_diff)):

            x = time_diff[i].round('1s')

            try:
                buckets[x] = buckets[x]+1
            except:
                buckets[x] = 1

        if verbose:
            [print('sample delay : {} \t events {}'.format(key.seconds, buckets[key])) for key in buckets.keys()]
    except:
        if verbose:
            print('instrument not found')


    # assuming that true sample rate is the most common rate

    if sample_rate == 'common':
        # invert key and values
        stekcub = dict()
        for key in buckets.keys():
            stekcub[buckets[key]] = key

        sample_rate = stekcub[max(stekcub)]
    else:
        sample_rate = datetime.timedelta(seconds=sample_rate)

    if verbose:
        print('sample rate : {} s = {:.3} Hz'.format(sample_rate.seconds, 1/sample_rate.seconds))

    data_index = list()
    section = list()
    section.append(0)

    # maximum amounts of samples skipped in a row before initiating new section
    for i in range( len(time_diff) ):

        sample = i+1 #t[instrument][i+1]

        time_difference = time_diff[i].round('1s')

        # skip when no time has passed (assume error)
        if time_difference.seconds==0:
            continue

        if time_difference==sample_rate:
            section.append(sample)
        else:

            skip = int(time_difference.seconds/sample_rate.seconds)
            if skip <= max_skip:
                    for j in range(skip-1):
                        section.append(None)
                    section.append(sample)
            else:
                data_index.append(section)
                section = list()
                section.append(sample)
    data_index.append(section)

    if verbose:
        print('sections : {} of length : \n'.format(len(data_index)))
        x=[print(len(data_index[i])) for i in range(len(data_index))]

    return data_index, sample_rate




def interpolate_sections(data, data_index, instrument, sample_rate, verbose = False):
    frame, t = data

    # transpose IMA and ELS so that the data may be interpolated
    reframe = dict()
    reframe[instrument] = frame[instrument].transpose()

    time = dict()
    time[instrument] = list()

    dataframe = dict()
    dataframe[instrument] = list()

    empty = [None for i in range(len(reframe[instrument][0]))]

    for section in data_index:

        data = list()
        data_time = list()

        for data_point in section:

            if data_point == None:
                data.append(empty)
                data_time.append(data_time[len(data_time)-1]+datetime.timedelta(seconds=sample_rate.seconds))

            else:
                data.append(reframe[instrument][data_point])
                data_time.append(t[instrument][data_point])


        dataframe[instrument].append( pandas.DataFrame(data, index=pandas.Series(data_time)) )
        time[instrument].append( pandas.Series(data_time) )

    for i, section in enumerate(dataframe[instrument]):
        dataframe[instrument][i] = section.interpolate()
        
    if verbose:
        pandas.options.display.max_columns = 10
        pandas.options.display.max_rows = None
        print(dataframe)
        
    return dataframe


def regroup(dataframe, instrument, sample_rate, max_loss = 100, verbose = False):
# regroup samples based on max loss permitted

    samples = list()
    starts = list()
    ends = list()

    for i, section in enumerate(dataframe[instrument]):

        end = len(section)

        amount_of_samples = 1+np.ceil((section.index[end-1]-section.index[0]).seconds/sample_rate.seconds).astype(int)

        starts.append(section.index[0])
        ends.append(section.index[end-1])

        samples.append(amount_of_samples)

        if verbose:
            print("section {} \t starts at {} ends at {} : total of {} samples".format(i,
                    section.index[0], section.index[end-1], amount_of_samples))

    final_sample=list()

    for i, section in enumerate(dataframe[instrument]):


        if i == 0:
            continue
        else:

            gap = starts[i]-ends[i-1]

            lost_samples = np.floor(gap.seconds/sample_rate.seconds).astype(int)

            samples.append(lost_samples)

            verdict = lost_samples<max_loss
            final_sample.append(verdict)

            if verbose:
                print("[{}]\t{}-{} \t are {} seconds apart ({} lost samples)".format(
                            "O" if verdict else "X",i-1, i, gap.seconds, lost_samples))

    combined_section_lengths = list()
    combined_section_lengths.append(len(dataframe[instrument][0]))

    group = list()
    group.append(0)

    for i in range(len(final_sample)):

        verdict = final_sample[i]

        group_index = len(group)-1
        group_number = group[group_index]

        if verdict:
            index = len(combined_section_lengths)-1
            combined_section_lengths[index]+=(len(dataframe[instrument][i+1]))
            group.append(group_number)
        else:
            combined_section_lengths.append(len(dataframe[instrument][i+1]))
            group.append(group_number+1)

    largest_combined_section = combined_section_lengths.index(max(combined_section_lengths))
    approved_section_start = group.index(largest_combined_section)
    approved_section_end   = len(group)-group[::-1].index(largest_combined_section)
    approved_section = dataframe[instrument][approved_section_start:approved_section_end]

    
    # recombine and pad whole dataframe
    depth = len(dataframe[instrument][0].T)
    NoneSlice = [None for i in range(depth)]

    combined_section = pandas.DataFrame()

    for i, section in enumerate(approved_section):

        start = section.index[i]

        if i > 0:
            gap = start-end

            lost_samples = np.floor(gap.seconds/sample_rate.seconds).astype(int)

            last_time = end;

            for j in range(lost_samples):
                last_time = last_time+datetime.timedelta(seconds=sample_rate.seconds)
                missing_data = pandas.DataFrame({last_time:NoneSlice}).T
                combined_section=combined_section.append(missing_data)

        end = section.index[len(section)-1]

        combined_section=combined_section.append(section)
    
    return combined_section



def display_data(dataframe, sample_rate=None, figsize=(20,10)):

    if sample_rate == None:
        try:
            sample_rate == dataframe['sample_rate'][0]
        except:
            print("Error: No sample rate found!")

    image = dataframe.T
    T = dataframe.index
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, interpolation='nearest', aspect='auto')

    if sample_rate == 0:
        return

    try:
        try:
            ticks = np.round((60*60)/sample_rate.seconds).astype(int)
        except:
            try:
                ticks = np.round((60*60)/sample_rate).astype(int)
            except:
                ticks = 1000
        ax.set_xticks(np.arange(0, len(T), ticks))
        ax.set_xticklabels(["{}h {}min {}s".format(Y.hour, Y.minute, Y.second) for Y in T[np.arange(0, len(T), ticks)]])
        
    except:
        print("No time index found: Using default index instead.")
        
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.xlim([0, len(T)])
    plt.show()

def combine_IMA(dataframe):
    new_values = list()
    new_times = list()

    for j, value in enumerate(dataframe.to_numpy()):

        if j%16 == 0:
            new_values.append( value )
            new_times.append( dataframe.index[j] )


    for i in range(1,16):
        index = 0
        for j, value in enumerate(dataframe.to_numpy()):

            if j%16-i == 0:
                new_values[index] += value
                index+=1

    new_values = np.stack( new_values, axis=0 )
    new_dataframe = pandas.DataFrame( new_values, index = new_times )

    return new_dataframe

def separate_IMA(dataframe):
    new_values = list()
    new_times = list()
    new_dataframes = list()

    for i in range(16):
        new_values.append(list())
        new_times.append(list())

    for i, value in enumerate(dataframe.to_numpy()):

        new_values[i%16].append( value )
        new_times[i%16].append( dataframe.index[i] )

    for i in range(16):

        new_values[i] = np.stack( new_values[i], axis=0 )
        new_dataframes.append( pandas.DataFrame( new_values[i], index = new_times[i] ) )

    return new_dataframes

def separate_ELS(dataframe):
 
    A = pandas.DataFrame(dataframe.T[0:31].T, dataframe.index)
    B = pandas.DataFrame(dataframe.T[32:63].T, dataframe.index)
    C = pandas.DataFrame(dataframe.T[64:95].T, dataframe.index)
    D = pandas.DataFrame(dataframe.T[96:127].T, dataframe.index)

    return [A,B,C,D]
     
     
def combine_ELS(dataframe):
    A = dataframe.T[0:31].T
    B = dataframe.T[32:63].T
    C = dataframe.T[64:95].T
    D = dataframe.T[96:127].T

    # improve by doing a denoising pixel check (by vote from each stream for each pixel to exclude anomalies)
    E = A.values + B.values + C.values + D.values
    E = pandas.DataFrame(E)
    E.index = A.index
    
    return E
