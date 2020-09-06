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


def get_data(orbit, instrument, datapath, preview = False, debug=False, verbose=False):
    if verbose:
        print('orbit :{}'.format(orbit))

    # pull data
    frame, t = get_orbit_from(path=datapath, orbit=orbit)

    if preview:
        plot_data(frame,t)

    data = (frame, t)

    data_index, sample_rate = temporal_sections(data, instrument=instrument, sample_rate = 'common', max_skip = 5, verbose=verbose)

    dataframe = interpolate_sections(data, data_index, sample_rate = sample_rate, instrument=instrument, verbose=verbose)

    dataframe = regroup(dataframe, instrument=instrument, sample_rate = sample_rate,  max_loss = 100, verbose=verbose)
        
    return dataframe, sample_rate




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



def display_data(dataframe):
    image = dataframe.T
    T = dataframe.index
    fig, ax = plt.subplots(figsize=(20,10))
    im = ax.imshow(image, interpolation='nearest', aspect='auto')

    try:
        try:
            ticks = np.round((60*60)/sample_rate.seconds).astype(int)
        except:
            ticks = 1000
        ax.set_xticks(np.arange(0, len(T), ticks))
        ax.set_xticklabels(["{}h {}min {}s".format(Y.hour, Y.minute, Y.second) for Y in T[np.arange(0, len(T), ticks)]])
        
    except:
        print("No time index found: Using default index instead.")
        
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.xlim([0, len(T)])
    plt.show()
