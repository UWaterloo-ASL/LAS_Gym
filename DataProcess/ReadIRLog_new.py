# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 18:35:52 2018

@author: Lin Daiwei
"""

import numpy as np
import re
import math
import csv
import warnings

from scipy.signal import butter, lfilter


def ReadTXT(file_name, start_time, s_freq=10):
    """
    Read IR sensor logs
    
    """
    #file_name = "IR_samples_ROM_July_04_2018_at_12-59-55" + ".txt"
    num_sensor = 24 # number of sensors
    num_group = int(num_sensor/2)
    freq = s_freq # Hz
        
    all_readings = np.zeros((num_sensor,1))
    all_times = np.array([0])
    
    last_time = start_time
    t_offset = convert_time(last_time)
    t_gap = 1/freq
    
    one_sec_readings = np.zeros((num_sensor,freq))
    one_sec_index = np.zeros(num_group,dtype=int)
    
    file = open(file_name,'r')
    line_count = 0
    for line in file:
        # skip emtpy lines
        if line == '\n':
            continue
        
        if line_count % 100000 == 0:
            print("Process " + str(line_count/1000) + "k lines")
        
        # Get time, readings and cluster ids
        m = re.search(r'[0-9]+-[0-9]+-[0-9]+', line)
        time = m.group(0)

        m = re.search(r'[0-9]+, [0-9]+',line)
        m_num = re.findall(r'[0-9]+',m.group(0))
        value = [int(m_num[0]), int(m_num[1])]

        m = re.search(r'BPC [0-9]+', line)
        m_id = re.search(r'[0-9]+',m.group(0))
        group_id = int(m_id.group(0)) - 1
        
        # Process data by 1-second-long chunks
        if time != last_time:
            
            one_sec_readings = fill_missing_data(one_sec_readings, one_sec_index)
            all_readings = np.append(all_readings, one_sec_readings, axis = 1)
            time_in_sec = convert_time(time) - t_offset
            for i in range(freq):
                all_times = np.append(all_times, time_in_sec + i*t_gap)
            # Reset
            one_sec_readings = np.zeros((num_sensor,freq))
            one_sec_index = np.zeros(num_group,dtype=int)
            
            last_time = time
        else:
            if one_sec_index[group_id] > 9:
#                print("groupID overflow at " + time)
                continue
            one_sec_readings[group_id*2, one_sec_index[group_id]] = value[0]
            one_sec_readings[group_id*2+1, one_sec_index[group_id]] = value[1]
            one_sec_index[group_id] += 1
        
        line_count += 1            
      

    file.close()
    
    # Get data collection date from file name
    m = re.search(r'[a-zA-Z]+_[0-9]+_[0-9]+',file_name)
    collect_date = m.group(0)
    
    return all_times[1:], all_readings[:,1:], collect_date

def convert_time(time_string):
    """
    Input: time string 'xx-xx-xx'
    """
    [h, m, s] = time_string.split('-')
    h = int(h)
    m = int(m)
    s = int(s)

    return s+m*60+h*3600

def fill_missing_data(data, fill_index):
    """
    Input: 2D array
    Output: 2D array
    """
    n_row, n_col = data.shape
    
    for i in range(n_row):
        g_idx = int(i/2)
        if fill_index[g_idx] < n_col-1 :
            diff = n_col-1 - fill_index[g_idx]
            last_data = data[i, fill_index[g_idx]]
            for _ in range(diff):
                data[i,fill_index[g_idx]+1] = last_data
                fill_index[g_idx] += 1
    
    return np.array(data)

def cal_disc_occupancy(data):
    """
    Rough calculation of occupancy 
    
    Input: row -- sensor
           col -- time
    
    """
    row, col = data.shape
    
    # Get average reading of each sensors
    means = np.zeros(row)
    for r in range(row):
        means[r] = np.mean(data[r,:])
    # Manually set a threshold 
    threshold = 50
    occupancy = np.zeros(col,dtype = int)
    
    for r in range(row):
        occupancy = occupancy + (data[r,:] - (means[r] + threshold) > 0)
    
    return occupancy

def cal_cont_occupancy(data, isFilter=False, s_freq=10, high_cut = 2, order = 10):
    """
    Calculation of occupancy (continuous)
    
    Input: row -- sensor
           col -- time
    
    """
    row, col = data.shape
    
    # Get average reading of each sensors
    means = np.zeros(row)
    for r in range(row):
        means[r] = np.mean(data[r,:])
    
    occupancy = np.zeros(col)
    
    for r in range(row):
        val_range = np.max(data[r,:]) - np.min(data[r,:])
        if val_range > 100:
            occupancy = (data[r,:] - means[r])/val_range + occupancy
    
    # filter the data through low-pass filter
    # order of the filter = # of data used for every filtering calculation
    if isFilter:
        highcut = high_cut # cutoff frequency
        
        b, a = butter_lowpass(highcut, s_freq, order)
        occupancy = lfilter(b,a,occupancy)
    
    return occupancy


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs # Nyquist frequency = 0.5*sample Frequency
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a

def WriteCSV(data, file_date):
    timestamps, occ = data
    file_name = file_date + "_IR_processed_occupancy.csv"
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for t in range(timestamps.shape[0]):
            writer.writerow([timestamps[t],occ[t]])
            
    csv_file.close()

def differencing(data):
    """
    Perform differencing tranform to remove trend in data.
    
    Input: each row represents a time-series data of one sensor
    """
    row, col = data.shape
    diff_data = np.zeros((row,col-1)) # because it is a difference
    for r in range(0, row):
        for t in range(0,col-1):
            diff_data[r,t] = data[r,t+1] - data[r,t]
    return diff_data

if __name__ == '__main__':
    sample_frequency = 10
	file_name = "IR_samples_ROM_July_04_2018_at_12-59-55" + ".txt"
    time, readings, date = ReadTXT(file_name=file_name, start_time='13-00-13', s_freq=sample_frequency)
    occupancy = cal_cont_occupancy(readings,isFilter=True,s_freq=sample_frequency)
    
    WriteCSV((time, occupancy), date)