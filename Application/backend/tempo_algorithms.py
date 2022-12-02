import os
import numpy as np
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import IPython.display as ipd
import math
import scipy
import itertools
import tensorflow as tf
import concurrent.futures
# computational resources needed to train a neural network on the full mel spectrogram was not worth it

# Standardize BPM function
BPM_MAX = 200
BPM_MIN = 60

def normalize_bpm(theor_bpm):
    new_bpm = theor_bpm
    while (new_bpm < BPM_MIN):
        new_bpm *= 2
    while (new_bpm > BPM_MAX):
        new_bpm /=2
    return new_bpm

# https://www.youtube.com/watch?v=Aht4letBAmA
# Lowpass Filtering
def lowpass_filter(data, samp_rate, cutoff):
    # allpass_output = np.zeros_like(data)
    # dn_1 = 0

    # for n in range(data.shape[0]):
    #     break_frequency = cutoff
    #     tan = np.tan(np.pi * break_frequency/samp_rate)
    #     a1 = (tan-1)/(tan+1)
    #     allpass_output[n] = a1 * data[n] + dn_1
    #     dn_1 = data[n] - a1 * allpass_output[n]

    # filtered = data + allpass_output
    b, a = scipy.signal.butter(5, cutoff, fs=samp_rate, btype='lowpass', analog=False)
    filtered = scipy.signal.lfilter(b, a, data)
    return filtered

def lpf_get_peaks(windows, window_thres, skip):
    peaks_no_skip = itertools.chain.from_iterable([[amp > window_thres[win_idx] for amp_idx, amp in enumerate(win)] for win_idx, win in enumerate(windows)])
    peaks = []

    skip_samples = 0
    is_skipping = False
    for i, el in enumerate(peaks_no_skip):
        if skip_samples == skip:
            is_skipping = False
            skip_samples = 0
        else:
            if is_skipping == False and el == True:
                peaks.append((i, el))
                is_skipping = True
            elif is_skipping == True:
                skip_samples += 1
    return peaks

def lpf_get_bpm_histogram(peaks, sr, num_neighbors=10):
    distance_histogram = {}

    for i, (sr_idx_ref, _) in enumerate(peaks):
        first = i + 1
        if first == len(peaks)-1:
            break

        last = i + num_neighbors + 1
        if last >= len(peaks):
            last = len(peaks)-1

        for (sr_idx_query, _) in peaks[first:last]:
            sr_dist = sr_idx_query - sr_idx_ref
            time_dist = int(60/(sr_dist/sr))

            time_dist = normalize_bpm(time_dist)

            if time_dist in distance_histogram:
                distance_histogram[time_dist] += 1
            else:
                distance_histogram[time_dist] = 1
    return distance_histogram
    

def lpf_algorithm(y_sr, cutoff=350, C=0.95, skip_ratio=0.25):
    print(f"LPF Processing File")

    y = y_sr[0]
    sr = y_sr[1]

    filtered = lowpass_filter(y, sr, cutoff)
    skip = int(skip_ratio * sr)

    filtered = np.abs(filtered)
    
    windows = np.split(filtered, np.arange(sr,len(filtered),sr))
    window_thres = [np.max(w) * C for w in windows] 

    peaks = lpf_get_peaks(windows, window_thres, skip)
    bpm_histogram = lpf_get_bpm_histogram(peaks, sr)
    pred_bpm = sorted(bpm_histogram.items(), key=lambda item: item[1] * -1)[0][0]

    return pred_bpm

# http://archive.gamedev.net/archive/reference/programming/features/beatdetection/index.html
# https://mziccard.me/2015/05/28/beats-detection-algorithms-1/
def sound_energy_algorithm(y_sr):
    print(f"SE Processing File")

    y = y_sr[0]
    sr = y_sr[1]

    samples_per_block = 1024
    C_mult = -0.0000015
    C_add = 1.5142857
    window_size = 21

    blocks = np.split(y, np.arange(samples_per_block, len(y), samples_per_block))
    energy_per_block = [np.sum(np.square(x)) for x in blocks]
    beat_counter = 0

    for i in range(window_size, len(energy_per_block)):
        cur_energy = energy_per_block[i] # energy of current block
        energy_window = energy_per_block[i-window_size:i-1] # energies of 43 previous blocks
        avg_energy_window = np.average(energy_window) # average energy of the 43 previous blocks
        variance_window = np.average(np.square(avg_energy_window - energy_window)) # variance of the 43 previous blocks

        C = (C_mult * variance_window + C_add) * avg_energy_window
        if cur_energy > C:
            beat_counter += 1

    theor_bpm = beat_counter/((len(y)/sr)/60)
    pred_bpm = normalize_bpm(theor_bpm)
    return pred_bpm

def note_onset_algorithm(y_sr, hop_length=200, n_fft=2048):
    print("NO Processing File")

    y = y_sr[0]
    sr = y_sr[1]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=2048)
    frames = range(len(onset_env))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    tmp = np.log1p(onset_env)
    r = librosa.autocorrelate(tmp)

    bpm_axis = (60/t)[::-1]
    corr_axis = r[::-1]

    bpm_window = np.argwhere((bpm_axis > 60) & (bpm_axis < 200))
    begin_frame = bpm_window[0]
    end_frame = bpm_window[-1]

    bpm_axis = bpm_axis[begin_frame[0]:end_frame[0]+1]
    corr_axis = corr_axis[begin_frame[0]:end_frame[0]+1]

    max_idx = np.argmax(corr_axis)
    pred_bpm = bpm_axis[max_idx]

    return pred_bpm
