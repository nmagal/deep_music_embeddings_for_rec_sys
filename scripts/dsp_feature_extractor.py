#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:33:19 2023

@author: nicholasmagal

Script used to extract embeddings from: "TZANETAKIS AND COOK: MUSICAL GENRE CLASSIFICATION OF AUDIO SIGNALS"
"""
import numpy as np
import librosa
from librosa import feature
import os
from tqdm import tqdm
from scipy import ndimage

def get_files(path, audio_file_type):
    files_storage = []
    for (root, dirs, files) in os.walk(path):
        #append on full path and make sure it is a wav file 
        file_paths = [os.path.join(root,file_name) for file_name in files if audio_file_type in file_name]
        files_storage.extend(file_paths)
    
    return files_storage

def calc_spectral_centriod_mean_var(track, sample_rate):
    '''        
    Calculates the center of mass of a spectrum.  This correlates to the 
    impression of the brightness of a sound. 
    Parameters
    ----------
    track : Loaded power spectogram
    sampling_rate : the rate at which the track is read in as

    Returns
    -------
    spec_cent_mean : mean of the spectral centriod
    spec_cent_var : Var of spectral centriod

    '''
    
    #Obtaining spec centriod
    spec_cent = feature.spectral_centroid(y = track, sr = sample_rate)
    
    #Finding mean and var
    spec_cent_mean = np.mean(spec_cent)
    spec_cent_var = np.var(spec_cent)
    
    return spec_cent_mean, spec_cent_var

def spectral_rolloff_mean_var(track, sampling_rate):
    '''
    From librosa: 
    Parameters The roll-off frequency is defined for each frame as 
    the center frequency for a spectrogram bin such that at 
    least roll_percent (0.85 by default) of the energy of the spectrum in 
    this frame is contained in this bin and the bins below. 
    ----------
    track : Loaded power spectogram
    sampling_rate : the rate at which the track is read in as
    
    Returns
    -------
    mean_spec_roll_off : mean of spectral roll off
    var_spec_roll_off : variance of spectral roll off
    
    '''
    
    #Obtaining spec rolloff
    spec_roll_off = feature.spectral_rolloff(y=track, sr = sampling_rate)
    
    #mean/var
    mean_spec_roll_off = np.mean(spec_roll_off)
    var_spec_roll_off = np.var(spec_roll_off)
    
    return mean_spec_roll_off, var_spec_roll_off
    
def zero_crossing_rate_mean_var(track, sampling_rate):
    '''
    From wiki - the zero-crossing rate (ZCR) is the rate at which a signal changes from positive to zero to negative or from 
    negative to zero to positive.

    Parameters
    ----------
    track : Loaded power spectogram
    sampling_rate : the rate at which the track is read in as

    Returns
    -------
    mean_zero_crossing : float
        mean of zero crossing

    var_zero_crossing : float
        var of zero crossing

    '''
    
    #getting zero crossing rate
    zero_crossing = feature.zero_crossing_rate(track)
    
    #mean/var
    mean_zero_crossing = np.mean(zero_crossing)
    var_zero_crossing = np.var(zero_crossing)
    
    return mean_zero_crossing, var_zero_crossing

def mfcc_mean_var(y, sr):
    '''
    Extract 20 mfcc and return mean and var of both 
    
    '''
    mfcc = feature.mfcc(y=y, sr=sr, n_mfcc  = 12)
    
    dx = ndimage.sobel(mfcc, 0)
    mean_mfcc = np.mean(mfcc, axis = 1)
    var_mfcc = np.var(mfcc, axis = 1)

    
    mfcc_mean_var = np.concatenate((mean_mfcc,var_mfcc), axis = 0)
    return mfcc_mean_var

def get_full_feature_vector(song_path):
    
    #loading song
    y, sr = librosa.load(song_path)
    
    #get the embeddings of all the features 
    embedding_storage = []
    
    embedding_storage.append(mfcc_mean_var(y, sr))
    embedding_storage.append(zero_crossing_rate_mean_var(y, sr))
    embedding_storage.append(spectral_rolloff_mean_var(y, sr))
    embedding_storage.append(calc_spectral_centriod_mean_var(y, sr))
    
    return np.concatenate(embedding_storage, axis = 0)
    
#Actually extracting the features and saving them to a new directory
if __name__ == "__main__":
    dataset_dir = '/Users/nicholasmagal/Documents/Research/datasets/GTZAN/genres_original'
    embedding_dir = '../embeddings/dsp/'
    
    #list of files to get
    files = get_files(dataset_dir, '.wav')
    
    #loop through and get embeddings for all songs and save them
    for song_file in tqdm(files):
        try:
            embedding = get_full_feature_vector(song_file)
            
            song_name = song_file.split('/')[-1][:-4]
            np.save(embedding_dir+song_name, embedding)
        except:
            print("error loading ", song_file)
    

    
    
#TO UPDATE: Using more mfccs, as well as first order information 
    
    
    
    
