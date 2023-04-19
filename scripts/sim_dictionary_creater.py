# -*- coding: utf-8 -*-
import numpy as np
import collections
import pandas as pd
from scipy import spatial
from numpy.linalg import norm
from collections import defaultdict
from dsp_feature_extractor import get_files
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import multiprocessing

def cosine_similairty_adjusted(files):
    '''
    Parameters
    ----------
    files : list of embeddings
        Creates dictionary for holding cosine sim for each track. Keys are song names and values are = [[sim, candiate_track_name, candiate_track_full_path], ...]

    Returns
    -------
    consine_sim_dict : Dictionary

    '''
    #
    consine_sim_dict = dict()
    
    #For each track find the consine similairty
    for i, track in (enumerate(files)):
        track_sim = []
        
        for j, canadiate_track in enumerate(files):
            
            #skip if comparing to self
            if i != j:
                #calcualting cosine sim 
                track_a = np.load(track)
                track_b = np.load(canadiate_track)
                
                mean_ab = sum(sum(track_a, track_b)) / (len(track_a) + len(track_b))
                
                cosine_sim = 1 - spatial.distance.cosine(track_a-mean_ab, track_b-mean_ab)
                track_sim.append([cosine_sim, canadiate_track.split('/')[-1], canadiate_track])
        
        #finally sort and append to our dictionary
        import pdb
        pdb.set_trace()
        track_sim =sorted(track_sim, key=lambda x: x[0], reverse = True)
        consine_sim_dict[track.split('/')[-1]] = track_sim.copy()

        
    return consine_sim_dict

def cosine_similairty(files):
    '''
    Parameters
    ----------
    files : list of embeddings
        Creates dictionary for holding cosine sim for each track. Keys are song names and values are = [[sim, candiate_track_name, candiate_track_full_path], ...]

    Returns
    -------
    consine_sim_dict : Dictionary

    '''
    # grabbing scaler for standardization
    scaler = create_scaler(files)
    
    consine_sim_dict = dict()
    
    #For each track find the consine similairty
    for i, track in (enumerate(files)):
        track_sim = []
        track_a = scaler.transform(np.load(track).reshape(1,-1))
        
        for j, canadiate_track in enumerate(files):
            
            #skip if comparing to self
            if i != j:
                #calcualting cosine sim 
                track_b = scaler.transform(np.load(canadiate_track).reshape(1,-1))
                
                cosine_sim = 1 - spatial.distance.cosine(track_a, track_b)
                track_sim.append([cosine_sim, canadiate_track.split('/')[-1], canadiate_track])
        
        #finally sort and append to our dictionary
        track_sim = sorted(track_sim, key=lambda x: x[0], reverse = True)
        consine_sim_dict[track.split('/')[-1]] = track_sim.copy()
        
    return consine_sim_dict

def create_scaler(files):
    #pool data together to calc mean and variance
    data = []
    for file in files:
        data.append(np.load(file))
    
    #fit scaler
    scaler = StandardScaler()
    scaler.fit(data)
    
    return scaler

def run_extraction(path):
    files = get_files(path, '.npy')
    print(len(files))
    cosine_sim = cosine_similairty(files)
    return cosine_sim
    
    

if __name__ == '__main__':
    
    save_path = '../output/'
    #path for embeddings
    dsp_path = '../embeddings/dsp/' 
    musicnn_path = '../embeddings/musicnn/' 
    jukebox_path = '../embeddings/Jukebox/' 
    
    paths = [dsp_path, musicnn_path] #jukebox_path
    
    
    #capturing cosine sim
    with multiprocessing.Pool() as pool:    
        cosine_dicts = pool.starmap(run_extraction, zip(paths))
    
    np.save(save_path+'dsp_cosine', cosine_dicts[0])
    np.save(save_path+'muiscnn_cosine', cosine_dicts[1])
    np.save(save_path+'jukebox_cosine', cosine_dicts[2])   
        
    
        
    
    
    '''
    #list of embeddings
    dsp_files = get_files(dsp_path, '.npy')
    musicnn_files = get_files(musicnn_path, '.npy')
    jukebox_files = get_files(jukebox_path, '.npy')
    
    #dictionaries containing cosine similairty of files
    dsp_cosine = cosine_similairty(dsp_files)
    musicnn_cosine = cosine_similairty(musicnn_files)
    jukebox_cosine = cosine_similairty(jukebox_files)
    
    np.save(save_path+'dsp_cosine', dsp_cosine)
    np.save(save_path+'muiscnn_cosine', musicnn_cosine)
    np.save(save_path+'jukebox_cosine', jukebox_cosine)   
    '''
    
    