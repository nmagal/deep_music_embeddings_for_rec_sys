#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:52:58 2023

@author: nicholasmagal

Script used to extract embeddings from the musicnn library
"""

import numpy as np
import skimage
from musicnn.extractor import extractor
from dsp_feature_extractor import get_files
from tqdm import tqdm
import tensorflow as tf


if __name__ == '__main__':
    
    #Data and embedding paths
    dataset_dir = '/Users/nicholasmagal/Documents/Research/datasets/GTZAN/genres_original'
    embedding_dir = '../embeddings/musicnn/'
    
    #list of files to get
    files = get_files(dataset_dir, '.wav')
    
    #loop through and get embeddings for all songs
    for song_file in files:
        try:

            taggram, tags, features = extractor(song_file, model='MSD_musicnn', extract_features=True)
            
            #getting the same features as did the Codified Paper did
            mean_pool_features = features['mean_pool']
            max_pool_features = features['max_pool']
            #max_pool_features = max_pool_features.flatten()
            
            #Concate and performing mean pooling to bring size down
            features = np.concatenate((mean_pool_features,max_pool_features), axis = 1)
            features = skimage.measure.block_reduce(features, (2,2), func = np.mean)
            features = features.flatten()

            song_name = embedding_dir + song_file.split('/')[-1][:-4]
            #Save embeddings
            np.save(song_name, features)
            
        except:
            print("error loading ", song_file)
    

    
    
    
