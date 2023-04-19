#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:05:29 2023

@author: nicholasmagal

This song picks 5 random songs for auditory evaluation
"""

from random import *
from dsp_feature_extractor import get_files
import numpy as np

#Get our dataset
song_directory = '/Users/nicholasmagal/Documents/Research/datasets/GTZAN'
songs = get_files(song_directory, '.wav')

#Choose 5 random indexes and store song names
random_songs_index = []
for i in range(5):
    random_songs_index.append(randint(1,1000))

seed_songs = []
for song_index in random_songs_index:
    seed_songs.append(songs[song_index].split('/')[-1])
    



#load cosine sim dictionary
dsp_sim = np.load('../output/dsp_cosine.npy', allow_pickle=True).item()
musicnn_sim = np.load('../output/muiscnn_cosine.npy', allow_pickle=True).item()
jukebox_sim = np.load('../output/jukebox_cosine.npy', allow_pickle=True).item()

#Now print out top tracks per seed song
for seed_song in seed_songs:
    print("Seed Song: ", seed_song)
    key = seed_song.split('.wav')[0] + '.npy'
    
    print("DSP Top Song: ", dsp_sim[key][0][1])
    print("Musicnn Top Song: ", musicnn_sim[key][0][1])
    print("Jukebox Top Song: ", jukebox_sim[key][0][1])
    print()
