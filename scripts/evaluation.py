#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:33:14 2023

@author: nicholasmagal

This script calculates the percent genre matching per top 5, 10, and 50 
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def top_n(cosine_sim_dict, n):
    
    percentages = []
    
    #iterating through songs and calcualting the percentage of genre matches
    for song, sim in cosine_sim_dict.items():
        
        #seed song genre
        seed_genre = song.split('.')[0]
        
        #iterating through top n songs and seeing how many match
        total_matches = 0
        for i in range(n):
            recommended_genre = sim[i][1].split('.')[0]
            
            if recommended_genre == seed_genre:
                total_matches+=1
        
        percentage = total_matches/n
        
        #update storage
        percentages.append(percentage)
    
    #finally return the average of percentages
    return np.mean(percentages)

def top_n_genre(cosine_sim_dict, n):
    
    #storing how each genre does 
    genre_percentages = {'blues': [],
                         'classical': [],
                         'country': [],
                         'disco': [],
                         'hiphop': [],
                         'jazz': [],
                         'metal':[],
                         'pop': [],
                         'reggae':[],
                         'rock':[]}
    
    #iterating through songs and calcualting the percentage of genre matches
    for song, sim in cosine_sim_dict.items():
        
        #seed song genre
        seed_genre = song.split('.')[0]
        
        #iterating through top n songs and seeing how many match
        total_matches = 0
        for i in range(n):
            recommended_genre = sim[i][1].split('.')[0]
            
            if recommended_genre == seed_genre:
                total_matches+=1
        
        percentage = total_matches/n
        
        #update storage
        genre_percentages[seed_genre].append(percentage)
        
    
    
    #averaging over genre
    for genre, scores in genre_percentages.items():
        genre_percentages[genre] = np.mean(scores)
        
    return genre_percentages

#given the performance across n_values, visualizes our data
def visualize_prato(performance, n_values):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(15.5, 10.5)
    colors = ['blue', 'red', 'g']
    
    for i, (feature, perf) in enumerate(performance.items()):
        ax.scatter(n_values, perf, c=colors[i], label = feature)
    
    ax.legend(prop={'size': 18})
    ax.set_xlabel('Top Tracks', fontsize = 20)
    ax.set_ylabel('Average Acc', fontsize = 20)
    ax.set_title("Recommendation System Performance", fontsize= 20)
    ax.grid(True)
    plt.show()
    plt.savefig('../output/rec_perf.png')

def visualize_genre(dsp_perf, musicnn_perf, jukebox_perf, n_value):
    dsp_val = list(dsp_perf.values())
    musicnn_val = list(musicnn_perf.values())
    jukebox_val = list(jukebox_perf.values())
    
    width = .2
    labels = list(dsp_perf.keys())
    X_axis = np.arange(len(labels))
    
    fig, ax = plt.subplots()
    fig.set_size_inches(15.5, 10.5)
    
    ax.bar(X_axis, dsp_val, width, label = 'DSP', color ='blue' )
    ax.bar(X_axis + width, musicnn_val, width, label = 'Musicnn', color = 'red' )
    ax.bar(X_axis + width*2, jukebox_val, width, label = 'Jukebox', color ='green')
    
    plt.xticks(X_axis, labels, fontsize = 15)
    plt.xlabel("Genres", fontsize = 15)
    plt.ylabel("Avg Acc", fontsize = 15)
    ax.set_title("Recommender Performance on Top "+ str(n_value)+ " Tracks by Genre", fontsize= 20)
    ax.legend(prop={'size': 18})
    plt.show()
    
    
    
    
    

if __name__ == '__main__':
    #loading cosine sim dicts per embeddings
    dsp_sim = np.load('../output/dsp_cosine.npy', allow_pickle=True).item()
    musicnn_sim = np.load('../output/muiscnn_cosine.npy', allow_pickle=True).item()
    jukebox_sim = np.load('../output/jukebox_cosine.npy', allow_pickle=True).item()
    
    
    #top number of tracks to evaluate
    n_values = np.arange(0, 105, 5)
    
    performance = defaultdict(list)
    for n in n_values:
        dsp_perf = top_n(dsp_sim, n)
        musicnn_perf = top_n(musicnn_sim, n)
        juke_box_perf = top_n(jukebox_sim, n)
        
        #saving results
        performance['dsp'].append(dsp_perf)
        performance['musicnn'].append(musicnn_perf)
        performance['jukebox'].append(juke_box_perf)
    
    visualize_prato(performance, n_values)
    
    
    #evaluating performance on granular level
    dsp_gran_perf = top_n_genre(dsp_sim, 10)
    musicnn_gran_perf = top_n_genre(musicnn_sim, 10)
    jukebox_gran_perf =top_n_genre(jukebox_sim, 10)
    
    visualize_genre(dsp_gran_perf, musicnn_gran_perf, jukebox_gran_perf, 10)
    
    #average performance results
    jukebox_vs_musicnn = (np.array(performance['jukebox']) - np.array(performance['musicnn']))
    jukebox_vs_musicnn = np.mean(jukebox_vs_musicnn[1:])
    print(jukebox_vs_musicnn)
    
    jukebox_vs_dsp = (np.array(performance['jukebox']) - np.array(performance['dsp']))
    jukebox_vs_dsp = np.mean(jukebox_vs_dsp[1:])
    print(jukebox_vs_dsp)
    
    
    
    
    
 