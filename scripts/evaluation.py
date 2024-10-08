#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:33:14 2023

@author: nicholasmagal

This script evaluates our content based cosine similarity recommendation system
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import copy

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
        genre_percentages[genre] = [np.mean(scores)]
        
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
    ax.set_title("Recommender Performance by Genre", fontsize= 20)
    ax.legend(prop={'size': 18})
    plt.show()

def eval_prato(dsp_sim, musicnn_sim, jukebox_sim, n_values):
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
    
    jukebox_vs_musicnn = (np.array(performance['jukebox']) - np.array(performance['musicnn']))
    jukebox_vs_musicnn = np.mean(jukebox_vs_musicnn[1:])
    print("Percentage Better Jukebox vs Musicnn")
    print(jukebox_vs_musicnn)
    
    print("Percentage Better Jukebox vs DSP")
    jukebox_vs_dsp = (np.array(performance['jukebox']) - np.array(performance['dsp']))
    jukebox_vs_dsp = np.mean(jukebox_vs_dsp[1:])
    print(jukebox_vs_dsp)

def combine_list_of_dictionaries(dicts):
    
    #use the 0 index as storage
    for dict_index in range(1, len(dicts)):
        
        for genre, value in dicts[dict_index].items():
            
            #store all results in the first dictionary
            dicts[0][genre].append(value)
    
    #now aggregating results
    for genre, scores in dicts[0].items():
        dicts[0][genre] = np.mean(scores)[0]
        
    return dicts[0]

#finds the value of n that works the best
def find_best_n(dsp_perf, musicnn_perf, jukebox_perf, target):
    
    res = []
    for i in range(len(dsp_perf)):
        genres_won = []
        
        if target == 'jukebox':
            for genre, acc in jukebox_perf[i].items():
                if acc > musicnn_perf[i][genre] and acc > dsp_perf[i][genre]:
                    genres_won.append(genre)
                    
        elif target == 'musicnn':
            for genre, acc in musicnn_perf[i].items():
                if acc > jukebox_perf[i][genre] and acc > dsp_perf[i][genre]:
                    genres_won.append(genre)
        else:
            for genre, acc in dsp_perf[i].items():
                if acc > jukebox_perf[i][genre] and acc > musicnn_perf[i][genre]:
                    genres_won.append(genre)
        
        res.append((len(genres_won), genres_won, (i+1)*5))
        res = sorted(res, reverse = True)
    
    return res

#rounds numbers
def clean_dict(dict_to_clean):
    for key, values in dict_to_clean.items():
        dict_to_clean[key] = round(values[0],2)
    
    return dict_to_clean

def create_tabular_data_over_n(DSP, Musicnn, Jukebox, n_values):
    
    #iterating over each value of n
    for i in range(len(DSP)):
        
        dict_names = ['DSP', 'Musicnn', 'Jukebox']
        dict_list = [clean_dict(DSP[i]), clean_dict(Musicnn[i]), clean_dict(Jukebox[i])]
        
        #create df 
        table = pd.DataFrame.from_records(data = dict_list, index= ['DSP', 'Musicnn', 'Jukebox'])
        
        #Save 
        new_file_name = '../output/table_' + str(n_values[i]) + '.tex'
        with open(new_file_name, 'w') as tf:
            tf.write(table.to_latex())

#finds the best performance over genre
def create_matrix_and_find_max(feature_set_DSP, feature_set_musicnn, feature_set_jukebox):
    
    #clean and extract
    values = []
    for i in range(len(feature_set_DSP)):
        #clean
        feature_set_DSP[i] = clean_dict(feature_set_DSP[i])
        feature_set_musicnn[i] = clean_dict(feature_set_musicnn[i])
        feature_set_jukebox[i] = clean_dict(feature_set_jukebox[i])
        
        
        values.append(list(feature_set_DSP[i].values()))
        values.append(list(feature_set_musicnn[i].values()))
        values.append(list(feature_set_jukebox[i].values()))
    
    #combinate
    values = np.asarray(values)
    
    #get argmax and max values
    argmax = np.argmax(values, axis = 0) 
    max_ = np.max(values, axis = 0)

#finds the best performance over genre
def create_matrix_and_find_max_feature_set(feature_set):
    
    #clean and extract
    values = []
    for i in range(len(feature_set)):
        #clean
        feature_set[i] = clean_dict(feature_set[i])
        values.append(list(feature_set[i].values()))

    #combinate
    values = np.asarray(values)
    argmax = np.argmax(values, axis = 0) 
    

def eval_gran(dsp_sim, musicnn_sim, jukebox_sim, n_values):
    
    #Create storage to hold results of different levels of n
    dsp_gran_store = []
    musicnn_gran_store = []
    jukebox_gran_store = []
    
    for n in n_values:
    
        #loop over different values of n
        dsp_gran_perf = top_n_genre(dsp_sim, n)
        musicnn_gran_perf = top_n_genre(musicnn_sim, n)
        jukebox_gran_perf = top_n_genre(jukebox_sim, n)
        
        #store
        dsp_gran_store.append(dsp_gran_perf)
        musicnn_gran_store.append(musicnn_gran_perf)
        jukebox_gran_store.append(jukebox_gran_perf)
    
    #find best performing N value for each genre
    create_matrix_and_find_max(copy.deepcopy(dsp_gran_store), copy.deepcopy(musicnn_gran_store), copy.deepcopy(jukebox_gran_store))
    
    
    #Create tabular data showing how genre performs across different levels of N
    create_tabular_data_over_n(copy.deepcopy(dsp_gran_store), copy.deepcopy(musicnn_gran_store), copy.deepcopy(jukebox_gran_store),n_values )
    create_matrix_and_find_max_feature_set(copy.deepcopy(dsp_gran_store))
    create_matrix_and_find_max_feature_set(copy.deepcopy(musicnn_gran_store))
    create_matrix_and_find_max_feature_set(copy.deepcopy(jukebox_gran_store))
    
    #Find out granular information on where each feature set works best overall
    best_dsp = find_best_n(dsp_gran_store, musicnn_gran_store, jukebox_gran_store, 'dsp')
    best_musicnn = find_best_n(dsp_gran_store, musicnn_gran_store, jukebox_gran_store, 'musicnn')
    best_jukebox = find_best_n(dsp_gran_store, musicnn_gran_store, jukebox_gran_store, 'jukebox')
    
    #aggregate and visualize results from each feature set
    dsp_res = combine_list_of_dictionaries(dsp_gran_store)
    musicnn_res = combine_list_of_dictionaries(musicnn_gran_store)
    jukebox_res = combine_list_of_dictionaries(jukebox_gran_store)

    visualize_genre(dsp_res, musicnn_res, jukebox_res, 10)

    
if __name__ == '__main__':
    #loading cosine sim dicts per embeddings
    dsp_sim = np.load('../output/dsp_cosine.npy', allow_pickle=True).item()
    musicnn_sim = np.load('../output/muiscnn_cosine.npy', allow_pickle=True).item()
    jukebox_sim = np.load('../output/jukebox_cosine.npy', allow_pickle=True).item()
    
    #top number of tracks to evaluate
    base_n_values = np.arange(1,10)
    n_values = (np.arange(10, 105, 5))
    n_values = np.concatenate((base_n_values,n_values))
    
    #evaluate prato frontiner
    #eval_prato(dsp_sim, musicnn_sim, jukebox_sim, n_values)
    
    #evaluating performance on granular level
    eval_gran(dsp_sim, musicnn_sim, jukebox_sim, n_values)

 