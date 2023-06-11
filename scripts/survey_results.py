#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:02:53 2023

@author: nicholasmagal

Aggregates scores from survey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_df(df):
    
    #get last col
    cols = df.columns
    
    #convert overall col to integers
    df.loc[:, cols[-1]] = pd.to_numeric(df[cols[-1]], errors = 'coerce')
    
    #drop na values
    df.dropna(axis = 'index', how='all')
    
    #for those that have a nonnumeric value for overall average the results
    df[cols[-1]] = df.apply(lambda row: (row[cols[0]] + row[cols[1]] + row[cols[2]])/3
                            if pd.isna(row[cols[-1]])
                            else row[cols[-1]],
                            axis = 1)
    
    return df

def get_results_one_question(results_df):
    
    #Triming off test input
    results_df = results_df.iloc[2:]
    
    #Seperating into different feature sets
    columns = list(results_df.columns)
    dsp = results_df[columns[:4]].copy()
    musicnn = results_df[columns[4:8]].copy()
    jukebox = results_df[columns[8:12]].copy()
    
    #Cleaning and averaging
    dsp = clean_df(dsp).mean()
    musicnn = clean_df(musicnn).mean()
    jukebox = clean_df(jukebox).mean()
    
    return dsp, musicnn, jukebox

def visualize_genre(dsp_perf, musicnn_perf, jukebox_perf):
    dsp_val = list(dsp_perf)
    musicnn_val = list(musicnn_perf)
    jukebox_val = list(jukebox_perf)
    
    width = .2
    labels = list(dsp_perf.index.tolist())
    X_axis = np.arange(len(labels))
    
    fig, ax = plt.subplots()
    fig.set_size_inches(15.5, 10.5)
    
    ax.bar(X_axis, dsp_val, width, label = 'DSP', color ='blue' )
    ax.bar(X_axis + width, musicnn_val, width, label = 'Musicnn', color = 'red' )
    ax.bar(X_axis + width*2, jukebox_val, width, label = 'Jukebox', color ='green')
    
    plt.xticks(X_axis, labels, fontsize = 15)
    plt.xlabel("Categories", fontsize = 15)
    plt.ylabel("Score", fontsize = 15)
    ax.set_title("Survey Results", fontsize= 20)
    ax.legend(prop={'size': 18})
    plt.show()


if __name__ == '__main__':
    path_0 = '/Users/nicholasmagal/Documents/Research/music_embeddings/survey/results/question_1.csv'
    path_1 = '/Users/nicholasmagal/Documents/Research/music_embeddings/survey/results/question_2.csv'
    path_2 = '/Users/nicholasmagal/Documents/Research/music_embeddings/survey/results/question_3.csv'
    
    question_0 = pd.read_csv(path_0)
    question_1 = pd.read_csv(path_1)
    question_2 = pd.read_csv(path_2)
    
    question_0_dsp, question_0_musicnn, question_0_jukebox = get_results_one_question(question_0)
    question_1_dsp, question_1_musicnn, question_1_jukebox = get_results_one_question(question_1)
    question_2_dsp, question_2_musicnn, question_2_jukebox = get_results_one_question(question_2)
    
    #getting final results
    dsp_perf = (question_0_dsp + question_1_dsp + question_2_dsp)/3
    musicnn_perf = (question_0_musicnn + question_1_musicnn + question_2_musicnn)/3
    jukebox_perf = (question_0_jukebox + question_1_jukebox + question_2_jukebox)/3
    
    #Graph results
    visualize_genre(dsp_perf, musicnn_perf, jukebox_perf)
    
    
    
    
    