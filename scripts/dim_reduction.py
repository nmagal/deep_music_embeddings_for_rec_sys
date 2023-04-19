#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:24:39 2022

@author: nicholasmagal

Given audio embeddings, this will use PCA and t-sne
to give us a 2D representation of our dataset
"""

import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from dsp_feature_extractor import get_files

#visualize 2d data
def two_d_graph(df, labels, col_0, col_1, title):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    #ax.set_xlabel('Principal Component 0', fontsize = 15)
    #ax.set_ylabel('Principal Component 1', fontsize = 15)
    ax.set_title(title, fontsize = 20)
    colors = list(np.random.choice(range(256), size=10))
    for target in (labels):
        indicesToKeep = df['label'] == target
        ax.scatter(df.loc[indicesToKeep, col_0]
                   , df.loc[indicesToKeep, col_1]
                   , s = 20)
    ax.legend(labels)
    ax.grid()
    
    plt.savefig('../output/' + title + ".png")
#visualize 3d data
def three_d_graph(df, labels, title):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlabel('Principal Component 0', fontsize = 15)
    ax.set_ylabel('Principal Component 1', fontsize = 15)
    ax.set_zlabel('Principal Component 2', fontsize = 15)
    ax.set_title(title, fontsize = 20)
    colors = list(np.random.choice(range(256), size=10))
    for target in (labels):
        indicesToKeep = df['label'] == target
        ax.scatter(df.loc[indicesToKeep, 'pc_0']
                   , df.loc[indicesToKeep, 'pc_1'],
                   df.loc[indicesToKeep, 'pc_2']
                   , s = 10)
    ax.legend(labels)
    ax.grid()

#performs 2d or 3d pca depending on mode 
def pca(mode, music_features, genre, title):
    
    if mode == '2d':
        #Standardizing
        standardized_music_features = StandardScaler().fit_transform(music_features)
        
        #performing 2d PCA
        pca = PCA(n_components=2)
        pca_music_features = pca.fit_transform(standardized_music_features)
        print("Total explained variance 2d: ", sum(pca.explained_variance_ratio_))
        
        #Creating our 2d PCA df
        principal_comps = pd.DataFrame(data = pca_music_features, columns = ['pc_0', 'pc_1'])
        principal_comps['label'] = genre
        
        #Visualizing our 2d PCA
        labels_set = set(genre)
        two_d_graph(principal_comps, labels_set, 'pc_0', 'pc_1', title)
        
        return (principal_comps, labels_set)
    
    #perform 3d PCA otherwise
    else:
        #Standardizing
        standardized_music_features = StandardScaler().fit_transform(music_features)
        
        #performing 3d PCA
        pca = PCA(n_components=3)
        pca_music_features = pca.fit_transform(standardized_music_features)
        print("Total explained variance 3d: ", sum(pca.explained_variance_ratio_))
        
        #Creating our 3d PCA df
        principal_comps = pd.DataFrame(data = pca_music_features, columns = ['pc_0', 'pc_1', 'pc_2'])
        principal_comps['label'] = genre
        
        #Visualizing our 3d PCA
        labels_set = set(genre)
        three_d_graph(principal_comps, labels_set, title)


#performs 2d or 3d tsne
def tsne(mode, music_features, genre, title):
    
    #Parameters to use with tsne 
    tsne_parameters = {'init':'random', 'perplexity' : 500, 'n_iter_without_progress' : 400,
                      'n_jobs' : 4, 'n_iter_without_progress': 400, 'n_iter' : 1000, 'learning_rate' : 1000  }
    
    #Used later for visualization
    labels_set = set(genre)
    
    #Chooses either 2d or 3d tsne
    #TODO - Implement 3d tsne 
    if mode == '2d':
        t_sne_2d = TSNE(n_components=2,
                       init=tsne_parameters['init'], perplexity =tsne_parameters['perplexity'],
                       n_jobs = tsne_parameters['n_jobs'], n_iter_without_progress=tsne_parameters['n_iter_without_progress'],
                       n_iter = tsne_parameters['n_iter'], learning_rate = tsne_parameters['learning_rate'])
        
        music_features = StandardScaler().fit_transform(music_features)
        t_sne_2d = t_sne_2d.fit_transform(music_features)
        
        #Creating our 2d t-sne df
        t_sne_2d = pd.DataFrame(data = t_sne_2d, columns = ['tsne_0', 'tsne_1'])
        t_sne_2d['label'] = genre
        
        #visualizng t-sne
        two_d_graph(t_sne_2d, labels_set, 'tsne_0', 'tsne_1', title )

#given a list of files containing embeddings, returns a list of embeddings with corresponding labels
def return_data_labels(files):
    embeddings = []
    labels = []
    
    for file_name in files:
        embedding = np.load(file_name)
        genre = file_name.split('/')[-1].split('.')[0]
        
        #updating storage
        embeddings.append(embedding)
        labels.append(genre)
        
    return embeddings, labels
    

if __name__ == '__main__':
    
    #name for plot
    title = "DSP Features"
    embedding_dir = '../embeddings/dsp/' 
    
    #path for embeddings
    files = get_files(embedding_dir, '.npy')
    
    #loaded embeddings and labels
    embeddings, labels = return_data_labels(files)
    
    
    #pass in 
    #pca('3d', embeddings, labels, title)
    tsne('2d',embeddings, labels, title)
    
    
   