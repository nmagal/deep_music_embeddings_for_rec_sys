#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:42:03 2022

@author: nicholasmagal

Using multiple techniques, anayzle relationships between genres 
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from dsp_feature_extractor import get_files
from dim_reduction import return_data_labels

#Class containing helper functions for clustering algorithms
class ClusterTools():
    
    #Aggregate our dataset into genres, in order to do hierarchical clustering later
    def prepare_dataset_hierarchical_clustering(self, dataset):
        
        consolidated_genres=[]
        #Loop through each genre group, and consolidate 
        for genre, genre_df in dataset.groupby('label'):
            
            #consolidate by using average over rows
            genre_df = genre_df.mean(axis=0, numeric_only = True)
            genre_df['label'] = genre
            consolidated_genres.append(genre_df)
        
        #Crating a dataframe
        consolidated_df = pd.DataFrame(consolidated_genres)
        
        #Switching label to be first col
        #cols = list(consolidated_df.columns)
        #cols = [cols[-1]] + cols[:-1] 
        #consolidated_df= consolidated_df[cols]
        
        return consolidated_df
    
    def hierarchical_clustering(self, dataset, genre_labels, title):
        linkage_data = linkage(dataset, method='ward', metric='euclidean', optimal_ordering = True)
        fig = plt.figure(figsize=(5, 5))
        dendrogram(linkage_data, leaf_rotation=90, leaf_font_size = 8, labels = genre_labels)
        fig.suptitle(title)
        plt.show()
        #plt.savefig('../output/' + title + "dendogram.png")
    
if __name__ == '__main__':
    
    title = "Jukebox Embeddings"
    embedding_dir = '../embeddings/Jukebox/' 
    
    #path for embeddings
    files = get_files(embedding_dir, '.npy')
    
    #loaded embeddings and labels
    embeddings, labels = return_data_labels(files)
    
    #creating df from embeddings
    data = pd.DataFrame(data = embeddings)
    data['label'] = labels
    
    c_t = ClusterTools()
    
    #Combining all genre tracks into one
    grouped_df = c_t.prepare_dataset_hierarchical_clustering(data)
    
    standardized_music_features = StandardScaler().fit_transform(grouped_df.iloc[:, :-1])
    c_t.hierarchical_clustering(standardized_music_features, list(grouped_df['label']), title)




    

    