# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:28:14 2020

@author: mustafa
"""

import numpy as np
import pandas as pd
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from sklearn.model_selection import train_test_split
from numpy import savetxt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.models import Model

# load the data 
folder='C:/CS230/dendritic'
#expressions_url='https://mousescexpression.s3.amazonaws.com/dendritic_cell.h5'


genes= pd.read_table(folder+"/genes.txt",index_col=1,header=None,names=['gene', 'id'],delimiter=' ')
genes.head()



#load the matrix
expression_df = pd.read_hdf(folder+"/exprMatrix.h5",index_col=0)

#clean the data

expression_df.index.rename('cell_id',inplace=1)
expression_df.shape

gene_names=[genes['gene'].ix[id] for id in expression_df.columns.values]
expression_df.columns=gene_names

expression_df.head()

expression_df=expression_df.loc[:, (expression_df != expression_df.iloc[0]).any()]  # remove constant columns
expression_df.shape

expression_df=expression_df[(expression_df.T != 0).any()] # remove rows of zeros
expression_df.shape


gene_count=expression_df.astype(bool).sum(axis=0)
cells=expression_df.count()
expression_df=expression_df[gene_count[gene_count>cells/10].index]
expression_df.shape


expression_df.head()



expression_df=(100*expression_df.transpose() / expression_df.sum(1)).round(2).transpose()

"""

# sanity check

expression_df


gene_sum=expression_df.sum()
top_genes=gene_sum.sort_values().tail(10)
top_genes


expression_df_top=expression_df[top_genes.index]
expression_df_top.plot(figsize=(15,10))

corr=expression_df_top.corr()


import seaborn as sns
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


     
        
gene_a='tmsb4x'
gene_b='rpl41'

        
np.log2(expression_df_top+1).plot.scatter(gene_a,gene_b)





d=expression_df_top[[gene_a,gene_b]]
d=d[d[gene_a]*d[gene_b]>1]


x=np.log2(d+1)[gene_a].values
y=np.log2(d+1)[gene_b].values   
"""


def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    return heatmap.T   

def generate_data(expression_df,pairs,y_values,s=16):
    
    
    #expression_df = expression_df[~expression_df.index.duplicated()]
    
    expression_df = expression_df.loc[:,~expression_df.columns.duplicated()]
    X=[]
    y_out=[]
    for pair,k in zip(pairs,y_values):
        
        #if len(X)>40000:
        #    break
        
        gene_a=pair[0]  
        gene_b=pair[1]
        if gene_a in expression_df and gene_b in expression_df :
            d=expression_df[[gene_a,gene_b]]
            d=d[d[gene_a]*d[gene_b]>0.05]
        
        
            x=np.log2(d+1)[gene_a].values
            y=np.log2(d+1)[gene_b].values  
            
            if len(x)!=0:
                if len(x.shape)!=1:
                    
                    continue
                else:
                    
                      
                    img = myplot(x, y, s,bins=150)
                    

                    X.append(img)
                    y_out.append(k)
                    
                    #plt.imsave('X_'+gene_a+'_'+gene_b+'_.jpg',img ,origin='lower', cmap=cm.jet)
                    #savetxt('X_'+gene_a+'_'+gene_b+'_.txt', img, delimiter=',')
                    
                    #savetxt('y_'+gene_a+'_'+gene_b+'_.txt', np.array([k]), delimiter=',')
                    
            else:
                
                    img = myplot(x, y, s,bins=150)
                
                    #print(img.shape)
                    X.append(img)
                    y_out.append(k)
                    #plt.imsave('X_'+gene_a+'_'+gene_b+'_.jpg',img ,origin='lower', cmap=cm.jet)
                    #savetxt('X_'+gene_a+'_'+gene_b+'_.txt', img, delimiter=',')
                    
                    #savetxt('y_'+gene_a+'_'+gene_b+'_.txt', np.array([k]), delimiter=',')                
                
                    #X.append(img)
                    #y_out.append(k)                

        
    
    return np.array(X),np.array(y_out)




def load_point_cloud_from_hd(path):
     entiregraph=[]
     y=[]
     with open(path) as f:
         for line in f:
             numbers_str = line.split()
             numbers_float = [str(x) for x in numbers_str]  #map(float,numbers_str) works too
             val=float(numbers_float[-1])
             if val!=2:          
                 y.append(val)
                 entiregraph.append(numbers_float[:-1])
                 
     return np.array(entiregraph),y


  



pairs,y_=load_point_cloud_from_hd("C:/CS230/dendritic_gene_pairs_200.txt")

X,y=generate_data(expression_df,pairs,y_,s=16)



import pickle

with open('C:/CS230/X_Data_all.pkl','wb') as f:
    pickle.dump(X, f)

with open('C:/CS230/y_Data_all.pkl','wb') as f:
    pickle.dump(y, f)






"""

fig, axs = plt.subplots(2, 2,figsize=(15,15))



sigmas = [0, 8, 16]

for ax, s in zip(axs.flatten(), sigmas):
    ax.set(xlabel='gene A expression', ylabel='gene B expression')
    if s == 0:
        ax.plot(x, y, 'k.', markersize=5)
        ax.set_title("Scatter plot")
    else:
        img, extent = myplot(x, y, s)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        
        plt.imsave('image_new.jpg',img ,origin='lower', cmap=cm.jet)
        
        #ax.imsave(img, extent=extent, origin='lower', cmap=cm.jet)
        
        ax.set_title("Smoothing with  $\sigma$ = %d" % s)
"""     