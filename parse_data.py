# -*- coding: utf-8 -*-
"""
Sierra Lear
Mustafa Hajij
Mahdi Moqri

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

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
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

from sklearn.model_selection import train_test_split
import pickle

import numpy as np

from keras.layers import InputLayer

from keras.layers import Conv2D

from keras.layers import Input, Dense



def myplot(x, y, s, bins=150):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    return heatmap.T

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
             else:
                 y.append(1.0)
                 entiregraph.append(numbers_float[:-1])

                 
     return np.array(entiregraph),y
 
    
def generate_data(expression_df,pairs,y_values,s=16,threshold=0.05,size=None):

    expression_df = expression_df.loc[:,~expression_df.columns.duplicated()]
    X=[]
    y_out=[]
    for pair,k in zip(pairs,y_values):
        
        if size!=None:
            
            if len(X)>size:
                break
        
        gene_a=pair[0]  
        gene_b=pair[1]
        
        if gene_a in expression_df and gene_b in expression_df : # check if the genes are in the table before plugging them in
            d=expression_df[[gene_a,gene_b]]
            d=d[d[gene_a]+d[gene_b]>threshold] # here we use the threshold 
        
        
            x=np.log2(d+1)[gene_a].values
            y=np.log2(d+1)[gene_b].values  


            
            if len(x)!=0:
                if len(x.shape)!=1:
                    
                    continue
                else:
                    
                      
                    img = myplot(x, y, s,bins=150) # create the image
                    
                    X.append(img)
                    y_out.append(k)
                    
                    
            else:
                
                    img = myplot(x, y, s,bins=150)  # create the image
             

                    X.append(img)
                    y_out.append(k)
              

        
    
    return np.array(X),np.array(y_out)    

folder='C:/CS230/dendritic'
expressions_url='https://mousescexpression.s3.amazonaws.com/dendritic_cell.h5'
genes_url='https://raw.githubusercontent.com/moqri/deep_cell/master/dendritic/genes.txt'
labels_url='https://raw.githubusercontent.com/moqri/deep_cell/master/dendritic_gene_pairs_200.txt'


genes= pd.read_table(folder+"/genes.txt",index_col=1,header=None,names=['gene', 'id'],delimiter=' ')
genes.head()

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

    
labels= pd.read_table(folder+"/dendritic_gene_pairs_200.txt",index_col=False,header=None,names=['gene1', 'gene2','value'],delimiter='\t')
labels.head()    


pairs,y_=load_point_cloud_from_hd(folder+"dendritic_gene_pairs_200.txt")

X,y=generate_data(expression_df,pairs,y_,s=16,threshold=0.00)

#X_=X[:20000]
#y_=y[:20000]

with open(folder+"X_Data_all.pkl",'wb') as f:
    pickle.dump(X_, f)

with open(folder+"y_Data_all.pkl",'wb') as f:
    pickle.dump(y_, f)



