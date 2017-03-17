# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:59:04 2017

@author: sergio
"""
#%%
import numpy as np
#%%
#The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
#examples
num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape)
weight_size = output_dim * np.prod(input_shape)

x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs,*input_shape)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

#%%

N = x.shape[0]
vect_lenght= np.prod(x[0].shape)
print N, vect_lenght
#%%
X = np.array( x.reshape(N, vect_lenght))
print X.shape
#%%

out = X.dot(w) + b
print out

#%%
dist_norm = np.random.normal(0,2,[2,2])
print dist_norm
#%%
params = {}
N, D, H, C = 3, 5, 50, 7
params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
print params['W1'].shape

X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
print X.shape

#%%
for i in np.arange(5):
    print 'gola' + str(i)
    
#%%
    
config = {}
config.setdefault('learning_rate', 1e-2)
config.setdefault('momentum', 0.9)    

v = config.get('velocity', np.zeros_like(w))