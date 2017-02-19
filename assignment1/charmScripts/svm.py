# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:04:09 2017

@author: sergio
"""

#SVM
import numpy as np
#%%
matrix = np.random.randn(27).reshape(3,3,3)
#%%
num = 5000
print np.arange(10)
num_selec_rand = 100
mask = np.random.choice(num, num_selec_rand, replace=False)
print mask

#%%
X_train = np.reshape(np.arange(25) , (5,5))
unos = np.ones((X_train.shape[0] , 1))
print unos
X_train = np.hstack([X_train , unos])
print X_train

#X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

#%%
dim = 1
num_classes = 3
print np.random.randn(dim, num_classes)
#%%

vector = np.arange(3)
print matrix, '\n' , vector
result =vector.dot(matrix)
print ''
print ''
print result
#%% 08/02/2017. Fully vectorized SVM
X = np.arange(20).reshape(4,5) # N X D; N = 4, D =5
W = np.arange(15).reshape(5,3) # D X C; D=5,C=3 clases
dW = np.zeros(W.shape) 
Q = np.zeros([W.shape[1],X.shape[0]])
Y = np.array([0,0,1,2])
Y_enum = np.array(list(enumerate(Y)))
Scores = (X).dot(W) 
#1)
Scores = Scores.T # Each column represent the scores for the sample Xi
#2)
Scores_yi = Scores[Y_enum[:,1],Y_enum[:,0]]
Margin = Scores - Scores_yi 
print Margin + 1
#%% SVM TOTAL


Margin_Hinge = (Margin > 0)*1
#Margin_Hinge = Margin_Hinge

MarginSVM = Margin * Margin_Hinge
SVM = np.sum(MarginSVM)
print SVM
#%% Grad
#3)
SumMargin_Hinge = -np.sum(Margin_Hinge , axis = 0)
print SumMargin_Hinge
#Q_index_yi = np.array([SumMargin_Hinge,Y_enum[:,1]])
Q[Y_enum[:,1],Y_enum[:,0]] = SumMargin_Hinge
Q += Margin_Hinge
#4) Hay que compinar los vectores X con estos coeficientes para producir el gradiente
Q_reshape = Q.reshape(-1,1)
repeatX = np.tile(X, (Q_reshape.shape[0]/X.shape[0],1))
grad_column = Q_reshape * repeatX
g = grad_column.reshape(W.shape[1],grad_column.shape[0]/W.shape[1],W.shape[0])
grad = np.sum(g, axis = 1)
##
##
##
##
##
## 8 de Febrero 2017
#%%
vector = np.arange(15)
sample = np.random.choice(vector,3,replace = True)
print sample
print type(range(vector.shape[0]))
vector = vector.reshape(3,5)
print vector
np.argmax(vector, axis = 1)

#%%
d={}
d[(2,3)] = (4,5)
d[(5,6)] = (6,7)
print d
for i in d:
    print d[i][0]
#%% SOFTMAX
vector2 = vector.copy()
vector2 = 0
print vector2, vector
#%%
np.zeros(4,)
#%%
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5
np.random.seed(1)
X = 10 * np.random.randn(num_inputs, input_size)
y = np.array([0, 1, 2, 2, 1])
t = np.arange(4)
X = np.vstack([X, t])
print X[0:X.shape[0]-1,:]
print y
#%% 16/ Febrero 20017
norma = np.linalg.norm(X, ord = 'fro')
print norma