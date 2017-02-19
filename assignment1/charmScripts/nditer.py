# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:49:43 2017

@author: sergio
"""

import numpy as np
#%% Forma de lectura
matrix = np.arange(6,12,1).reshape(2,3)
print matrix,'\n\n'
matrixit = np.nditer(matrix)
for it in matrixit:
    print it,
print '\n\n',matrix.T,'\n\n'
matrixit = np.nditer(matrix.T) # A nivel python las guarda en el mismo orden.
for it in matrixit:
    print it,

#%% Como si las leyera en C o fortran. Date cuenta que con C se traspone previamente. 
matrixit = np.nditer(matrix.T , order = 'C') 
for it in matrixit:
    print it,
print "\n\n"
matrixit = np.nditer(matrix, order = 'F')
for it in matrixit:
    print it,

#%% Read and write. Para que haga efecto it[...]
matrix = np.arange(6,12,1).reshape(2,3)
print matrix
matrixit = np.nditer(matrix, op_flags=['readwrite'])
print '\n\n\n\n'
for it in matrixit:
    it[...] = 2 * it
print matrix


#%% No solo tenemos acceso a los elementos, a los índices también.
a = np.arange(6,12,1).reshape(2,3)
print a
it = np.nditer(a, flags=['multi_index'])
while not it.finished:
     print "%s <%s>" % (it[0], it.multi_index)
     it.iternext()
     
#%%
     
a=np.array([2, 4, 6, 8 ,10])
a
i=np.array([0, 1, 0])
i
j=np.array([0, 1, 2])

a[i]


a=np.array([[0.99, 0.2, 0.3], [0.5, 0.6, 0.1]])
a[i,j]
#%%

x = range(16)
change = np.arange(100,109,1).reshape(3,3)
x = np.reshape(x,(4,4))
sub_x = x[0:3,0:3]
x[0:3,0:3] = change
print x
