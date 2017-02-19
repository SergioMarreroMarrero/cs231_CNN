#%%
import os
import numpy as np



#%%
#%%
def initNANmatrix(a,b):
    dists = np.empty([a, b])
    dists[:] = np.NAN
    return dists

def someClassesWin(theClasses , frecuencyOfClasses):
    iOfFocMax = np.argmax(frecuencyOfClasses)
    lOfMostRepFreq = frecuencyOfClasses[iOfFocMax] == frecuencyOfClasses
    return theClasses[lOfMostRepFreq]


# Distance functions
def distanceTwoLoops(num_test , num_train , matrix_test, matrix_train , L):
    dists = initNANmatrix(num_test,num_train)
    for i in xrange(num_test):
        for j in xrange(num_train):
            dist_abs= np.abs(matrix_test[i,:] - matrix_train[j,:])
            dists[i, j] = np.sum(dist_abs ** L) ** (1/L)
    return dists


def distanceOneLoops(num_test , num_train , matrix_test, matrix_train , L):
    dists = initNANmatrix(num_test , num_train)
    for i in xrange(num_test):
        dist_abs = np.abs(matrix_test[i] - matrix_train)
        dists[i,:] = np.sum(dist_abs ** L , axis = 1) ** (1/L)
    return dists

def distanceNoLoops(num_test , num_train , matrix_test, matrix_train , L):
    dists = initNANmatrix(num_test , num_train)
    dist_abs = np.abs(matrix_test - matrix_train)
    
    return dists

#%%
###############################################################
## 1) Calculate the distance
#%% Create the training/test sets
matrix_train = np.random.randint(0,10,size = (50,5))
matrix_test = np.random.randint(0,10,size = (5,5))

num_train = matrix_train.shape[0]
num_test = matrix_test.shape[0]

y_train = np.random.randint(0,10,size = (num_train,1))
y_test = np.random.randint(0,10,size = (num_test,1))

#print "matrix_test: \n\n", matrix_test, "\n\n", "matrix_train: \n\n",matrix_train
#%% Create distances matrix
#print "number of training samples is: ", num_train , "\n\n", "number of testing samples is: ", num_test , "\n\n"

#print "Shape for the dists matrix is: ", dists.shape
## xrange: Useful type for safe RAM space
#print type(xrange(num_test))
#%% Pruebas sobre restas en numpy

#print matrix_test
#print matrix_train[0,:]
#print matrix_train[0,:] 

#%%
L = float(2)
dists = distanceTwoLoops(num_test, num_train, matrix_test, matrix_train, L)
dists2 = distanceOneLoops(num_test, num_train, matrix_test, matrix_train, L)

if not np.isnan(dists).any():
    print dists[0:3,0:10]
else:
    print "bug in dists"
    
if not np.isnan(dists2).any():
    print dists2[0:3,0:10]
else:
    print "bug in dists2"



#print dists
#print "\n\n\n\n\n"
#print dists2
#%%
        ########################


## 2) Calculate the k-nearest



k = 2
num_test = dists.shape[0]
y_pred = np.zeros(num_test)
for i in xrange(num_test):
    # A list of length k storing the labels of the k nearest neighbors to
    # the ith test point.
    currentRow  = dists[i]
    closest_y = []
    #########################################################################
    # TODO:                                                                 #
    # Use the distance matrix to find the k nearest neighbors of the ith    #
    # testing point, and use self.y_train to find the labels of these       #
    # neighbors. Store these labels in closest_y.                           #
    # Hint: Look up the function numpy.argsort.                             #
    #########################################################################

    #1) Calculate the k-minimum values in i row ---> Select the column where they are
    indexOfCurrentRowSorted = np.argsort(currentRow)
    indexOfClosestImages= indexOfCurrentRowSorted[:k]

    #2) Use these indexes to look for the labels in y_train. Store it in closest_y
    closest_y = np.take(y_train , indexOfClosestImages)

    #########################################################################
    # TODO:                                                                 #
    # Now that you have found the labels of the k nearest neighbors, you    #
    # need to find the most common label in the list closest_y of labels.   #
    # Store this label in y_pred[i]. Break ties by choosing the smaller     #
    # label.                                                                #
    #########################################################################
    #3) Calculate the most common label in the vector closest_y. In case of tie
    # select the smaller. Save this in y_predi[i]

    theClasses , frequencyOfTheClasses = np.unique(closest_y , return_counts=True)
    y_pred[i] = np.amin(someClassesWin(theClasses, frequencyOfTheClasses))

    #########################################################################
    #                           END OF YOUR CODE                            #
    #########################################################################

print y_pred

