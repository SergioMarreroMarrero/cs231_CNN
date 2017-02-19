import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X , L = 2.0):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    L = float(L)
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dist_abs= np.abs(X[i] - self.X_train[j])
        dists[i, j] = np.sum(dist_abs ** L) ** (1/L)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X, L = 2):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    L = float(L)
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dist_abs= np.abs(X[i] - self.X_train)
      dists[i, :] = np.sum(dist_abs ** L , axis = 1) ** (1/L)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X , L = 2.0):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
   
        ## One general way. This way doesn't support the memory required    
        # testChange = X.reshape(X.shape[0] , 1 , X.shape[1]) # here we add the new dimension in order to force a broadcast where is required
        # squareResult = np.abs(testChange - self.X_train) ** L
        # dists = np.sum(squareResult, axis = 2) ** (1/L)
        ##

    L2_each_test_image = np.sum(X ** 2 , axis = 1)
    L2_each_train_image = np.sum(self.X_train ** 2 , axis = 1)
    dotTestTrain = np.dot(X , self.X_train.T) 

    ## Reshape
    rs_L2_each_test_image = L2_each_test_image.reshape(L2_each_test_image.shape[0] , 1)
    rs_L2_each_train_image = L2_each_train_image.reshape(1, L2_each_train_image.shape[0])

    ## Broadcast
    dists = (rs_L2_each_test_image + rs_L2_each_train_image - 2*dotTestTrain) ** (1/L)
    
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1 ):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      currentRow = dists[i]
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
      closest_y = self.y_train[indexOfClosestImages]



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

    return y_pred


def someClassesWin(theClasses , frecuencyOfClasses):
    iOfFocMax = np.argmax(frecuencyOfClasses)
    listOfTheMostRepeatedFrecuency = frecuencyOfClasses[iOfFocMax] == frecuencyOfClasses
    return theClasses[listOfTheMostRepeatedFrecuency]
