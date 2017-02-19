import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # matrix multiplication. The result will be shaped: [X.shape[0],W.shape[1]]
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue 
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]
        dW[:,y[i]] -= X[i,:] 
    



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg * W 

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # X: N X D; W: D X C
  y_enum = np.array(list(enumerate(y)))
  scores = X.dot(W).T 
  sclass = scores[y_enum[:,1] , y_enum[:,0]]
  margin = scores - sclass + 1
  # Remove values from  margin
  margin[y_enum[:,1] , y_enum[:,0]] = 0 # delete the "ones" from the class 
  marginb = (margin > 0) * 1 
  margin = margin * marginb # delete the values over zero: max(0,margin)
  #
  loss = np.sum(margin)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Matrix q:
  # the q matrix store how the images must combinate in order to produce the gradient
  # q: C X N 
  # ej.
  # q = [[ 1,-2, 0, 0],  dw1 =  1*x1 - 2*x2 + 0*x3 + 0*x4
  #      [-2, 1, 0, 1],  dw2 = -2*x1 + 1*x2 + 0*x3 + 1*x4
  #      [ 1, 1, 0,-1]]  dw3 =  1*x1 + 1*x2 + 0*x3 - 1*x4
  #- W: A numpy array of shape (D, C) containing weights.
  #- X: A numpy array of shape (N, D) containing a minibatch of data.

  #
  q = np.zeros([W.shape[1] , X.shape[0]]) # Init the matrix q
  sum_yi = np.sum(marginb , axis = 0) # Calculate the related coefficient of the class classifier
  q[y_enum[:,1] , y_enum[:,0]] = -sum_yi # Put this coefficiente in the matrix q
  q += marginb # Complete the matrix with the related coefficientes of the j classifier
  #
  #
  # Perform operations using q with X
  dW = (q.dot(X)).T # q: C X N, X: N x D, q*X = C x D  dw = C x D 
  # q_col = q.reshape(-1,1)
  # X_tile = np.tile(X , (W.shape[1],1))
  # s1grad = q_col * X_tile
  # stocks = s1grad.shape[0]/W.shape[1]
  # s2grad = s1grad.reshape(W.shape[1], stocks , W.shape[0])
  #dW =  np.sum(s2grad , axis = 1).T
  dW /= num_train
  dW += reg * W
  


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #return loss, dW
  return loss, dW