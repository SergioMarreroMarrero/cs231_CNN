import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  pscores = scores.copy()
  L = np.zeros(N,)
  for image in range(N):
    logC = np.max(scores[image,:])
    pscores[image,:] -= logC
    exp_pscore = np.exp(pscores[image,:])
    sum_exp_score = np.sum(exp_pscore)
    pscores[image,:] = exp_pscore/sum_exp_score
    L[image] = -np.log(pscores[image,y[image]])

  # Add regularization to the loss.
  loss = np.sum(L)/N + 0.5 * reg * np.sum(W * W)

  
  # Gradient
  dW = np.zeros(W.shape)
  dpscores = pscores.copy()
  xIm = np.zeros([1,D])
  dp = np.zeros([1,C])
  for image in range(N):
    
    dpscores[image,y[image]] = pscores[image,y[image]] - 1
    xIm[0,:] = X[image,:]
    dp[0,:] = dpscores[image,:]
    dW += np.dot(xIm.T,dp)
    

  dW /= N
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW
  #return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  all_samples = range(y.shape[0])
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores = X.dot(W)
  pscores = scores.copy()
  L = np.zeros(N,)
  logC = np.max(scores , axis = 1).reshape(scores.shape[0],1)
  pscores -= logC
  exp_pscore = np.exp(pscores)
  sum_exp_score = np.sum(exp_pscore , axis = 1).reshape(scores.shape[0],1)
  pscores = exp_pscore/sum_exp_score
  Lyimage =  pscores[all_samples,y]
  L = -np.log(Lyimage)
  loss = np.sum(L)/N #+ 0.5*reg*np.sum(W*W) # Add regularization to the loss.

  # Gradient
  dW = np.zeros(W.shape)
  dpscores = pscores.copy()
#  xIm = np.zeros([1,D])
#  dp = np.zeros([1,C])
    
  dpscores[all_samples,y] = pscores[all_samples,y] - 1
#  xIm[0,:] = X[image,:]
#  dp[0,:] = dpscores[image,:]
  dW = np.dot(X.T,dpscores)
    

  dW /= N
  #dW += reg*W    
  
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return  loss, dW

