import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    H = W1.shape[1]
    C = W2.shape[1]



    scores = np.zeros((N, C))
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    # Compute the forward pass until the scores
    axon1 = np.dot(X,W1) + b1 # fully connected the inputs with with W1 add up the bias

    activation1 = np.maximum(0, axon1) # activation function RELU

    axon2 = np.dot(activation1,W2) + b2 # fully connected the inputs of two layers with W2

    scores = axon2.copy()


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################

    ## SOFTMAX ##
    #1) Transform the scores matrix in a probabilities space

    #1.1) Create the matrix which stores the probabilities
    pscores = scores.copy()

    #1.2) Add the constant to avoid the convergence problems
    logC = np.max(scores , axis = 1).reshape(scores.shape[0],1)
    pscores -= logC

    #1.3) exp() all values
    exp_pscore = np.exp(pscores)

    # 1.4) Calculate the sum of each row 
    sum_exp_score = np.sum(exp_pscore , axis = 1).reshape(scores.shape[0],1)

    #1.5) Calculate the probabilities space of each sample
    pscores = exp_pscore/sum_exp_score

    #2) Calculate the Loss
    
    #2.1) all_samples only is a index list of all samples
    all_samples = range(y.shape[0]) 

    #2.2) Extract the pscores of each class. This will be the image loss' cause, 
    # the more near to 1, the less the loss will be.
    pscore_y_image =  pscores[all_samples,y]

    #2.3) Loss of each image 
    L = -np.log(pscore_y_image)
    #2.4) Sum all loss (each image contribuite with its loss) and divide between N samples.
    loss = np.sum(L)/N

    #3) Add the regularization
    loss +=  0.5*reg*np.linalg.norm(W1, ord = 'fro') **2 + 0.5*reg*np.linalg.norm(W2, ord = 'fro') **2




    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    # backpropagation
    # Second Layer: W2, b2
    # 1) Calculate the gradient respect of the scores (axon2). This is: softmax derivate

    # 1.1 Calculate the softmax derivate
    # 1.1.1 Create the dLdaxon2 matrix. 
    dLdaxon2 = pscores.copy() # axon = scores

    # 1.1.2 For any samples substract 1 in the correct class
    dLdaxon2[all_samples,y] -= 1

    # 2) Gradiente respect of b2
    dLdb2 = np.zeros_like(b2)
    dLdb2 = np.sum(dLdaxon2, axis = 0)/N

    # 3) Gradient respect W2
    dLdW2 = np.zeros_like(W2)
    dLdW2 = np.dot(np.transpose(activation1), dLdaxon2)/N 
    # 3.2) Add regularization to dlW2
    dLdW2 += reg*W2



    # First layer W1, b1
    
    # 1) Calculate the gradient of the L respect of the output of activation1
    dLdactivation1 = np.zeros_like(activation1)
    dLdactivation1 = np.dot(dLdaxon2, W2.T) 

    # 2) The RELU gate route the gradient follow its argument: max(0,backpro_gradient)
    dLdaxon1 = dLdactivation1.copy()
    dLdaxon1[activation1 <= 0] = 0  # Select the maximum

    # 2) The gradient respect of b1
    dLdb1 = np.zeros_like(b1)
    dLdb1 = np.sum(dLdaxon1, axis = 0)/N

    # 3) The gradient respect of W1
    dLdW1 = np.zeros_like(W1)
    dLdW1 = np.dot(X.T, dLdaxon1)/N

    # Add regularization
    dLdW1 += reg*W1 

    # Put the results in the dictionary
    grads = {'W1': dLdW1, 'b1': dLdb1,'W2': dLdW2, 'b2': dLdb2}

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      index = np.random.choice(X.shape[0],batch_size)
      X_batch = X[index, :] 
      y_batch = y[index]

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.         

      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #                    #
      #########################################################################
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################



      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['b2'] -= learning_rate * grads['b2']
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = np.zeros((X.shape[0],))

    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1']
    b2 = self.params['b2'] 

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################

    # Calculate the proobabilities
    ##################################################################
    # We use the architecture of NNa and stop just before calculate the Loss
    # Compute the forward pass until the scores
    axon1 = np.dot(X,W1) + b1 # fully connected the inputs with with W1 add up the bias

    activation1 = np.maximum(0, axon1) # activation function RELU

    axon2 = np.dot(activation1,W2) + b2 # fully connected the inputs of two layers with W2

    scores = axon2.copy()

 ## SOFTMAX ##
    #1) Transform the scores matrix in a probabilities space

    #1.1) Create the matrix which stores the probabilities
    pscores = scores.copy()

    #1.2) Add the constant to avoid the convergence problems
    logC = np.max(scores , axis = 1).reshape(scores.shape[0],1)
    pscores -= logC

    #1.3) exp() all values
    exp_pscore = np.exp(pscores)

    # 1.4) Calculate the sum of each row 
    sum_exp_score = np.sum(exp_pscore , axis = 1).reshape(scores.shape[0],1)

    #1.5) Calculate the probabilities space of each sample
    pscores = exp_pscore/sum_exp_score
    #####################################################33
 
    # 2.1) Select the class predicted
    y_pred = np.argmax(pscores, axis = 1)

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


