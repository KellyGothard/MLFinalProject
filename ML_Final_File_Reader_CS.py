#!/usr/bin/env python
# coding: utf-8

# In[39]:


##### Always import all needed libraries in the first cell
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
#from planar_utils import plot_decision_boundary, sigmoid
#from dataset import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import pandas as pd
from string import punctuation
import sqlite3
import os
import sys
np.random.seed(1) # this sets the seed so that the runs are consistent
import math

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


notbanned = pd.read_csv("notbanned.csv", delimiter=',')

pd.set_option('display.max_columns', 5)  # Set to actually print out the full columns, change if needed
# print(notbanned.head(n=10))

banned = pd.read_csv("banned.csv", delimiter=',')

pd.set_option('display.max_columns', 5)  # Set to actually print out the full columns, change if needed
# print(banned.head(n=10))

banned_comments = []
for line in banned['body']:
    if "I am a bot" in str(line):
        #print("SKIPPING THIS LINE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        pass
    else:
        banned_comments.append(str(line))

not_banned_comments = []
for line in notbanned['body']:
    if "I am a bot" in str(line):
        #print("SKIPPING THIS LINE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        pass
    else:
        not_banned_comments.append(str(line))

exclude = set(punctuation)  # Keep a set of "bad" characters.
# then make a string of all the words in Obama and Romney tweets without punctuation
banned_words_no_punct = " ".join(["".join(str(char) for char in text if char not in exclude) for text in banned_comments])
not_banned_words_no_punct = " ".join(["".join(str(char) for char in text if char not in exclude) for text in not_banned_comments])



# In[41]:


banned_dict = {}

for word in banned_words_no_punct.split(" "):
    if word in banned_dict:
        banned_dict[word] += 1
    else:
        banned_dict[word] = 1
        
print(dict(list(banned_dict.items())[0: 5]))


# In[42]:


not_banned_dict = {}

for word in not_banned_words_no_punct.split(" "):
    if word in not_banned_dict:
        not_banned_dict[word] += 1
    else:
        not_banned_dict[word] = 1
        
print(dict(list(not_banned_dict.items())[0: 5]))


# In[43]:


banned_counts = {}
for word in list(banned_dict.keys()):
    if word in not_banned_dict:
        banned_counts[word] = banned_dict[word] - not_banned_dict[word]
    else:
        banned_counts[word] = banned_dict[word]

print(dict(list(banned_counts.items())[0: 5]))


# In[44]:


not_banned_counts = {}
for word in list(not_banned_dict.keys()):
    if word in banned_dict:
        not_banned_counts[word] = not_banned_dict[word] - banned_dict[word]
    else:
        not_banned_counts[word] = not_banned_dict[word]
        
print(dict(list(not_banned_counts.items())[0: 5]))


# In[45]:


def getCounts(str):
    """
    get how many times the words show up in the banned dictionary - how many times it shows up in the not banned dictionary
    """
    weight = 0
    for word in str.split(" "):
        if word in banned_counts:
            weight += banned_counts[word]
        if word in not_banned_counts:
            weight -= not_banned_counts[word]
    return weight


# In[46]:


def getBannedCount(str):
    """
    get how many times the words show up in the banned dictionary 
    """
    weight = 0
    for word in str.split(" "):
        if word in banned_counts:
            weight += banned_counts[word]
    return weight


# In[47]:


def getNotBannedCount(str):
    """
    get how many times the words show up in the not banned dictionary 
    """
    weight = 0
    for word in str.split(" "):
        if word in not_banned_counts:
            weight += not_banned_counts[word]
    return weight


# In[93]:


feature = []
#labels = np.array(int)
labels = []
feature1 = []
feature2 = []
feature3 = []
count = 0
for comment in banned_comments:
    if count < 10000:
        feature1.append(float(getBannedCount(comment)))
        feature2.append(float(getNotBannedCount(comment)))
        feature3.append(float(getCounts(comment)))
        labels.append(float(1))
        count += 1
    else:
        break
        #labels = np.append(labels, 1)
        
count = 0
for comment in not_banned_comments:
    if count < 10000:
        feature1.append(float(getBannedCount(comment)))
        feature2.append(float(getNotBannedCount(comment)))
        feature3.append(float(getCounts(comment)))
        labels.append(float(0))
        #labels = np.append(labels, 0)
        count += 1
    else:
        break

feature.append(feature1)
feature.append(feature2)
feature.append(feature3)
labels1 = []
labels1.append(labels)
print('done')


# In[94]:


features = np.array([np.array(xi) for xi in feature])
type(features)


# In[95]:


label = np.array([np.array(xi) for xi in labels1])
type(label)


# In[96]:


shape_X = features.shape
shape_Y = label.shape
m = 2 * 400

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have %d training sample!' % (m))


# In[97]:

def sigmoid(X):

    y = math.exp(X)
    y = y / (math.exp(X) + 1)
    return y


def layerSizes(X, Y):
    """
    X -- input dataset of shape 
    Y -- labels of shape
    """
    input_layer_size= X.shape[0]
    hidden_layer_size= 4
    output_layer_size= Y.shape[0]
    # hardcode as 1 bc we have to 
    
    """
    Returns:
    input_layer_size -- the size of the input layer
    hidden_layer_size -- the size of the hidden layer
    output_layer_size -- the size of the output layer
    """
    
    return (input_layer_size, hidden_layer_size, output_layer_size)


# In[98]:


# just run this code to test layerSizes function 
# input_layer_size, hidden_layer_size, output_layer_size=layerSizes(features,label)
# print("The size of the input layer is: = " + str(input_layer_size))
# print("The size of the hidden layer is: = " + str(hidden_layer_size))
# print("The size of the output layer is: = " + str(output_layer_size))


# In[99]:


def initialize_parameters(input_size, hidden_size, output_size):
    """
    input_size-- size of the input layer
    hidden_size -- size of the hidden layer
    output_size-- size of the output layer
    
    
    """
    
    np.random.seed(2)  # you can pick any seed in this case
    
    Weight1 = np.random.randn(hidden_size,input_size) * 0.01
    Weight2 = np.random.randn(output_size,hidden_size) * 0.01
    bias1 = np.zeros(shape=(hidden_size, 1))
    bias2 = np.zeros(shape=(output_size, 1))
    
    parameters = {"Weight1": Weight1,
                  "bias1": bias1,
                  "Weight2": Weight2,
                  "bias2": bias2}
    
    
    """
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape 
                    b1 -- bias vector of shape 
                    W2 -- weight matrix of shape 
                    b2 -- bias vector of shape
    
    """
    
    return parameters


# In[100]:


# parameters = initialize_parameters(input_layer_size, hidden_layer_size, output_layer_size)
# print("Weight1 = " + str(parameters["Weight1"]))
# print("Weight2 = " + str(parameters["Weight2"]))
# print("bias1 = " + str(parameters["bias1"]))
# print("bias2 = " + str(parameters["bias2"]))


# In[101]:


def forward_prop(X, parameters):
    """
    X -- input data of size
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    
    """
    # Retrieve each parameter from the dictionary "parameters"
    Weight1 = parameters['Weight1']
    bias1 = parameters['bias1']
    Weight2 = parameters['Weight2']
    bias2 = parameters['bias2']
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(Weight1,X) + bias1
    A1 = np.tanh(Z1)
    Z2 = np.dot(Weight2,A1) + bias2
    A2 = sigmoid(Z2)
    
    #Values needed in the backpropagation are stored in cache. Later, it will be given to back propagation.
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    
    """
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    return A2, cache


# In[102]:


# #test function

# np.random.seed(1) 
# X_assess = np.random.randn(2, 3)

# parameters = {'Weight1': np.array([[-0.00416758, -0.00056267],
#         [-0.02136196,  0.01640271],
#         [-0.01793436, -0.00841747],
#         [ 0.00502881, -0.01245288]]),
#      'Weight2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
#      'bias1': np.array([[ 0.],
#         [ 0.],
#         [ 0.],
#         [ 0.]]),
#      'bias2': np.array([[ 0.]])}


# A2, cache = forward_prop(X_assess, parameters)

# # Note: we use the mean here just to make sure that your output matches ours. 
# print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

#the output of this print should be like below:

#-0.0004997557777419902 -0.000496963353231779 0.00043818745095914653 0.500109546852431


# In[103]:


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation
    Y -- "true" labels vector of shape 
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    """
    
    m = Y.shape[1]  # number of example # changed to 0 bc it is messed up

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y[0]) + np.multiply((1 - Y[0]), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    ### Remember that, if you want to use different cross-entropy loss, you need to change logprobs and cost accordingly
    
    cost = float(np.squeeze(cost))   
    
    return cost


# In[104]:


# #test function
# np.random.seed(1) 
# Y_assess = np.random.randn(1, 3)
# parameters = {'W1': np.array([[-0.00416758, -0.00056267],
#         [-0.02136196,  0.01640271],
#         [-0.01793436, -0.00841747],
#         [ 0.00502881, -0.01245288]]),
#      'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
#      'b1': np.array([[ 0.],
#         [ 0.],
#         [ 0.],
#         [ 0.]]),
#      'b2': np.array([[ 0.]])}

# a2 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]))



# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
# #the cost will be around 0.69


# In[105]:


def backward_propagation(parameters, cache, X, Y):
    """
    
    parameters -- dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data 
    Y -- "true" labels vector 
    
    
    """
    m = X.shape[1]
    
    # Copy W1 and W2 from the dictionary "parameters"
    Weight1 = parameters['Weight1']
    Weight2 = parameters['Weight2']
    
        
    # Copy A1 and A2 from dictionary "cache".
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    #  calculate dW1, db1, dW2, db2. 
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(Weight2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    gradient = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return gradient


# In[106]:


# #test function
# np.random.seed(1) 
# X_assess = np.random.randn(2, 3)
# Y_assess = np.random.randn(1, 3)
# parameters = {'Weight1': np.array([[-0.00416758, -0.00056267],
#         [-0.02136196,  0.01640271],
#         [-0.01793436, -0.00841747],
#         [ 0.00502881, -0.01245288]]),
#      'Weight2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
#      'bias1': np.array([[ 0.],
#         [ 0.],
#         [ 0.],
#         [ 0.]]),
#      'bias2': np.array([[ 0.]])}

# cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
#          [-0.05225116,  0.02725659, -0.02646251],
#          [-0.02009721,  0.0036869 ,  0.02883756],
#          [ 0.02152675, -0.01385234,  0.02599885]]),
#   'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
#   'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
#          [-0.05229879,  0.02726335, -0.02646869],
#          [-0.02009991,  0.00368692,  0.02884556],
#          [ 0.02153007, -0.01385322,  0.02600471]]),
#   'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}

# grads = backward_propagation(parameters, cache, X_assess, Y_assess) # call the back propagation here with appropriate parameters
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

#The output should be 


#db1 = [[-0.00069728]
 #[-0.00060606]
 #[ 0.000364  ]
# [ 0.00151207]]
#dW2 = [[ 0.00363613  0.03153604  0.01162914 -0.01318316]]
#db2 = [[0.06589489]]


# In[107]:


def update_parameters(parameters, grads, learning_rate = 0.5):
    """
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    """
    # Copy the following parameter from the dictionary "parameters"
    Weight1 = parameters['Weight1']
    Weight2 = parameters['Weight2']
    bias1 = parameters['bias1']
    bias2 = parameters['bias2']
    
    # Copy each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # Update rule for each parameter
    Weight1 = Weight1 - learning_rate * dW1
    Weight2 = Weight2 - learning_rate * dW2
    bias1 = bias1 - learning_rate * db1
    bias2 = bias2 - learning_rate * db2
    
    parameters = {"Weight1": Weight1,
                  "Weight2": Weight2,
                  "bias1": bias1,
                  "bias2": bias2}
    
    """
    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    #print(parameters)
    return parameters


# In[108]:


# #test function
# np.random.seed(1) 
# parameters = {'Weight1': np.array([[-0.00615039,  0.0169021 ],
#         [-0.02311792,  0.03137121],
#         [-0.0169217 , -0.01752545],
#         [ 0.00935436, -0.05018221]]),
#  'Weight2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
#  'bias1': np.array([[ -8.97523455e-07],
#         [  8.15562092e-06],
#         [  6.04810633e-07],
#         [ -2.54560700e-06]]),
#  'bias2': np.array([[  9.14954378e-05]])}

# grads = {'dW1': np.array([[ 0.00023322, -0.00205423],
#         [ 0.00082222, -0.00700776],
#         [-0.00031831,  0.0028636 ],
#         [-0.00092857,  0.00809933]]),
#  'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,
#           -2.55715317e-03]]),
#  'db1': np.array([[  1.05570087e-07],
#         [ -3.81814487e-06],
#         [ -1.90155145e-07],
#         [  5.46467802e-07]]),
#  'db2': np.array([[ -1.08923140e-05]])}
# parameters = update_parameters(parameters, grads)

# print("Weight1 = " + str(parameters["Weight1"]))
# print("bias1 = " + str(parameters["bias1"]))
# print("Weight2 = " + str(parameters["Weight2"]))
# print("bias2 = " + str(parameters["bias2"]))

#Output should be 

#Weight1 = [[-0.006267    0.01792921] [-0.02352903  0.03487509] [-0.01676255 -0.01895725] [ 0.00981865 -0.05423187]]
#bias1 = [[-9.50308498e-07] [ 1.00646934e-05][ 6.99888206e-07] [-2.81884090e-06]]
#Weight2 = [[-0.01042311 -0.04204123  0.01670053  0.04568113]]
#bias2 = [[9.69415948e-05]]


# In[109]:


def model(X, Y, n_h, num_iterations = 1000, print_cost=True):
    """
    X -- dataset
    Y -- labels
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent
    print_cost -- if True, print the cost in every 100 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(42)
    n_x = layerSizes(X, Y)[0]
    n_y = layerSizes(X, Y)[2]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    Weight1 = parameters['Weight1']
    bias1 = parameters['bias1']
    Weight2 = parameters['Weight2']
    bias2 = parameters['bias2']
    
    # gradient descent

    for i in range(0, num_iterations):
         
        # Call the Forward propagation with X, and parameters.
        A2, cache = forward_prop(X, parameters)
        
        # Call the Cost function with A2, Y and parameters.
        cost = compute_cost(A2, Y, parameters)
 
        # Call Backpropagation with Inputs, parameters, cache, X and Y.
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Update gradient descent parameter with  parameters and grads and learning rate.
        parameters = update_parameters(parameters, grads)
        
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


# In[110]:


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    parameters -- python dictionary containing your parameters 
    X -- input data
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_prop(X,parameters)
    predictions = (A2 > 0.5)
    
    return predictions


# In[111]:


# Build a model with a n_h-dimensional hidden layer
parameters = model(features, label, n_h = 5, num_iterations = 5000, print_cost=True)

# Plot the decision boundary
# plot_decision_boundary(lambda x: predict(parameters, x.T), features, label[0])
# plt.title("Decision Boundary for hidden layer size " + str(4));

# Print accuracy
predictions = predict(parameters, features)
print ('Accuracy: %d' % float((np.dot(label, predictions.T) + np.dot(1-label,1-predictions.T))/float(label.size)*100) + '%')


# In[ ]:




