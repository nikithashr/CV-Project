#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 00:08:27 2018

@author: NikithaShravan
"""
import numpy as np




def relu_function(z):
    return np.maximum(z,0)

def tanh_function(z):
    return np.tanh(z)

def sigmoid_function(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_backward(z):
    return sigmoid_function(z)*(1.0 - sigmoid_function(z))
def tanh_backward(z):
    return np.arctanh(z)
def linear_forward(X, W, b):
    #  X: NxD, W1: dimensions: DxL1, b: 1xL1
    return np.dot(X ,W ) + b

def non_linear_activation(z,activation):
    if activation == "relu":
        return relu_function(z)
    if activation == "tanh":
        return tanh_function(z)
    if activation == "sigmoid":
        return sigmoid_function(z)
def relu_backward(z):
    r = relu_function(z)
    if np.max(r) == 0:
        pass
    else :
        r = np.ceil(r/np.max(r))
    return r
    
def non_linear_backward(z,activation):
    if activation == "relu":
        return relu_backward(z)
    if activation == "tanh":
        return tanh_backward(z)
    if activation == "sigmoid":
        return sigmoid_backward(z)
    
def compute_data_loss(scores,y):
    
#    f = scores - np.max(scores)
    
#    print("y shape: ", y.shape)
#    print("score shape: ", scores.shape)
    f1 = y*scores
    f2 = (1-y)*scores
    f1 = f1[y!=0] 
    f2 = f2[(1-y)!=0]
    t = np.sum(np.log(f1))+np.sum(np.log(f2)) 
#    print("t: ",t)
#    print("t1: ",np.sum(np.log(f1)))
#    print("t2: ",np.sum(np.log(f2)))
#    t1 = np.sum([[y[i][0]*np.log(f[i][0]),0][y[i][0]==0] for i in range(y.shape[0])])
#    t2 = np.sum([[(1-y[i][0])*(1-np.log(f[i][0])),0][y[i][0]==1] for i in range(y.shape[0])])
#    t2 = np.multiply(1 - y, np.log(1 - f))
    loss = (-1 / y.shape[0]) * t
#    loss = np.squeeze(loss)
    #for numerical stability we subtract maximum of scores
#    f = scores - np.max(scores)
#    print(f)
    
    #compute probabilities of each class for every training example
#    temp = np.exp(f)/np.sum(np.exp(f))#,1, keepdims=True)
#    print("p: ", p.shape)
    
    #computing the probability p for the true class for every example 'i': 
#    p_yi = p[range(len(y)),np.array(y)]
    # f = Wx, Li = -fyi + log Sigma(e^fj)
#    li = -1*np.log(p_yi)
    
#    return np.sum(li)/scores.shape[0]
    return loss/scores.shape[0]
def compute_data_loss_grad(scores, y):
    
    #for numerical stability we subtract maximum of scores
#    f = scores - np.max(scores)
##    print("F")
##    print(f)
#    
#    #compute probabilities of each class for every training example
#    df =  np.exp(f)/np.sum(np.exp(f),1, keepdims=True)
#    
#    #computing the probability p for the true class for every example 'i': 
#    df[range(len(y)),np.array(y)] -= 1
#       
#    df = df/scores.shape[0]
#
    epsilon = 1e-8
    scores[scores==0] = epsilon
    scores[scores==1] = 1+epsilon
    f1 = y*scores
    f2 = (1-y)*scores
    f1 = f1[y!=0] 
    f2 = f2[(1-y)!=0]    
    df = - (np.divide(y, scores) - np.divide(1 - y, 1 - scores))
     
    return df

def loss_estimation(y, y_train,loss_function):
    if loss_function == "cross-entropy":
#        print(y_train)
#        print(y)
        return (-1 / y_train.shape[1]) * np.sum(np.multiply(y_train, np.log(y)) + np.multiply(1 - y_train, np.log(1 - y)))
    
def loss_estimation_grad(y,y_train, loss_function):
    if loss_function == "cross-entropy":
#        print(y_train)
#        print(y)

        return - (np.divide(y_train, y) - np.divide(1 - y_train, 1 - y))

#    elif loss_function == "svm":
#        return svm_loss(y,y_train)
def linear_backward(A,dZ):
    return np.dot(dZ.T,A).T
class neural_net(object):
    def __init__(self, layers):
        
        self.params = {}
        self.layers = layers
        for i in range(len(self.layers)-1):
            self.params['W'+str(i+1)] = np.random.randn(layers[i],layers[i+1])
            self.params['b'+str(i+1)] = np.random.randn(layers[i+1])
            
    def update_parameters(self):
        pass
    def epoch(self,X_train, y_train,learning_rate, activation):
        
        # X_train dim: NxD
        A = X_train
        N = X_train.shape[0]
        H = len(self.params)//2
        assert(y_train.shape == (N,1))
        cache = []               
        #Z: NxL1  
        for i in range(H-1):
            
            # Z: Nxlayer(i+1),A: Nxlayer(i), W: layer(i)xlayer(i+1) 
            Z = linear_forward(A, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            cache.append((A,Z))
            # non linear activation: A:Nxlayer(i+1)
            A = non_linear_activation(Z, activation)
            assert(A.shape == (N,self.params['W'+str(i+1)].shape[1]))
            
        
        Z = linear_forward(A, self.params['W'+str(H)], self.params['b'+str(H)])
        cache.append((A,Z))
        assert(Z.shape == (N, self.layers[-1]))
        
        y = non_linear_activation(Z, "sigmoid")
#        print(Z)
#        print(y)
        
        loss = compute_data_loss(y, y_train)#, "cross-entropy")
        
        grads = {}
       
        dy = compute_data_loss_grad(y, y_train)#, "cross-entropy")
#        print(len(cache))
        dZ = dy*non_linear_backward(Z,"sigmoid")
        
        for i in reversed(range(H)):
            #A : Nxlayer(i+1)
#            print(dZ.shape, cache[i][0].shape )
            dW = np.dot(cache[i][0].T, dZ)
            dA = np.dot(dZ,self.params['W'+str(i+1)].T)
            db = np.squeeze(np.sum(dZ,axis=0))
            
            grads['W'+str(i+1)] = dW
            grads['b'+str(i+1)] = db
            if i == 0:
                break
#            print("cache size: ", cache[i-1][1].shape)
#            print("A: ",dA.shape )
#            print("i: ", i)
#            print("H: ", H)
            dZ = dA*non_linear_backward(cache[i-1][1],activation)
        
        for i in range(H):
            self.params['W'+str(i+1)] -= learning_rate*grads['W'+str(i+1)]
            self.params['b'+str(i+1)] -= learning_rate*grads['b'+str(i+1)]
        return np.sum(loss)
    def train(self,X_train, y_train,num_epochs,learning_rate=1e-1):
         
         loss = []
         for i in range(num_epochs):
             l = self.epoch(X_train,y_train,learning_rate,"relu")
             print(l)
             loss.append(l)
             
num_inputs = 100000
input_dims = 90
X_train = 10*np.random.randn(num_inputs, input_dims)
#print(X_train)
y_train = np.random.randint(2,size=(1,num_inputs)).T 
#print(y_train)
np.random.seed(1)
layers =[input_dims,50,1] 
net = neural_net(layers)
net.train(X_train, y_train,10)