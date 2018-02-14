# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:15:27 2018

@author: Ittipat
"""

import numpy as np
import matplotlib.pyplot as plt

#xdata = np.array([0.0,   1.0,  3.0, 4.3, 7.0,   8.0,   8.5, 10.0, 12.0])
#ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43,  0.7, 0.89, 0.95, 0.99])

xdata = np.array([1.393618635, 0.791558644, 0.189498652, -0.412561339, -1.01462133, -1.616681322, -2.218741313])
ydata = np.array([33.16666667, 1147.333333, 3596.333333, 4546.333333, 4549.333333, 4800, 6813.333333])
xdata = -xdata

ydata = ((ydata - np.average(ydata)) / 6780) + 0.52495435


def sigmoid(x, k, w):
     y = 1 / (1 + np.exp(-k*(x-w)))
     return y
 
def derivative_k(x, k, w):
    return -((np.exp(-k*(x-w)))*(w-x))/((1 + np.exp(-k*(x-w))) ** 2)

def derivative_w(x, k, w):
    return -((np.exp(-k*(x-w)))*(k))/((1 + np.exp(-k*(x-w))) ** 2)  

def error(xata, ydata, k, w):
    total_error = 0
    for i in range(len(xdata)):
        total_error += (ydata[i] - sigmoid(xdata[i], k, w)) ** 2
    return total_error / float(len(xdata))

k = 0
w = 0

error(xdata, ydata, k, w)

def learn(k, w):

    learning_rate = 1e-3
    errors = []
    N = len(xdata)
    for epoch in range(1000000):
        r = error(xdata, ydata, k, w)
        
        if epoch % 100 == 0:
            print("Epoch_num: ", epoch, "error: ", r)
        
        errors.append(r)
        
        grad_k = 0
        grad_w = 0
        
        for i in range(len(xdata)):
            grad_k += -(2/N) * (ydata[i] - sigmoid(xdata[i], k, w)) * (xdata[i] - w)
            grad_w += (2/N) * (ydata[i] - sigmoid(xdata[i], k, w)) * k
        
        k = k - learning_rate * grad_k
        w = w - learning_rate * grad_w
               
    plt.plot(errors)
    plt.show()    

    return k, w

k, w = learn(k, w)

x = np.linspace(-1.5, 2.3, 50)
y = sigmoid(x, k, w)

plt.plot(xdata, ydata, 'o', label='RI01_activity', color='blue')
plt.plot(x, y, label='fit_activity')
plt.ylim(-0.1, 1.2)
plt.legend(loc='best')
plt.show()
