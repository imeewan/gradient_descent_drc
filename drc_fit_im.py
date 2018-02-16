# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 00:02:44 2018

@author: Ittipat
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

xdata = np.array([1.393618635, 0.791558644, 0.189498652, -0.412561339, -1.01462133, -1.616681322, -2.218741313])
ydata = np.array([33.16666667, 1147.333333, 3596.333333, 4546.333333, 4549.333333, 4800, 6813.333333])
xdata = -xdata

ydata = ((ydata - np.average(ydata)) / 6780) + 0.52495435



popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(popt)
x = np.linspace(-2.5, 2.9, 50)
y = sigmoid(x, popt[0], 2)

x_tox_data = np.array([1.393618635, 0.791558644, 0.189498652, -0.412561339, -1.01462133, -1.616681322, -2.820801304])
y_tox_data = np.array([3151, 21770, 24837, 19921, 26650.5, 22794.5, 17802.5])
negative_control = np.array([(24509 + 17938)/2])
negative_control = ((negative_control -  np.average(y_tox_data)) / 26650.5) + 0.8
y_tox_data = ((y_tox_data - np.average(y_tox_data)) / 26650.5) + 0.8
x_tox_data = -x_tox_data

popt_tox, pcov_tox = curve_fit(sigmoid, x_tox_data, y_tox_data)
print(popt_tox)


x_tox = np.linspace(-2.5, 2.9, 50)
y_tox = sigmoid(x, popt_tox[0], 4)

plt.plot(xdata, ydata, 'o', label='RI01_activity', color='blue')
plt.plot(x_tox_data, y_tox_data, 'x', label='RI01_toxicity', color='red')
plt.plot(0, negative_control, '+', label='negative_control', color='magenta')
plt.plot(x, y, label='fit_activity')
plt.plot(x_tox, y_tox, label='fit_toxicity')
plt.ylim(-0.1, 1.2)
plt.legend(loc='best')
plt.show()
plt.savefig('F:\\learn\\antivirus\\hcv\\sigmoid_grad_fit\\RI01.png')

10 ** 1.33706




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

xdata = np.array([1.393618635, 0.189498652, -0.412561339, -1.01462133, -1.616681322, -2.218741313, -2.8208])
ydata = np.array([466.3333333, 4317, 3876, 5404, 3783.833333, 3263.833333, 2899.166667])
xdata = -xdata
ydata = ((ydata - np.average(ydata)) / 3000) + 1



popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(popt)


x = np.linspace(-2.5, 2.5, 50)
y = sigmoid(x, -0.98, 6)

x_tox_data = np.array([1.393618635, 0.791558644, 0.189498652, -0.412561339, -1.01462133, -1.616681322, -2.218741313, -2.820801304])
y_tox_data = np.array([5098.5, 33031.5, 24543, 26960, 28110.5, 27750, 27654, 19343])
negative_control = np.array([(24509 + 17938)/2])
negative_control = ((negative_control -  np.average(y_tox_data)) / 17802.5) + 1.1
y_tox_data = ((y_tox_data - np.average(y_tox_data)) / 17802.5) + 1.1
x_tox_data = -x_tox_data

popt_tox, pcov_tox = curve_fit(sigmoid, x_tox_data, y_tox_data)
print(popt_tox)


x_tox = np.linspace(-2.5, 2.9, 50)
y_tox = sigmoid(x, popt_tox[0], 8)

plt.plot(xdata, ydata, 'o', label='RI02_activity', color='blue')
plt.plot(x_tox_data, y_tox_data, 'x', label='RI02_toxicity', color='red')
plt.plot(0, negative_control, '+', label='negative_control', color='magenta')
plt.plot(x, y, label='fit_activity')
plt.plot(x_tox, y_tox, label='fit_toxicity')
plt.ylim(-0.1, 1.2)
plt.legend(loc='best')
#plt.show()
plt.savefig('F:\\learn\\antivirus\\hcv\\sigmoid_grad_fit\\RI02.png')

10 ** 0.98
10 ** -popt_tox[0]








import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

xdata = np.array([1.393618635, 0.189498652, -0.412561339, -1.01462133, -1.616681322, -2.218741313, -2.8208])
ydata = np.array([466.3333333, 4317, 3876, 5404, 3783.833333, 3263.833333, 2899.166667])
xdata = -xdata
ydata = ((ydata - np.average(ydata)) / 3000) + 1



popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(popt)


x = np.linspace(-2.5, 2.5, 50)
y = sigmoid(x, -0.98, 6)

x_tox_data = np.array([1.393618635, 0.791558644, 0.189498652, -0.412561339, -1.01462133, -1.616681322, -2.218741313, -2.820801304])
y_tox_data = np.array([14257, 13799, 19043, 18159, 15475, 17752, 15672, 13823])
y_tox_data = ((y_tox_data - np.average(y_tox_data)) / 17802.5) + 1.1
x_tox_data = -x_tox_data

popt_tox, pcov_tox = curve_fit(sigmoid, x_tox_data, y_tox_data)
print(popt_tox)


x_tox = np.linspace(-2.5, 2.9, 50)
y_tox = sigmoid(x, popt_tox[0], 8)

plt.plot(xdata, ydata, 'o', label='RI02_activity', color='blue')
plt.plot(x_tox_data, y_tox_data, 'x', label='RI02_toxicity', color='red')
plt.plot(x, y, label='fit_activity')
plt.plot(x_tox, y_tox, label='fit_toxicity')
plt.ylim(-0.1, 1.2)
plt.legend(loc='best')
#plt.show()
plt.savefig('F:\\learn\\antivirus\\hcv\\sigmoid_grad_fit\\RI02.png')












