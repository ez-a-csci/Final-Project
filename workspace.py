#CSCI 6201 Lab 2

#Date Last Updated: October 25, 2026

#Dataset: 9.1.3. Optical recognition of handwritten digits dataset

#This workspace is just where I practiced examining/manipulating the data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


numbers = load_digits()
x = numbers.data
y = numbers.target


print("keys: \n", numbers.keys())


print("x shape: \n",x.shape)


print("y shape: \n", y.shape)


print("feature names: \n",numbers.feature_names)


print("target names: \n", numbers.target_names)


print("x[0]: \n", x[0])


print("y[0]: \n",y[0])


print("image shape: \n",numbers.images.shape)

import matplotlib.pyplot as plt
plt.imshow(numbers.images[0], cmap="gray_r")
plt.show()
