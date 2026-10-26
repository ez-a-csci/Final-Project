#CSCI 6201 Lab 2
# Ezra Anteneh | ezra.anteneh@gwu.edu | G37183321
#Date Last Updated: October 25, 2026

#Dataset: 9.1.3. Optical recognition of handwritten digits dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


numbers = load_digits()
x = numbers.data
y = numbers.target



print("\nDataset Keys: \n", numbers.keys())


#overview of dataset
print("\n\n\nDataset overview:\n\n")

print("\nTotal number of samples in dataset: \n",x.shape[0])


print("\nRecorded number of features per sample in dataset (formatted in an 8x8 grid): \n", x.shape[1])

print("\nName of features: \n", numbers.feature_names)

print("\nName of targets (possible numbers identified): \n", numbers.target_names)


#print table
table = pd.DataFrame(x, columns=numbers.feature_names)

print("\nTable: \n", table)

#print statistical summary
print("\nStatistical summary: \n", table.describe()) 

print("\nSmallest recorded pixel value in dataset: ", x.min())

print("\nLargest recorded pixel value in dataset: ", x.max())

print("\nAverage recorded pixel value in dataset: ", x.mean())

print("\nCheck for missing data values: ", table.isna().sum())


print("\nNumber of identified numbers in dataset (organized from 0 to 9): ", np.bincount(y))

print("\n")

#Visualization: Average image of each digit based on sample:
count = 0
while count < 10:
    plt.imshow(numbers.images[y == count].mean(axis=0), cmap="gray_r")
    plt.title(f"Average image of digit {count}")
    plt.show()
    count +=1



#Base prediction model (WIP)
print("\n\n\nPrediction Model:\n\n")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.50, random_state = 10, stratify = y)
print("\nModel's training set size: ", x_train.shape)
print("\nModel's test set size: ", x_test.shape)

model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)


#comparison between predicted and true values of first 30 inputs
i = 0
predicted_output = model.predict(x_test)
while i < 30:
    print(f"For input {i}:\n")
    print("Sample predictions: ", predicted_output[i])
    print("Real value: ", y_test[i])
    print("\n")
    i += 1






print("\n")