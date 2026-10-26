#CSCI 6201 Lab 2
# Ezra Anteneh | ezra.anteneh@gwu.edu | G37183321
#Date Last Updated: October 25, 2026

#Dataset: 9.1.3. Optical recognition of handwritten digits dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


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

counts = np.bincount(y)
plt.bar(range(10), counts)
plt.title("Number of images per digit")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.xticks(range(10))
plt.savefig("figures/class_counts.png")
plt.close()


#Visualization 1: Average Shape of Numbers (per dataset samples)
count = 0
while count < 10:
    plt.imshow(numbers.images[y == count].mean(axis=0), cmap="gray_r")
    plt.title(f"Average image of digit {count}")
    plt.show()
    count +=1


print("\n")