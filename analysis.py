#CSCI 6201 Lab 2

#Date Last Updated: October 27, 2026

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

print("\nTable:", table)

#print statistical summary
print("\nStatistical summary: \n", table.describe()) 

print("\nSmallest recorded pixel value in dataset: ", x.min())

print("\nLargest recorded pixel value in dataset: ", x.max())

print("\nAverage recorded pixel value in dataset: ", x.mean())

print("\nCheck for missing data values: ", table.isna().sum().sum())


print("\nNumber of identified numbers in dataset (organized from 0 to 9): ", np.bincount(y))

print("\n")



#Base prediction model
print("\n\n\nPrediction Model:\n\n")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.50, random_state = 10, stratify = y)
print("\nModel's training set size: ", x_train.shape)
print("\nModel's test set size: ", x_test.shape)

model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)


#comparison between predicted output and actual values of first 30 inputs
size_of_test_batch = 30
i = 0
got_correct = 0
got_incorrect = []
predicted_output = model.predict(x_test)

#compare output between model and actual results
while i < size_of_test_batch:
    predicted = predicted_output[i]
    actual = y_test[i]
    print(f"For input {i}:")
    print("Sample predictions: ", predicted)
    print("Real value: ", actual)

    if (predicted == actual):
        got_correct += 1
    else:
        got_incorrect.append(i)
    print("\n")
    i += 1

#print response
print(f"The model got {got_correct} out of {size_of_test_batch} predictions correct." )
print(f"The model failed for input(s) {got_incorrect}." )

#print accuracy (roughly 96%)
model_eval = accuracy_score(y_test, predicted_output)
confusion = confusion_matrix(y_test, predicted_output)

print("\nAccuracy: ", model_eval)
print("\nConfusion: \n", confusion)

#Based on a test sample of 30, my model has an accuracy of ~96%.
#This is reflected by the confusion matrix, which shows that a majority of predictions fall along the diagonal.
#I hope to see improvement to at least above 98% by the time of the Final Project's end. 
#When increasing the "size_of_test_batch" variable to 500, I still retained a 96% accuracy rate. 
#I would like to experiment with other train-test splits beyond 50/50, and measure how they impact my accuracy rate. I'm particularly interested in figuring out what the smallest amount of training data is needed for a model that meets a 96% accuracy rate.
#For my final project, I would compare various forms of predictions models in order to find one that is both lightweight and takes a relatively small size of input training data.
#The purpose of this would be to minimize work required to recreate a similar system for a smaller, web-based game with custom symbols rather than digits. 

#Data Visualizations:

#Visualization 1: Average image of each digit based on sample:
image, axes = plt.subplots(2, 5, figsize = (15, 6))
count = 0
while count < 10:
    axes.flat[count].imshow(numbers.images[y == count].mean(axis = 0), cmap = "gray_r")
    axes.flat[count].set_title(f"Average image of digit {count}", fontsize = 10)
    count +=1

plt.tight_layout()
#The above code creates a table of values averaging the input pixels provided for each digit, thus creating a  "heatmap" of most important pixels for each digit.
#By examining the average "shape" of each number, I hope to find which 8x8 coordinates play the most important role in identifiying digits.
#I found it interesting how various numbers shared the same critical pixel-coordinates.
# Going forward, I may try to find what coordinates play the most important role in discerning one digit from another. 

#Visualization 2: Bar graph showing how many times the model fails to predict each digit
failures = []
for digit in range(10):
    wrong = ((y_test == digit) & (predicted_output != y_test)).sum()
    failures.append(wrong)

plt.figure()
plt.bar(range(10), failures)
plt.title("Number of incorrect predictions per digit")
plt.xlabel("Digit")
plt.ylabel("Number of times guessed incorrectly")
plt.xticks(range(10))
plt.show()

#The above code creates a bar graph that captures how many times model fails to correctly guess each number in the test group.
#Interestingly, it seems that my current model has the most trouble with the digit 8.
# Comparing this information of Visualization 1, this makes sense since the digit 8 overlaps with multiple other digits' averages.
#I'd like to repeat this comparison with different models before drawing a final conclusions. 

print("\n")


