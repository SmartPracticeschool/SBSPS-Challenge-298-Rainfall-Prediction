#Import required libraries
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error


train_dataset = np.load('dataset/train_dataset.npy')
train_labels = np.load('dataset/train_labels.npy')
test_dataset = np.load('dataset/test_dataset.npy')
test_labels = np.load('dataset/test_labels.npy')
actual_rainfall = np.load('dataset/actual_rainfall.npy')
year = np.load('dataset/year.npy')

print('Train dataset shape :', train_dataset.shape)
print('Test dataset shape :',test_dataset.shape)
print('Train labels dataset shape :', train_labels.shape)
print('Test labels dataset shape :',test_labels.shape)
print('Actual rainfall dataset shape :', actual_rainfall.shape)
print('Year dataset shape :',test_dataset.shape)

#Load model
new_model = keras.models.load_model('saved model/my_model.h5')
new_model.summary()


loss, mae, mse = new_model.evaluate(test_dataset, test_labels)
print('restored model, accuracy: {:5.2f} ANN'.format(mae))

#Prediction
predictions = new_model.predict(test_dataset).flatten()

#Plot True Values vs. Predictions
a = plt.axes(aspect='equal')
plt.scatter(test_labels, predictions)
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
lims = [0,2500]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

#Plot Actuall vs. Prediction
plt.title('Actual Rainfall vs. Predicted Rainfall')
plt.plot(year, actual_rainfall)
plt.plot(year, predictions, 'r--')
plt.xlabel('[__Actual Rainfall],  [----Predicted Rainfall]')
plt.ylabel('Rainfall')
plt.show()

#Plot Prediction Error vs.Count
error = predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [ANN]")
_ = plt.ylabel("Count")
plt.show()



def Average(lst):
 return sum(lst) / len(lst) 

average_prediction = Average(predictions)
average_prediction.reshape(-1,1)

average_actual = Average(actual_rainfall)
average_actual

accuracy = average_actual / average_prediction *100
print('Accuracy = {:5.2f}%'.format(accuracy))
