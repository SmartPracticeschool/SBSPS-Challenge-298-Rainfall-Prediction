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


#Read The Dataset
Gangetic_West_Bengal_dataset = pd.read_csv('dataset/Gangetic West Bengal Rainfall Dataset.csv')

#View The Dataset
print(Gangetic_West_Bengal_dataset)

#Dataset info
print('Dataset infoormation:')
print(Gangetic_West_Bengal_dataset.info())

#Drop Unnecessary attributes
temp_data = pd.DataFrame(Gangetic_West_Bengal_dataset)
temp_dataset = temp_data.drop(['JF', 'MAM', 'JJAS', 'OND'], axis = 1)

print(temp_dataset)

#Clean The Dataset
print(temp_dataset.isnull().sum())

#Plot The Dataset
#Boxplot
plt.title('Boxplot')
temp_dataset.boxplot()
plt.show()

#histogram 
temp_dataset.hist()
plt.show()

#Kernel Density Estimation
temp_dataset.plot.kde()
plt.show()

#Correlation
sns.heatmap(temp_dataset.corr(),annot=True)
plt.show()

#Preprocess The Data
dataset = pd.get_dummies(temp_dataset, prefix='', prefix_sep='')

#Split the data into train and test
train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print('Train dataset shape :', train_dataset.shape)
print('Test dataset shape :',test_dataset.shape)

#Actual Annual Rainfall
actual_rainfall = test_dataset['ANNUAL']
year = test_dataset['YEAR']
print('Actual rainfall dataset shape :', actual_rainfall.shape)
print('Year dataset shape :',test_dataset.shape)

#Inspect the Train Data
sns.pairplot(train_dataset[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']], diag_kind="kde")
plt.show()

#Inspect the Test Data
sns.pairplot(test_dataset[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']], diag_kind="kde")
plt.show()

#The Overall Statistics of Total Data
statistics = temp_dataset.describe()
statistics = statistics.transpose()
print('Overall Statistics of Total Data:')
print(statistics)

#The Overall Statistics of Training Data
train_stats = train_dataset.describe()
train_stats.pop('ANNUAL')
train_stats = train_stats.transpose()
print('Overall Statistics of Training Data:')
print(train_stats)

#The Overall Statistics of Test Data
test_stats = test_dataset.describe()
test_stats.pop('ANNUAL')
test_stats = test_stats.transpose()
print('Overall Statistics of Test Data:')
print(test_stats)

#Labeling The datasets
train_labels = train_dataset.pop('ANNUAL')
test_labels = test_dataset.pop('ANNUAL')
print('Train labels dataset shape :', train_labels.shape)
print('Test labels dataset shape :',test_labels.shape)

#Save The Array in The Local Disk....
np.save('dataset/train_dataset.npy',train_dataset)
np.save('dataset/train_labels.npy',train_labels)
np.save('dataset/test_dataset.npy',test_dataset)
np.save('dataset/test_labels.npy',test_labels)
np.save('dataset/actual_rainfall.npy',actual_rainfall)
np.save('dataset/year.npy',year)

print('\nAll Dataset saved on the [dataset] Directory.....')