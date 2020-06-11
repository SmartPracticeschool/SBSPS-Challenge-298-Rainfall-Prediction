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


#Convolutional Neural Network(CNN) building 
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
 
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

print('Model Sructure:')
model.summary()

#Test the Network
example = train_dataset[:10]
result = model.predict(example)
print('Prediction Test:')
print(result)

ch_path = ('save/cp.ckpt')
cp_dir = os.path.dirname(ch_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(ch_path,
                                                 save_weights_only = True,
                                                 verbose = 1)
model = build_model()

#Train the model
EPOCHS = 50000

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots(), cp_callback])



  #Visualize the model's
model_hist = pd.DataFrame(history.history)
model_hist['epoch'] = history.epoch
print('Model History:')
print(model_hist)

#Plot The Model 
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

#LOSE
plotter.plot({'Basic': history}, metric = 'loss')
plt.show()

#Mean Absolute Error
plotter.plot({'Basic': history}, metric = 'mae')
plt.show()

#Mean Square Error
plotter.plot({'Basic': history}, metric = "mse")
plt.show()

#Calculate Lose, Mae, Mse
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} ANN".format(mae))

model.load_weights(ch_path)
loss, mae, mse = model.evaluate(test_dataset, test_labels)
print('restored model, accuracy: {:5.2f} ANN'.format(mae))

ch_path_2 = ('save/cp-{epoch:04d}.ckpt')
cp_dir_2 = os.path.dirname(ch_path_2)

cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(ch_path_2, 
                                                   save_weights_only =  True, 
                                                   verbose = 1,
                                                   period = 50)
model = build_model()

#Train the model
history = model.fit(train_dataset, train_labels, 
                      epochs=EPOCHS,  batch_size=32 ,verbose=0,
                      validation_data=(test_dataset, test_dataset), 
                      callbacks = [cp_callback_2])

latest_model = tf.train.latest_checkpoint(cp_dir_2)
print(latest_model)

#save
model.save_weights('./save/my_save')

#restore
model = build_model()
model.load_weights('./save/my_save')

loss, mae, mse = model.evaluate(test_dataset, test_labels)
print('restored model, accuracy: {:5.2f} ANN'.format(mae))

model = build_model()
new_history = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                     validation_split = 0.2, batch_size=32 ,verbose=0,
                    callbacks=[tfdocs.modeling.EpochDots()])


#save entire model to a HDF5 file
model.save('saved model/my_model.h5')

#Load model
new_model = keras.models.load_model('saved model/my_model.h5')
new_model.summary()


loss, mae, mse = new_model.evaluate(test_dataset, test_labels)
print('restored model, accuracy: {:5.2f} ANN'.format(mae))

#Visualize the model's
new_model_hist = pd.DataFrame(new_history.history)
new_model_hist['epoch'] = new_history.epoch
print('Model History:')
print(new_model_hist)

#Plot The Model 
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

#LOSE
plotter.plot({'Basic': new_history}, metric = 'loss')
plt.show()

#Mean Absolute Error
plotter.plot({'Basic': new_history}, metric = 'mae')
plt.show()

#Mean Square Error
plotter.plot({'Basic': new_history}, metric = "mse")
plt.show()

#Calculate Lose, Mae, Mse
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} ANN".format(mae))

#Prediction
predictions = model.predict(test_dataset).flatten()
print(predictions)

r2_Score = r2_score(actual_rainfall, predictions)
print('Score: %0.3f' % r2_Score)

rmse = sqrt(mean_squared_error(actual_rainfall, predictions))
print('Root Mean Squared Error = %0.3f' % rmse)

#Mean Squared Error
mse = mean_squared_error(actual_rainfall, predictions)
print('Mean Squared Error = %0.3f' % mse)

#Mean Absolute Error
mae = mean_absolute_error(actual_rainfall, predictions)
print('Mean Absolute Error = %0.3f' % mae)

#Mean Squared Log Error
msle = mean_squared_log_error(actual_rainfall, predictions)
print('Mean Squared Log Error = %0.3f' % msle)

#Max Error
me = max_error(actual_rainfall, predictions)
print('Max Error = %0.3f' % me)

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

# Plot training & validation loss values
plt.plot(new_history.history['loss'])
plt.plot(new_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(new_history.history['mae'])
plt.plot(new_history.history['val_mae'])
plt.title('Model loss')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(new_history.history['mse'])
plt.plot(new_history.history['val_mse'])
plt.title('Model loss')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

def Average(lst):
 return sum(lst) / len(lst) 

loss = new_history.history['loss']
mae = new_history.history['mae']
mse = new_history.history['mse']

avg_loss = Average(loss)
avg_mae = Average(mae)
avg_mse = Average(mse)

print('Loss = %0.3f' % avg_loss)
print('MAE = %0.3f' % avg_mae)
print('MSE = %0.3f' % avg_mse)
print('RMSE = %0.3f' % rmse)
print('RMSE = %0.3f' % rmae)

average_prediction = Average(predictions)
average_prediction.reshape(-1,1)

average_actual = Average(actual)
average_actual

accuracy = average_actual / average_prediction *100
print('Accuracy = {:5.2f}%'.format(accuracy))