import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split
import time


# get the start time
st = time.time()


data_path = "D:/cyclone/TCIR-ATLN_EPAC_WPAC.h5"
data_info = pd.read_hdf(data_path, key="info", mode='r')
with h5py.File(data_path, 'r') as hf:
    data_matrix = hf['matrix'][:]

print(data_matrix.shape)

## keep only IR and PMW
X_irpmw = data_matrix[:,:,:,0::3]
y = data_info['Vmax'].values[:]

X_irpmw[np.isnan(X_irpmw)] = 0
X_irpmw[X_irpmw > 1000] = 0


#Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X_irpmw, y, test_size=0.1, random_state=42)
# Define the input shape
input_shape = (201, 201, 2)



# Define the model
model = tf.keras.models.Sequential([
    
    # Convolutional layer 1
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    # Convolutional layer 2
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    
    # Convolutional layer 3
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

    # Flatten the output from the convolutional layers
    tf.keras.layers.Flatten(),
    # Fully connected layers for regression output
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

Adam = tf.keras.optimizers.Adam(learning_rate=5e-7)

# Compile the model with Mean Squared Error loss and Adam optimizer
model.compile(loss='mse', optimizer=Adam, metrics=['mae'])

# Print the summary of the model
model.summary()


# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=2)

#Training Model
model.fit(X_train,y_train ,epochs=100,validation_data=(X_test,y_test),verbose=2, callbacks=[early_stop])

#Saving the model
filename="model_0_AEW"
model.save("D:/cyclone/model/"+filename+".h5",save_format='h5')


y_pred=model.predict(X_test)
y_true=y_test

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error (MSE): ", mse)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE): ", rmse)

# R-squared score (R2)
r2 = r2_score(y_true, y_pred)
print("R-squared score (R2): ", r2)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error (MAE): ", mae)

# Explained Variance Score (EVS)
evs = explained_variance_score(y_true, y_pred)
print("Explained Variance Score (EVS): ", evs)


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')