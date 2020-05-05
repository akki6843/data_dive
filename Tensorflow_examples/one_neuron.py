# importing tensorflow and numpy

import numpy as np 
import tensorflow as tf 

# Defining One Neuron Model

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# Compile the model 
model.compile(optimizer='sgd', loss='mean_squared_error')

# display model 

print(model.summary())

# creating a dummy data.
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Training model for 500 epochs   
model.fit(xs, ys, epochs=500)

# checking the prediction 
print(model.predict([10.0]))