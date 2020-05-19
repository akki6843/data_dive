#%%
# Importing the necessary libraries for polynomial regression

import tensorflow as tf  
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 


# %%
# import dataset for this task
# In this example we are using the dataset provided by SuperDataScience.

data = pd.read_csv(".././regression/polynomial-linear-regression/Position_Salaries.csv")
print(data.head())


# %%
X = data['Level'].values
y = data['Salary'].values


# %%
print(y)

# %%
# Creating model for training 
def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.
                layers.Dense(units=1, input_shape=[1]))
    model.add(tf.keras.layers.Dense(units=1))
    model.add(tf.keras.layers.Dense(units=1))
    # Compile the model 
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model 
# display model 

model = build_model() 
print(model.summary())


# %%

model.fit(X, y, epochs=5)

# %%
plt.scatter(X, y, color = 'red')
plt.plot(X, model.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# %%
