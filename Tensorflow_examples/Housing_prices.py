# import libraries
#%% 
import tensorflow as tf 
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import statsmodels.api as sm


#%%
# Import dataset from keras for the analysis purposes.

dataset = tf.keras.datasets.boston_housing

# Read the following link https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html to understand bostan housing problem 
'''
Miscellaneous Details
-Origin
    The origin of the boston housing data is Natural.
-Usage
    This dataset may be used for Assessment.
-Number of Cases
    The dataset contains a total of 506 cases.
-Order
    The order of the cases is mysterious.
-Variables
        There are 14 attributes in each case of the dataset. They are:
        CRIM - per capita crime rate by town
        ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
        INDUS - proportion of non-retail business acres per town.
        CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
        NOX - nitric oxides concentration (parts per 10 million)
        RM - average number of rooms per dwelling
        AGE - proportion of owner-occupied units built prior to 1940
        DIS - weighted distances to five Boston employment centres
        RAD - index of accessibility to radial highways
        TAX - full-value property-tax rate per $10,000
        PTRATIO - pupil-teacher ratio by town
        B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        LSTAT - % lower status of the population
        MEDV - Median value of owner-occupied homes in $1000's
-Note
    Variable #14 seems to be censored at 50.00 (corresponding to a median price of $50,000); Censoring is suggested by the fact that the highest median price of exactly $50,000 is reported in 16 cases, while 15 cases have prices between $40,000 and $50,000, with prices rounded to the nearest hundred. Harrison and Rubinfeld do not mention any censoring.

'''
#%%
# Loading the dataset into the required labels 
(train_data, train_labels), (test_data, test_labels) = dataset.load_data()

print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

# creating the column names from the dataset description.
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

# creating dataframe for the data manipulation.


df = pd.DataFrame(train_data, columns=[column_names])
print(df.head())
# test_data = pd.DataFrame(test_data, columns=[column_names])

# train_labels = pd.DataFrame(train_labels, columns=["Price"])
# train_labels = pd.DataFrame(test_labels, columns=["Price"])

# normalizing the data using the mean and standard deviation 
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

#%%
# creating model using tensorflow keras sequential

def create_model() :
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_shape=(train_data.shape[1], )))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.relu))
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model


print(train_data.shape[1])
model = create_model()
print(model.summary())

#%%
# Train a model but use keras call back to show the progress

class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0: print('')
        print('.', end=' ')

EPOCHS = 500

#%%
# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

#%%
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0,5])
  plt.show()

#%%
plot_history(history)


# %%
# From the above observation we can see that convergence started little  over 200 
# So what if model is trainned using some callbacks to do early stopping.

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


#%%
plot_history(history)

# %%
# Evaluating model performance 


[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))


# %%
# Running prediction on the trainned model

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100],[-100,100])


# %%
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")




#%%
# Implementing backward elimination method with the p value threshold as 0.5
df  = pd.DataFrame(train_data, columns=[column_names])
print(df.head())
print(df.shape)

df.insert(0, 'B0', [1]*len(df))
print(df.head())
print(df.shape)


# %%
df_optimized = df

# %%
regressor_OLS = sm.OLS(endog = train_labels, exog = df_optimized).fit()

# %%
regressor_OLS.summary()

# %%
df_optimized = df[['B0', 'CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']]
# %%
regressor_OLS = sm.OLS(endog = train_labels, exog = df_optimized).fit()
regressor_OLS.summary()

# %%
df_optimized = df[['B0', 'CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']]

# %%
regressor_OLS = sm.OLS(endog = train_labels, exog = df_optimized).fit()
regressor_OLS.summary()



# %%
# after performing the backward elimination to optimize the input datawe recreate model for training 
df_optimized = df[['B0', 'CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']]
df_optimized = df_optimized.values

print(df_optimized)

def build_model() :
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_shape=(train_data.shape[1], )))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.relu))
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model


# %%
model = build_model()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


# %%
plot_history(history)


# %%
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# %%
