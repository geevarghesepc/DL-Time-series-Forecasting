# univariate cnn example
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from pandas import DataFrame
from pandas import concat
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import math

# fix random seed for reproducibility
numpy.random.seed(7)

# define dataset
csv_file = "data/daily-total-female-births.csv"
dataframe = pandas.read_csv(csv_file, usecols=[1], engine='python', skipfooter=3)
values = dataframe.values
values = values.astype('float32')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


dataset = series_to_supervised(values, 3)
y = dataset['var1(t)']
X = dataset.drop('var1(t)', axis=1)
X = X.values

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape', 'cosine'])
# fit model
model.fit(X, y, epochs=2000, verbose=0)

# Estimate model performance
trainScore = model.evaluate(X, y, verbose=0)
print(model.metrics_names)
print(trainScore)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))


# demonstrate prediction
# x_input = numpy.array([41, 52, 34])  # 53
x_input = numpy.array([38, 51, 31])   #31
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)