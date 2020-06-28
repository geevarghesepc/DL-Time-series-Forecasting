# univariate mlp example
import numpy
from keras.models import Sequential
from keras.layers import Dense
import pandas
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
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

# print(y, X)
# exit(1)

trainX, trainY = [X, y]
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.20)

# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape', 'cosine'])
# fit model
model.fit(trainX, trainY, epochs=2000, verbose=0)
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print(model.metrics_names)
print(trainScore)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# testScore = model.evaluate(testX, testY, verbose=0)
# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# exit(1)

# demonstrate prediction
x_input = numpy.array([41, 52, 34])  # 53
# x_input = numpy.array([38, 51, 31])   #31
x_input = x_input.reshape((1, 3))
yhat = model.predict(x_input, verbose=0)
print(yhat)
