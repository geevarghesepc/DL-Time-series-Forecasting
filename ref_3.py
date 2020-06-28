import pandas
import matplotlib.pyplot as plt

csv_file = "data/daily-total-female-births.csv"
dataset = pandas.read_csv(csv_file, usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()
