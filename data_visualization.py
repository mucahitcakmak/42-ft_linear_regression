import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/data.csv")
x_values = data["price"].values
y_values = data["km"].values

plt.scatter(x_values, y_values, color="blue", marker="o")
plt.title("")
plt.xlabel("Price")
plt.ylabel("Car Mileage")
plt.show()

