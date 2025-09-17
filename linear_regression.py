import pandas as pd
import numpy as np

data = pd.read_csv("data/data.csv")

x_feature = data["km"].values # feature
y_target = data["price"].values # target

# Gradient descent formula:
# 1- w = w - a * J(w, b)
# 2- b = b - a * J(w, b)
# J(w, b) -> cost function
def gradient_descent(X, Y, iterations = 1000):
    m = float(len(Y))
    learning_rate = 0.001
    w = 0.0 # weight/slope
    b = 0.0 # bias/intercept

    for i in range(iterations):
        y_pred =  (w * X) + b
        
        w_derivative = (2/m) * np.sum((y_pred - Y) * X)
        b_derivative = (2/m) * np.sum((y_pred - Y))

        w -= learning_rate * w_derivative
        b -= learning_rate * b_derivative

    return w, b

estimated_w, estimated_b = gradient_descent(x_feature, y_target)

print(f"Estimated Weight: {estimated_w} | Estimated Bias: {estimated_b}")