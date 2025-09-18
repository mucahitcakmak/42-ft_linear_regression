import json
import pandas as pd

class LinearRegression:
    
    def __init__(self, data, epochs=1000, debug=False):
        self.learning_rate = 0.01
        self.theta0 = 0.0  # bias
        self.theta1 = 0.0  # weight
        self.m = len(data) # data length
        self.X = data["km"]
        self.Y = data["price"]
        self.normalized_x, self.normalized_y = [], []
        self.debugMode = debug
        self.normalized()
        self.train(epochs)
        self.saveParameters()

    def normalized(self):
        max_x, min_x = max(self.X), min(self.X)
        max_y, min_y = max(self.Y), min(self.Y)
        for i in range(self.m):
            self.normalized_x.append( (float(self.X[i]) - min_x) / (max_x - min_x) )
            self.normalized_y.append( (float(self.Y[i]) - min_y) / (max_y - min_y) )

    def estimatePrice(self, mileage):
        return (mileage * self.theta1) + self.theta0

    def computeGradients(self):
        grad0, grad1 = 0.0, 0.0
        for i in range(self.m):
            error = self.estimatePrice(self.normalized_x[i]) - self.normalized_y[i]
            grad0 += error
            grad1 += error * self.normalized_x[i]
        grad0 /= self.m
        grad1 /= self.m
        return grad0, grad1

    def train(self, epochs=1000):
        for epoch in range(epochs):
            grad0, grad1 = self.computeGradients()
            self.theta0 -= self.learning_rate * grad0
            self.theta1 -= self.learning_rate * grad1
            
            if self.debugMode and epoch % 100 == 0:
                cost = sum((self.estimatePrice(self.normalized_x[i]) - self.normalized_y[i])**2 for i in range(self.m)) / (2 * self.m)
                print(f"Epoch {epoch}, Cost: {cost:.4f}, θ0={self.theta0:.4f}, θ1={self.theta1:.4f}")

        print("Final params:")
        print(f"θ0={self.theta0}, θ1={self.theta1}")

    def saveParameters(self):
        params = {"theta0": self.theta0, "theta1": self.theta1}
        with open("model.json", "w") as f:
            json.dump(params, f)

def main():
    data = pd.read_csv("data/data.csv")
    LinearRegression(data, epochs=2000, debug=True)


if __name__ == '__main__':
    main()