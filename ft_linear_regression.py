import pandas as pd

class LinearRegression:
    
    def __init__(self, data):
        self.learning_rate = 0.001
        self.theta0 = 0.0 # bias
        self.theta1 = 0.0 # weight
        self.m = len(data) # data length
        self.X = data["km"]
        self.Y = data["price"]
        self.normalized_x, self.normalized_y = [], []
        self.normalized()
    
    def normalized(self):
        max_x, min_x = max(self.X), min(self.X)
        max_y, min_y = max(self.Y), min(self.Y)
        for i in range(self.m):
            self.normalized_x.append( (float(self.X[i]) - min_x) / (max_x - min_x) )
            self.normalized_y.append( (float(self.Y[i]) - min_y) / (max_y - min_y) )
        print(self.normalized_x)
        print("******")
        print(self.normalized_y)
    
    def estimatePrice(self, mileage):
        return (mileage * self.theta1) + self.theta0

    def meanSquareError(self):
        sum = 0.0
        for i in range(self.m):
            sum += (self.estimatePrice(self.normalized_x[i]) - self.normalized_y[i]) ** 2
        return sum

    def train(self):
        pass # kafam karisti iyice


def main():
    data = pd.read_csv("data/data.csv")
    LinearRegression(data)


if __name__ == '__main__':
    main()