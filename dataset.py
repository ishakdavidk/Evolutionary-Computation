import numpy as np
import random

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


def generate_cityList(num_cities):
    cityList = []

    for i in range(0, num_cities):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

    return cityList


def ant_colony(file):
    dataset_dir = './dataset/' + file
    dataset = np.loadtxt(dataset_dir, delimiter=' ')[:, 1:]

    cityList = []

    for data in dataset:
        cityList.append(City(x=int(data[0]), y=int(data[1])))

    return cityList