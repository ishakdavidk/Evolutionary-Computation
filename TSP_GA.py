import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import sys

from natural_crossover import natural_breed, semi_natural_breed, semi_natural_breed2
from crossover import breed
from dataset import generate_cityList, ant_colony


parser = argparse.ArgumentParser(description='Select crossover method (1 / 2)')
parser.add_argument('-c', '--crossover', type=int, metavar='', required=True, help='1: simple crossover; '
                                                                                   '2: natural crossover '
                                                                                   '3: semi-natural crossover')
parser.add_argument('-g', '--group', type=int, metavar='', help='specify number of groups if natural or semi-natural crossover is selected')
args = parser.parse_args()

if args.crossover < 1 or args.crossover > 3:
    sys.exit('Please select between the following options:\n'
             '1: simple crossover\n'
             '2: natural crossover\n'
             '3: semi-natural crossover')

num_groups = args.group

# dataset_file = None

# 1d_i dataset
# dataset_file = '1d_i/att48'
# dataset_file = '1d_i/chn31'
# dataset_file = '1d_i/chn144'

# 1d_ii dataset
# dataset_file = '1d_ii/att532'
dataset_file = '1d_ii/u724'
# dataset_file = '1d_ii/dsj1000'

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breedPopulation(matingpool, eliteSize, population):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    if args.crossover == 1:
        # Crossover
        for i in range(0, length):
            child = breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
    elif args.crossover == 2:
        # Natural crossover
        for i in range(0, length):
            child = natural_breed(pool[i], pool[len(matingpool) - i - 1], population, num_groups)
            children.append(child)
    elif args.crossover == 3:
        # Natural crossover
        for i in range(0, length):
            # print('child: ', i)
            if num_groups == 2:
                child = semi_natural_breed2(pool[i], pool[len(matingpool) - i - 1], population)
                children.append(child)
            else:
                child = semi_natural_breed(pool[i], pool[len(matingpool) - i - 1], population, num_groups)
                children.append(child)

    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate, population):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, population)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    initial_test = "Initial distance: " + str(1 / rankRoutes(pop)[0][1])

    print('Dataset: ' + str(dataset_file) + '\nIn progress.. ')

    x_best_idx = 0
    best_distance = 1 / rankRoutes(pop)[0][1]

    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        print('generation: ', i)
        pop = nextGeneration(pop, eliteSize, mutationRate, population)
        current_distance = 1 / rankRoutes(pop)[0][1]
        progress.append(current_distance)
        print('current_distance: ', current_distance)

        if current_distance <= best_distance:
            x_best_idx = i
            best_distance = current_distance

    final_text = "Final distance: " + str(1 / rankRoutes(pop)[0][1])
    best_text = 'Best Distance: gen=' + str(x_best_idx) + ', dist=' + str(round(best_distance, 2))

    plt.figure(figsize=(10, 7))
    plt.plot(progress)

    plt.annotate(str(round(best_distance, 2)), xy=(x_best_idx, best_distance))

    plt.ylabel('Distance', fontsize=15)
    plt.xlabel('Generation', fontsize=15)

    if args.crossover == 1:
        plt.title(best_text + '\n', fontsize=18)

        if dataset_file == None:
            dir_name = 'results/simple_crossover/'
        else:
            dir_name = 'results/simple_crossover/' + dataset_file + '/'
    elif args.crossover == 2:
        if num_groups != None:
            plt.title('Number of group: ' + str(num_groups) + '\n' + best_text + '\n', fontsize=18)
        else:
            plt.title(best_text + '\n', fontsize=18)

        if dataset_file == None:
            dir_name = 'results/natural_crossover/'
        else:
            dir_name = 'results/natural_crossover/' + dataset_file + '/'
    elif args.crossover == 3:
        if num_groups != None:
            plt.title('Number of group: ' + str(num_groups) + '\n' + best_text + '\n', fontsize=18)
        else:
            plt.title(best_text + '\n', fontsize=18)

        if dataset_file == None:
            dir_name = 'results/semi_natural_crossover/'
        else:
            dir_name = 'results/semi_natural_crossover/' + dataset_file + '/'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ts = str(int(time.time()))
    plt.savefig(dir_name + ts + '.jpg', dpi=300)

    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    best_text = 'Best Distance: ' + str(best_distance)
    best_route_text = 'Best Route: ' + str(bestRoute).strip('[]')

    description = initial_test + '\n' + final_text + '\n' + best_text + '\n' + best_route_text

    print(description)

    description_file_name = ts + '_' + 'description.txt'
    with open(dir_name + description_file_name, 'w') as text_file:
        print(description, file=text_file)

    plt.show()

    return bestRoute

if dataset_file == None:
    cityList = generate_cityList(24)
else:
    cityList = ant_colony(dataset_file + '.txt')

if num_groups != None:
    if num_groups < 1 or num_groups > len(cityList)-1 :
        sys.exit('Number of groups should be more than zero and less than number of cities (' + str(len(cityList)) + ')')

bestRoute = geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
