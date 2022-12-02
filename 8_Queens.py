# Jared Goldsmith - jgolds@pdx.edu

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import torch
%matplotlib inline

def plot_values(average_fitness, best_fitness):
  fig = plt.figure(figsize=[10,3])
  ax2 = fig.add_subplot(121)
  ax2.plot(range(1, len(best_fitness)+1), best_fitness, label='Best Fitness')
  ax2.plot(range(1, len(average_fitness) + 1), average_fitness, label='Average Fitness')
  ax2.set_xlabel('Generations')
  ax2.set_ylabel('Fitness: Non-attacking pairs of Queens')
  ax2.legend()

def plot_generations_to_solution(iterations):
  fig = plt.figure(figsize=[10,3])
  ax = fig.add_subplot(121)
  ax.set_xlabel('Number of runs')
  ax.plot(range(1, len(iterations) + 1), iterations, label='Number of Iterations to Find Solution')
  ax.set_ylabel('Generations')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import random
from itertools import accumulate, dropwhile
from collections import Counter

class Queens(object):
    def __init__(self, rows=8, columns=8, population_size=2000, mutation=0.05):
        if rows != columns:
          print('Rows and columns need to be same size')
          return
        self.ROW = rows
        self.COL = columns
        self.population_size = population_size
        self.states = self.initialize_states()
        self.fitness_values = {}
        self.total_fitness = 0
        self.mutation = mutation
        self.average_fitness = []
        self.best_fitness = []
      
    # Picks a state based on its fitness level
    def pick_parent(self,d):
        draw = random.randrange(sum(d.values()))
        return list(d.keys())[next(dropwhile(lambda t: t[1] < draw, enumerate(accumulate(d.values()))))[0]]
    
    # Displays num_states based on same way for parent selection
    def display_states(self,num_states):
      list_of_states = []
      for i in range(num_states):
        state = self.pick_parent(self.fitness_values)
        fitness = self.fitness_values[state]
        print(f'State: {state} for fitness value of {fitness}')
      print('\n')

    # Picks two parents and creates two children. 
    # Loops until number of children matches original population
    def crossover(self):
        children = []
        fitness = {}
        while len(children) < self.population_size:
            parent1 = self.pick_parent(self.fitness_values)
            parent2 = self.pick_parent(self.fitness_values)
            child1 = []
            child2 = []
            
            while parent1 == parent2:
                parent2 = self.pick_parent(self.fitness_values)
            for i in range(3):
                child1.append(parent1[i])
                child2.append(parent2[i])
            for i in range(3,self.ROW):
                child1.append(parent2[i])
                child2.append(parent1[i])
            odds_of_mutation = random.randint(1,100)/100
            if odds_of_mutation < self.mutation:
                self.mutate(child1)
                self.mutate(child2)
            children.append(child1)
            children.append(child2)
        self.states = children
        return self.fitness()
        
        
    def run_crossovers(self, num_crossovers):
        for i in range(num_crossovers):
          if self.crossover() == -1:
            return -1

    # Mutates a random index with a random row value
    def mutate(self,state):
        index = random.randint(0,self.ROW-1)
        row = random.randint(1,self.ROW)
        state[index] = row     
             
    def initialize_states(self):
        states = []
        for i in range(self.population_size):
            state = self.initialize_state()
            states.append(state)
        return states
              
    def initialize_state(self):
        state = []
        for i in range(self.ROW):
            row = random.randint(1,self.ROW)
            state.append(row)   
        return state

    # Calculates exposed queens to an attack and subtracts by
    # max number of non-attacking pairs of queens (28 queens)
    def fitness(self):
        ROW, COL = len(self.states), len(self.states)
        cost = 0
        self.total_fitness = 0
        self.fitness_values = {}
        best_fitness = float('-infinity')
        for position in self.states:
            cost = 0
            for i in range(len(position)):
                cost += self.diagonal_cost(i,position)
                cost += self.row_cost(i,position)
            fitness = 28 - cost/2
            if fitness > best_fitness:
              best_fitness = fitness
            # Uncomment out to exit early for solution found
            
            if fitness == 28:
              fitness = int(self.total_fitness/self.population_size)
              self.average_fitness.append(fitness)
              self.best_fitness.append(best_fitness)
              return -1
            
            self.fitness_values[tuple(position)] = fitness
            self.total_fitness += fitness
        
        fitness = int(self.total_fitness/self.population_size)
        self.average_fitness.append(fitness)
        self.best_fitness.append(best_fitness)
        return fitness
       
    def diagonal_cost(self,index,state):
        pairs = self.convert_to_pairs(state)
        column = index + 1
        row = state[index]
        cost = 0
        movements = [(1,-1),(-1,1),(1,1),(-1,-1)]
        for row_adj, col_adj in movements:
            r,c = row+row_adj, column+col_adj
            while r in range(1,self.ROW+1) and c in range(1,self.COL+1):
                pair = (r,c)
                if pair in pairs:
                    cost += 1
                r += row_adj
                c += col_adj
        return cost

    def row_cost(self,index,state):
        cost = 0
        row = state[index]
        
        for i in range(self.COL):
            if i == index:
                continue
            if state[i] == row:
                cost += 1
        return cost

    # Converts string state to (row,column) pairs for 
    # simplifying fitness and cost calculations
    def convert_to_pairs(self,state):
        pairs = set()
        for i in range(len(state)):
            row = state[i]
            col = i + 1
            pair = (row,col)
            pairs.add(pair)
        return pairs
  
if __name__ == '__main__':
  
    # Parameters
  num_runs = 30
  population = 500
  crossovers = 300
  mutation = 0.1

  iterations_to_solve = []
  averages = [0]*(crossovers+1)
  bests = [0]*(crossovers+1)


  for run in range(1,num_runs+1):
    queens = Queens(population_size=population)
    queens.initialize_state()
    queens.fitness()
    if run == 1:
      print('Initial States:')
      queens.display_states(10)
    queens.run_crossovers(crossovers)
    if run == 1:
     print(f'States after {crossovers} crossovers:')
     queens.display_states(10)
    time_solved = crossovers *1.33
      
    for i in range(len(queens.average_fitness)):
      averages[i] += queens.average_fitness[i]/num_runs
      bests[i] += queens.best_fitness[i]/num_runs
      if queens.best_fitness[i] == 28:
        time_solved = min(i,time_solved)
    iterations_to_solve.append(time_solved)
    print(f'Generations to solve for run {run} is {time_solved}')
    
  av_gens_for_solution = sum(iterations_to_solve)/len(iterations_to_solve)
  print(f'The average number of crossovers required to find a solution is: \
    {av_gens_for_solution}')
  plot_generations_to_solution(iterations_to_solve)
  plot_values(averages,bests)
