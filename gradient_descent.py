import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
%matplotlib inline

def plot_values(greedy_reward, epsilon_reward):
  fig = plt.figure(figsize=[10,3])
  ax2 = fig.add_subplot(121)
  ax2.plot(greedy_reward, label='x values')
  ax2.plot(epsilon_reward, label='y values')
  ax2.set_xlabel('Timestep')
  ax2.set_ylabel('Average Reward')
  ax2.legend()
  
  
def equation(x,y):
    return 5*x**2 + 40*x + y**2 - 12*y + 127

def derivative_of_x(x):
    return 10*x + 40

def derivative_of_y(y):
    return 2*y - 12

def gradient_descent(x,y,learning_rate):
    z = equation(x,y)
    new_x = x - learning_rate*derivative_of_x(x)
    new_y = y - learning_rate*derivative_of_y(y)
    return new_x,new_y

if __name__ == '__main__':
    # Parameters
    num_tests = 10
    num_steps = 20000
    total_val = 0,0,0
    learning_rates = [.1,.01,.001]

    for learning_rate in learning_rates:
        total_x, total_y, total_z = 0,0,0
        x_values = [0]*(num_steps+1)
        y_values = [0]*(num_steps+1)
        print(f'\nFor learning rate {learning_rate}')
        for i in range(num_tests):
            x = random.randint(-10,10)
            y = random.randint(-10,10)
            x_values[0] += x/num_tests
            y_values[0] += y/num_tests
            for i in range(1, num_steps + 1):
                x,y = gradient_descent(x,y,learning_rate)
                x_values[i] += x/num_tests
                y_values[i] += y/num_tests
            total_x += x
            total_y += y
            total_z += equation(x,y)
        print(f'Average minimum value for f(x,y) is {total_z/num_tests}\
            \nwith values of x = {total_x/num_tests} and y = {total_y/num_tests}\n')
        plot_values(x_values,y_values)
