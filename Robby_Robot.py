'''
Jared Goldsmith
jgolds@pdx.edu
AI - Rhoades
Program 3
Solves Robby Robot problem using q-learning and q-matrix
'''
import numpy as np
import matplotlib.pyplot as plt
import statistics

class Robby_Robot(object):
  def __init__(self, epsilon = 0.1, learning_rate = 0.2, gamma = 0.9):
    self.epsilon = epsilon
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.q_matrix = {}

  # Creates 10x10 grid with half squares with a can
  def create_grid(self):
    grid = np.zeros((12,12))
    can_set = set()
    while len(can_set) < 50:
      row = np.random.randint(1,11)
      col = np.random.randint(1,11)
      can_set.add((row,col))
      grid[row,col] = 1
    for i in range(len(grid)):
      for j in range(len(grid[0])):
        if i == 0 or i == 11 or \
          j == 0 or j == 11:
          grid[i][j] = -1
    return grid

  # Sensor Readings

  def current_sensor(self):
    return self.grid[self.row][self.col]

  def north_sensor(self):
    return self.grid[self.row][self.col+1]

  def south_sensor(self):
    return self.grid[self.row][self.col-1]

  def east_sensor(self):
    return self.grid[self.row+1][self.col]

  def west_sensor(self):
    return self.grid[self.row-1][self.col]

  # Returns the sensor reading for current position
  def current_senses(self):
    current_state = (self.current_sensor(), self.north_sensor(), \
                     self.south_sensor(), self.east_sensor(), \
                     self.west_sensor())
    return current_state
  
  # Returns the index of max q value for current state
  def choose_action(self, current_state):
    random_val = np.random.randint(100)/100
    if random_val < self.epsilon:
      return np.random.randint(5)
    
    max_val = float('-infinity')
    max_index = 0
    max_indices = []
    
    for i in range(5):
      if self.q_matrix[current_state][i] > max_val:
        max_val = self.q_matrix[current_state][i]
        max_index = i
    return max_index
      
  # Available Actions

  def pick_up(self):
    if self.grid[self.row][self.col] == 1:
      self.grid[self.row][self.col] = 0
      return True
    return False

  def move_north(self):
    if self.north_sensor() == -1:
      return False
    self.col += 1
    return True
  
  def move_south(self):
    if self.south_sensor() == -1:
      return False
    self.col -= 1
    return True

  def move_east(self):
    if self.east_sensor() == -1:
      return False
    self.row += 1
    return True

  def move_west(self):
    if self.west_sensor() == -1:
      return False
    self.row -= 1
    return True

  # Performs action after choosing projected best action
  def perform_action(self, action):
    if action == 0:
      if self.pick_up():
        return 10
      return -1
    elif action == 1:
      if self.move_north():
        return 0
      return -5
    elif action == 2:
      if self.move_south():
        return 0
      return -5
    elif action == 3:
      if self.move_east():
        return 0
      return -5
    elif action == 4:
      if self.move_west():
        return 0
      return -5
    
  # Episode for training. Updates q-matrix after action
  def train_episode(self, num_reps):
    for i in range(num_reps):
      current_state = self.current_senses()
      if current_state not in self.q_matrix:
        self.q_matrix[current_state] = np.zeros(5)
      best_action = self.choose_action(current_state)
      reward = self.perform_action(best_action)
      self.reward += reward
      next_state = self.current_senses()
      
      if next_state not in self.q_matrix:
        self.q_matrix[next_state] = np.zeros(5)
      current_q_val = self.q_matrix[current_state][best_action]
      temporal_difference = reward + (self.gamma*max(self.q_matrix[next_state]) - current_q_val)
      next_q_val = current_q_val + self.learning_rate*temporal_difference
      self.q_matrix[current_state][best_action] = next_q_val
      
  # Runs each episode, plots rewards, gets average and std dev
  def training(self, num_episodes, num_reps):
    rewards = []
    average_reward = []
    total_reward = 0
    all_rewards = []
    for i in range(num_episodes):
      self.grid = self.create_grid()
      self.row = np.random.randint(1,11)
      self.col = np.random.randint(1,11)
      self.reward = 0
      self.train_episode(num_reps)
      total_reward += self.reward
      average_reward.append(total_reward/(i+1))
      all_rewards.append(self.reward)
      if i % 50 == 0:
        self.epsilon -= 0.007
        if self.epsilon < 0:
          self.epsilon = 0
      if i % 100 == 0:
        rewards.append(self.reward)
    
    # Plots average reward
    fig, ax = plt.subplots()
    ax.plot(average_reward)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    plt.show()
    print('\n')
    # Plots reward for every 100 episodes
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    plt.show()
    # Calculates and displays average and standard deviation
    average_rewards = total_reward / num_episodes
    standard_dev = statistics.stdev(all_rewards)
    print(f'Average Reward while training is: {average_rewards}')
    print(f'The Standard Deviation for training is: {standard_dev}')

  # Episode for testing. Chooses action but doesn't update q-matrix
  def test_episode(self, num_runs):
    for i in range(num_runs):
      current_state = self.current_senses()
      action = self.choose_action(current_state)
      reward = self.perform_action(action)
      self.reward += reward
  
  # Goes through each episode for testing, plots rewards, calculates 
  # average and standard deviation
  def testing(self, num_episodes, num_runs):
    rewards = []
    average_rewards = []
    all_rewards = []
    total_reward = 0

    for i in range(num_episodes):
      self.grid = self.create_grid()
      self.row = np.random.randint(1,11)
      self.col = np.random.randint(1,11)
      self.reward = 0
      self.test_episode(num_runs)
      total_reward += self.reward
      all_rewards.append(self.reward)
      if i % 100 == 0:
        rewards.append(self.reward)
      average_rewards.append(total_reward/(i+1))
    
    # Plots average rewards
    fig, ax = plt.subplots()
    ax.plot(average_rewards)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    plt.show()
    print('\n')
    # Plots reward for every 100 episodes
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    plt.show()
    # Calculates average reward and standard deviation
    average_reward = sum(all_rewards)/num_episodes
    standard_dev = statistics.stdev(all_rewards)
    print(f'Average reward for testing is: {average_reward}')
    print(f'The standard deviation for testing data is: {standard_dev}')
    
if __name__ == '__main__':
    
  #Parameters 
  epsilon = 0.1
  learning_rate = 0.2
  discount_rate = 0.9
  num_episodes = 5000
  num_runs = 200

  robot = Robby_Robot(epsilon, learning_rate, discount_rate)
  robot.training(num_episodes, num_runs)
  robot.testing(num_episodes, num_runs)
