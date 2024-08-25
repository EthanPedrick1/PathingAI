# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

# define enviroment shape
environment_rows = 11
environment_columns = 11

#Creates a 3D numpy array to hold Q values
q_values = np.zeros((environment_rows, environment_columns, 4))
#4 is the number of actions the AI can take

#Define actions
#Action codes: 0=up, 1=right, 2=down, 3=left
actions = ['up', 'right', 'down', 'left']

#Rewards
#Create a 2D numpy array to hold rewards
#Array is 11x11 to match environment
rewards = np.full((environment_rows, environment_columns), -100.)
#Initializes all array values to -100
rewards[0, 5] = 100. #Sets goal to 100 points

#Define aisle locations for rows 1-9
aisles = {} #store locations in a dictionary
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1,10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

#Set the rewards for all aisle locations
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.
        
#Print rewards matrix
for row in rewards:
    print(row)

#Define a function that determines if the location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
    #if the reward is -1, its not terminal
    if rewards[current_row_index, current_column_index] == -1.:
        return False
    else:
        return True
    
#define a function that will choose a random, non-terminal starting location
def get_starting_location():
    #get a random row and column
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
    #continue choosing until its non-terminal
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index

#define epsilon greedy algorithim to choose the next action
def get_next_action(current_row_index, current_column_index, epsilon):
    #if random value between 0 and 1 is less than epsilon
    #then choose the next best value
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else: #choose random action
        return np.random.randint(4)
    
#define a function that gets the next location based on action
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns -1:
        new_row_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

#define a function that will get the shortest path between locations
def get_shortest_path(start_row_index, start_column_index):
    #return immediately if starting location is invalid
    if is_terminal_state(start_row_index, start_column_index):
        return[]
    else: #if this is a valid starting location
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        #continue moving until the goal is reached
        while not is_terminal_state(current_row_index, current_column_index):
            #get best action to take
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            #move to next location
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path
    
#define training paramaters
epsilon = 0.9 #the percentage of time when we shoudl take the best action
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which it should learn

#run through 1,000 training episodes
for episode in range(1000):
    #get starting location
    row_index, column_index = get_starting_location()
    
    #keep taking actions until terminal state
    while not is_terminal_state(row_index, column_index):
        #choose which action
        action_index = get_next_action(row_index, column_index, epsilon)
        
        #perform chosen action
        old_row_index, old_column_index = row_index, column_index #stores old position
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        
        #recieve reward
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        
        #update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value
        
print('Training Complete!')
        
#display a few shortest paths
print(get_shortest_path(5, 0)) #starting at row, column
