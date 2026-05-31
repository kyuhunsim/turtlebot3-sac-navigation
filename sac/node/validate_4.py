#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
import csv
import time
from utils import action_unnormalized
from environment_stage_5 import Env
from sac import SAC
import copy
from default import config

args = config()

#---Directory Path---#
dirPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

agent = SAC(args.state_dim, args.action_dim, args)

rospy.loginfo('State Dimensions: ' + str(args.state_dim))
rospy.loginfo('Action Dimensions: ' + str(args.action_dim))
rospy.loginfo('Action Max: ' + str(args.ACTION_V_MAX) + ' m/s and ' + str(args.ACTION_W_MAX) + ' rad/s')
rospy.loginfo('Action Min: ' + str(args.ACTION_V_MIN) + ' m/s and ' + str(args.ACTION_W_MIN) + ' rad/s')


def load_model_and_test():
    agent.load_models(args.load_episode, args)
    rospy.loginfo('Model loaded')

    # Initialize environment
    env = Env()
    past_action = np.array([0., 0.])

    ratios = []
    goal_reached_count = 0
    total_episodes = 100  # Number of successful goal reaches to consider
    episode_count = 0

    state = env.reset()
    start_position = env.get_robot_position()  # Initial position.

    csv_dir = os.path.join(dirPath, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(csv_dir, 'test_results_9.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['start_point', 'goal_point', 'path_length', 'time', 'path','goal_reached'])

        while goal_reached_count < total_episodes:
            rospy.loginfo(f'Starting episode {episode_count + 1}/{total_episodes}')
            episode_count += 1
            done = False
            start_time = time.time()  # Record the start time.

            path = [env.get_robot_position()]  # Record the initial position.

            while not done:
                state = np.float32(state)
                action = agent.select_action(state, eval=True)

                unnorm_action = np.array([action_unnormalized(action[0], args.ACTION_V_MAX, args.ACTION_V_MIN), 
                                          action_unnormalized(action[1], args.ACTION_W_MAX, args.ACTION_W_MIN)])

                next_state, reward, done = env.step(unnorm_action, past_action)
                past_action = copy.deepcopy(action)

                path.append(env.get_robot_position())  # Record the position after each step.
                state = copy.deepcopy(next_state)

                rospy.loginfo(f'Step Reward: {reward}, Done: {done}')

                if reward == 100:
                    goal_reached_count += 1
                    goal_position = env.get_robot_position()  # Get the goal position.
                    straight_line_distance = distance.euclidean(start_position, goal_position)
                    path_distance = sum(distance.euclidean(path[i], path[i+1]) for i in range(len(path) - 1))
                    ratio = path_distance / straight_line_distance
                    ratios.append(ratio)
                    elapsed_time = time.time() - start_time  # Record elapsed time.

                    rospy.loginfo(f'Goal reached {goal_reached_count}/{episode_count}: Ratio: {ratio}')

                    # Write the result to the CSV file.
                    writer.writerow([
                        (round(start_position[0], 3), round(start_position[1], 3)), 
                        (round(goal_position[0], 3), round(goal_position[1], 3)), 
                        round(path_distance, 3), 
                        round(elapsed_time, 3), 
                        [(round(p[0], 3), round(p[1], 3)) for p in path], True
                    ])
                    
                    state = copy.deepcopy(next_state)
                    start_position = env.get_robot_position()  # Use the current position as the next start position.
                    break  # End the episode because the goal was reached.

                elif reward == -10:
                    rospy.loginfo(f'Collision detected, resetting environment.')
                    path_distance = sum(distance.euclidean(path[i], path[i+1]) for i in range(len(path) - 1))
                    elapsed_time = time.time() - start_time

                    writer.writerow([
                        (round(start_position[0], 3), round(start_position[1], 3)), 
                        (None, None),  # Goal position is None since it wasn't reached
                        round(path_distance, 3), 
                        round(elapsed_time, 3), 
                        [(round(p[0], 3), round(p[1], 3)) for p in path],
                        False
                    ])
                    state = env.reset()  # Reset to the initial position.
                    start_position = env.get_robot_position()  # Set the initial start position.
                    break  # Exit the loop and start a new episode.

    avg_ratio = np.mean(ratios)
    rospy.loginfo(f'Average Ratio after {total_episodes} goal reaches: {avg_ratio}')

if __name__ == '__main__':
    rospy.init_node('sac_test')
    load_model_and_test()
