#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
from utils import action_unnormalized
from environment_stage_1 import Env
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
    # Load model
    if args.load_model:
        agent.load_models(args.load_episode, args)
        rospy.loginfo('Model loaded')

    # Initialize environment
    env = Env()
    past_action = np.array([0., 0.])

    ratios = []
    goal_reached_count = 0
    total_episodes = 100  # Number of successful goal reaches to consider
    episode_count=0

    while goal_reached_count < total_episodes:
        episode_count+=1
        rospy.loginfo(f'Starting episode {goal_reached_count + 1}/{total_episodes}')
        done = False
        state = env.reset()
        start_position = env.get_robot_position()  # Assuming this function returns the starting position
        path = [start_position]

        while not done:
            state = np.float32(state)
            action = agent.select_action(state, eval=True)

            unnorm_action = np.array([action_unnormalized(action[0], args.ACTION_V_MAX, args.ACTION_V_MIN), 
                                      action_unnormalized(action[1], args.ACTION_W_MAX, args.ACTION_W_MIN)])

            next_state, reward, done = env.step(unnorm_action, past_action)
            past_action = copy.deepcopy(action)

            path.append(env.get_robot_position())  # Record the position after each step
            state = copy.deepcopy(next_state)

            if reward ==100:
                goal_reached_count += 1
                goal_position = env.get_robot_position()
                 # Assuming this function returns the goal position
                straight_line_distance = distance.euclidean(start_position, goal_position)
                path_distance = sum(distance.euclidean(path[i], path[i+1]) for i in range(len(path) - 1))
                ratio = path_distance / straight_line_distance
                ratios.append(ratio)
                    
                rospy.loginfo(f'Goal reached {goal_reached_count}/{episode_count}: Ratio: {ratio}')
                break  # Move on to the next episode whether goal was reached or not
            
            elif reward == -10:
                rospy.loginfo(f'Goal not reached ')
                done = True
                break

    avg_ratio = np.mean(ratios)
    rospy.loginfo(f'Average Ratio after {total_episodes} goal reaches: {avg_ratio}')

if __name__ == '__main__':
    rospy.init_node('sac_test')
    load_model_and_test()
