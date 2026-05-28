#!/usr/bin/env python3
import torch
import rospy
import numpy as np
import copy
import datetime
import os
from std_msgs.msg import Float32
from default import config
import itertools
from sac import SAC
from buffer import ReplayMemory
import sys

dirPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dirPath)
from src.env.environment_stage_1 import Env

from torch.utils.tensorboard import SummaryWriter

args = config()


agent = SAC(args.state_dim, args.action_dim, args)
memory = ReplayMemory(args.replay_size)


runs_dir = os.path.join(dirPath, 'runs')
os.makedirs(runs_dir, exist_ok=True)
writer = SummaryWriter(os.path.join(runs_dir, '{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.world)))


def train():
    rospy.init_node('sac')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    past_action = np.array([0.0, 0.0])
    rospy.loginfo('Start training')
    env=Env()

    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and episode_steps < args.num_steps:
            if args.start_steps > total_numsteps:
                action_v = np.random.uniform(low=args.ACTION_V_MIN, high=args.ACTION_V_MAX)  # Sample random v
                action_w = np.random.uniform(low=args.ACTION_W_MIN, high=args.ACTION_W_MAX)  # Sample random w
                action = np.array([action_v, action_w])  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            
            next_state, reward, done = env.step(action, past_action)  # Step
            past_action = copy.deepcopy(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            memory.push(state, action, reward, next_state, float(not done)) # Append transition to memory

            state = next_state 
        

        rospy.loginfo(f'Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {episode_reward}')
        writer.add_scalar('reward/train', episode_reward, i_episode)


        if i_episode % 10 == 0:
            result.data = episode_reward
            pub_result.publish(result)

        if i_episode % 20 == 0:
            agent.save_models(i_episode, args)
    
    
    writer.close()
    env.close()


if __name__ == '__main__':
    train()
