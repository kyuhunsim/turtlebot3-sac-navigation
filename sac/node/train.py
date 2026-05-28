#!/usr/bin/env python3

import rospy
import numpy as np
import copy
import datetime
import os
from std_msgs.msg import Float32
from default import config
from sac import SAC
from buffer import ReplayMemory
import sys
dirPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dirPath)
from src.env.environment_stage_1 import Env
from utils import action_unnormalized
from tensorboardX import SummaryWriter

args = config()

agent = SAC(args.state_dim, args.action_dim, args)
memory = ReplayMemory(args.replay_size)

#Tesnorboard
runs_dir = os.path.join(dirPath, 'runs')
os.makedirs(runs_dir, exist_ok=True)
writer = SummaryWriter(os.path.join(runs_dir, '{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.world)))


if __name__ == '__main__':
    rospy.init_node('sac')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()
    before_training = 4
    past_action = np.array([0.0, 0.0])
    rospy.loginfo('Start training')
    
    start_episode = 0

    if args.load_model:
        agent.load_models(args.load_episode,args)
        rospy.loginfo('Loaded model at episode %d', args.load_episode)
        start_episode = args.load_episode + 1

    for ep in range(start_episode, args.max_episode):
        done = False
        state = env.reset()
        
        training_ready = len(memory) > before_training * args.batch_size
        rospy.loginfo(f'Memory size: {len(memory)}, Training ready: {training_ready}')

        if args.is_training and ep % 10 != 0 and training_ready:
            rospy.loginfo('Episode: %d training', ep)
        elif training_ready:
            rospy.loginfo('Episode: %d evaluating', ep)
        else:
            rospy.loginfo('Episode: %d adding to memory', ep)

        rewards_current_episode = 0.0

        for step in range(args.num_steps):
            state = np.float32(state)

            if args.is_training and ep % 10 != 0:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state, evaluate=True)

            # unnorm_action = np.array([
            #     action_unnormalized(action[0], args.ACTION_V_MAX, args.ACTION_V_MIN),
            #     action_unnormalized(action[1], args.ACTION_W_MAX, args.ACTION_W_MIN)
            # ])
            next_state, reward, done = env.step(action, past_action)
            past_action = copy.deepcopy(action)
            # print("action",action,"reward",reward)

            rewards_current_episode += reward
            next_state = np.float32(next_state)

            if not (ep % 10 == 0 and training_ready):
                memory.push(state, action, reward, next_state, done)

            if training_ready and args.is_training and ep % 10 != 0:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size)
                writer.add_scalar('loss/critic_1', critic_1_loss)
                writer.add_scalar('loss/critic_2', critic_2_loss)
                writer.add_scalar('loss/policy', policy_loss)
                writer.add_scalar('loss/entropy_loss', ent_loss)
                writer.add_scalar('entropy_temprature/alpha', alpha)
            
            state = copy.deepcopy(next_state)

            if done:
                break

        rospy.loginfo(f'Episode: {ep}, reward: {rewards_current_episode}') 
        writer.add_scalar('Reward/episode', rewards_current_episode, ep)

        if ep % 10 == 0 and training_ready:
            result.data = rewards_current_episode
            pub_result.publish(result)
        
        if ep % 20 == 0:
            agent.save_models(ep, args)

writer.close()
