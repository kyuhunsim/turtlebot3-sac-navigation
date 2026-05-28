#!/usr/bin/env python3

import rospy
import os
import numpy as np
import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from utils import action_unnormalized
from environment_stage_1 import Env
from buffer import ReplayBuffer
from sac import SAC
from collections import deque
import copy
from default import config


from torch.utils.tensorboard import SummaryWriter

args = config()

#---Directory Path---#
dirPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

runs_dir = os.path.join(dirPath, 'runs')
os.makedirs(runs_dir, exist_ok=True)
writer = SummaryWriter(os.path.join(runs_dir, '{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.world)))
#****************************************************

agent = SAC(args.state_dim, args.action_dim, args)
replay_buffer = ReplayBuffer(args.replay_size)

print('State Dimensions: ' + str(args.state_dim))
print('Action Dimensions: ' + str(args.action_dim))
print('Action Max: ' + str(args.ACTION_V_MAX) + ' m/s and ' + str(args.ACTION_W_MAX) + ' rad/s')
print('Action Min: ' + str(args.ACTION_V_MIN) + ' m/s and ' + str(args.ACTION_W_MIN) + ' rad/s')


if __name__ == '__main__':
    rospy.init_node('sac')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()
    before_training = 4
    past_action = np.array([0.,0.])
    updates=0
    start_episode=0
    #load model
    if args.load_model:
        agent.load_models(args.load_episode, args)
        start_episode = args.load_episode

    for ep in range(start_episode, args.max_episodes):
        done = False
        state = env.reset()
        
        if args.is_training and not ep%10 == 0 and len(replay_buffer) > before_training*args.batch_size:
            print('Episode: ' + str(ep) + ' training')
        else:
            if len(replay_buffer) > before_training*args.batch_size:
                print('Episode: ' + str(ep) + ' evaluating')
            else:
                print('Episode: ' + str(ep) + ' adding to memory')

        rewards_current_episode = 0.

        for step in range(args.max_steps):
            state = np.float32(state)
            # print('state___', state)
            if args.is_training and not ep%10 == 0:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state, eval=True)

            if not args.is_training:
                action = agent.select_action(state, eval=True)

            unnorm_action = np.array([action_unnormalized(action[0], args.ACTION_V_MAX, args.ACTION_V_MIN), 
                                      action_unnormalized(action[1], args.ACTION_W_MAX, args.ACTION_W_MIN)])

            next_state, reward, done = env.step(unnorm_action, past_action)
            # print('action', unnorm_action,'r',reward)
            past_action = copy.deepcopy(action)

            rewards_current_episode += reward
            next_state = np.float32(next_state)
            if not ep%10 == 0 or not len(replay_buffer) > before_training*args.batch_size:
                if reward == 100.:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        replay_buffer.push(state, action, reward, next_state, done)
                else:
                    replay_buffer.push(state, action, reward, next_state, done)
            
            if len(replay_buffer) > before_training*args.batch_size and args.is_training and not ep% 10 == 0:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(replay_buffer, args.batch_size,updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

            state = copy.deepcopy(next_state)

            if done:
                break
        print('reward per ep: ' + str(rewards_current_episode))
        print('reward average per ep: ' + str(rewards_current_episode) + ' and break step: ' + str(step))
        writer.add_scalar('reward/train', rewards_current_episode, ep)
        
        if ep%10 == 0:
            if len(replay_buffer) > before_training*args.batch_size:
                result = rewards_current_episode
                pub_result.publish(result)
        
        if ep%20 == 0:
            agent.save_models(ep,args)
