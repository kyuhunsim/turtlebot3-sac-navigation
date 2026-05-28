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
from torch.utils.tensorboard import SummaryWriter
import gym

args = config()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)

env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = SAC(args.state_dim, args.action_dim, args,action_space=None)
memory = ReplayMemory(args.replay_size)

runs_dir = os.path.join(dirPath, 'runs')
os.makedirs(runs_dir, exist_ok=True)
writer = SummaryWriter(os.path.join(runs_dir, '{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.world)))


def train():
    rospy.init_node('sac')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    rospy.loginfo('Start training')
    
    total_numsteps = 0
    updates = 0
    # start_episode = 0

    # if args.load_model:
    #     agent.load_models(args.load_episode, args)
    #     rospy.loginfo('Loaded model at episode %d', args.load_episode)
    #     start_episode = args.load_episode + 1
        
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()[0]

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy
            
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size,updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            done = terminated or truncated


            state = next_state 
        # total_numsteps는 지금 0~ num_steps까지 쭉 진행, num_steps는 지금 50000으로 설정되어있음.
        if total_numsteps > args.num_steps:
            print('End of training')
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
        #i_episode는 1부터 시작함. for문이 돈 횟수를 의미함.

        if i_episode % 10 == 0 and args.eval is True:
            #for문이 10번 돌때마다 test를 진행함. 그런데 eval이 True이어야 함.
            #그런데 eval이 True이면 항상 그런거 아닌가??
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()[0]
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes
            


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
        #여기는 1000단위로 출력되는 곳임.

        if i_episode % 10 == 0:
            result.data = episode_reward
            pub_result.publish(result)

        if i_episode % 20 == 0:
            agent.save_models(i_episode, args)

    env.close()


if __name__ == '__main__':
    train()
