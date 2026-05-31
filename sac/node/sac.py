#!/usr/bin/env python3

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from network import PolicyNetwork, QNetwork
import numpy as np
import default as config


dirPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

class SAC(object):
    def __init__(self, state_dim, action_dim, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.lr=args.lr

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.critic = QNetwork(state_dim, action_dim, args.hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        self.critic_target = QNetwork(state_dim, action_dim, args.hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        print('entropy', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        

        self.policy = PolicyNetwork(state_dim, action_dim, args.hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        
        
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]
        return action

  
    def update_parameters(self, memory, batch_size,updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # 
        qf2_loss = F.mse_loss(qf2, next_q_value) # 
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_models(self, episode_count,args):
        model_dir = os.path.join(dirPath, 'SAC_model', args.world)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(model_dir, str(episode_count) + '_policy_net.pth'))
        torch.save(self.critic.state_dict(), os.path.join(model_dir, str(episode_count) + 'value_net.pth'))
        # hard_update(self.critic_target, self.critic)
        # torch.save(soft_q_net.state_dict(), dirPath + '/SAC_model/' + world + '/' + str(episode_count)+ 'soft_q_net.pth')
        # torch.save(target_value_net.state_dict(), dirPath + '/SAC_model/' + world + '/' + str(episode_count)+ 'target_value_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")
    
    # Load model parameters
    def load_models(self, episode,args):
        dir_path = os.path.join(dirPath, 'SAC_model', args.world)
        policy_path = os.path.join(dir_path, f'{episode}_policy_net.pth')
        critic_path = os.path.join(dir_path, f'{episode}value_net.pth')

        if not os.path.exists(policy_path) or not os.path.exists(critic_path):
            dir_path = os.path.join(dirPath, os.pardir, 'pretrained', args.world)
            policy_path = os.path.join(dir_path, f'{episode}_policy_net.pth')
            critic_path = os.path.join(dir_path, f'{episode}value_net.pth')

        if not os.path.exists(policy_path) or not os.path.exists(critic_path):
            raise FileNotFoundError(
                f'Model checkpoint not found for {args.world} episode {episode}'
            )

        print(f'Loading models from {os.path.abspath(dir_path)}')
        self.policy.load_state_dict(torch.load(policy_path))
        self.critic.load_state_dict(torch.load(critic_path))
        hard_update(self.critic_target, self.critic)
        # soft_q_net.load_state_dict(torch.load(dirPath + '/SAC_model/' + world + '/'+str(episode)+ 'soft_q_net.pth'))
        # target_value_net.load_state_dict(torch.load(dirPath + '/SAC_model/' + world + '/'+str(episode)+ 'target_value_net.pth'))
        print('***Models load***')
