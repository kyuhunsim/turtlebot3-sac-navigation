#!/usr/bin/env python3

class config:
    # env_name="HalfCheetah-v4"
    eval = True
    gamma = 0.99
    tau = 0.01
    lr = 0.0003
    alpha = 0.2
    # seed=123456
    automatic_entropy_tuning = True
    batch_size = 256
    hidden_dim = 256 #hidden_state
    updates_per_step = 1
    start_steps = 2000
    replay_size = 50000
    cuda = False
    save_model = './model'
    load_model = True
    rewards=[]
    max_steps=500
    num_steps=max_steps
    action_dim=2
    # action_dim=6
    # state_dim=17
    state_dim=30
    ACTION_V_MIN = 0.0 # m/s
    ACTION_W_MIN = -2. # rad/s
    ACTION_V_MAX = 0.22 # m/s
    ACTION_W_MAX = 2. # rad/s
    is_training = True
    world = 'stage_4'
    max_episodes=2510
    max_episode=max_episodes
    load_episode = 2500
    target_update_interval = 1
