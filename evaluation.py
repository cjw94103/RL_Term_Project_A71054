import argparse
import numpy as np
import torch
import gymnasium as gym

from buffer import ReplayBuffer
from network import TD3_train, Actor, Critic

## argparse
parser = argparse.ArgumentParser()

parser.add_argument("--env", default="HalfCheetah-v4")        
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--render_mode", default=None)
parser.add_argument("--trained_actor_path", default="./trained_model/01_HalfCheetah/TD3_Ours_actor.pth")

args = parser.parse_args()

## policy evaluation 함수 (gymnasium의 기본 설정에 따른 return값 계산)
def eval_policy(policy, env_name, seed, render_mode=None):
    
    eval_env = gym.make(env_name, render_mode=render_mode)

    state, _ = eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed) 
    eval_env.observation_space.seed(seed)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    terminated = False
    truncated = False
    
    return_record = 0.

    while not (terminated or truncated): 
        action = policy.select_action(np.array(state))
        state, reward, terminated, truncated, _ = eval_env.step(action)
        return_record += reward
        
    return return_record

# env. 생성
env = gym.make(args.env)

# env 정보 추출
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# actor, critic network 로드
actor_network = Actor(state_dim, action_dim, max_action)
critic_network = Critic(state_dim, action_dim)

actor_network.load_state_dict(torch.load(args.trained_actor_path))

# TD3 policy 초기화
policy = TD3_train(actor_network=actor_network,
                   critic_network=critic_network,
                   state_dim=state_dim,
                   action_dim=action_dim,
                   max_action=max_action,
                   discount=None,
                   tau=None,
                   policy_noise=None,
                   noise_clip=None,
                   policy_frequency=None)

avg_return = 0
for i in range(10):
    random_seed = int(np.random.choice(1000, 1)[0] + 1)
    total_return = eval_policy(policy, args.env, random_seed, render_mode=args.render_mode)
    avg_return += total_return
    
avg_return = avg_return/10

print("########################################")
print(f"Average return for 10 episodes and 10 random seeds: {avg_return:.3f}")
print("########################################")