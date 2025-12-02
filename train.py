import argparse
import numpy as np
import torch
import gymnasium as gym

from buffer import ReplayBuffer
from network import TD3_train, Actor, Critic

## argparse
"""
TD3 학습을 위한 하이퍼파라미터 및 실험 설정

주요 파라미터:
    - env: 학습할 OpenAI Gym 환경
    - seed: 재현성을 위한 랜덤 시드
    - start_timesteps: 초기 random exploration 스텝 수 (policy의 초기 파라미터에 대한 의존성을 제거하기 위함)
    - eval_frequency: 평가 주기
    - max_timesteps: 총 학습 스텝 수
    - exploration_noise: exploration을 위한 gaussian noise std.
    - batch_size: mini-batch size
    - discount: discount factor gamma
    - tau: target newtork soft update 비율
    - policy_noise: target policy smoothing noise
    - noise_clip: target policy noise clipping 범위
    - policy_frequency: policy delayed update frequency
"""

parser = argparse.ArgumentParser()

parser.add_argument("--env", default="HalfCheetah-v4")        
parser.add_argument("--seed", default=0, type=int)        
parser.add_argument("--start_timesteps", default=10000, type=int)
parser.add_argument("--eval_frequency", default=5000, type=int)
parser.add_argument("--max_timesteps", default=1000000, type=int)
parser.add_argument("--exploration_noise", default=0.1, type=float)  
parser.add_argument("--batch_size", default=256, type=int)     
parser.add_argument("--discount", default=0.99, type=float)   
parser.add_argument("--tau", default=0.005, type=float)       
parser.add_argument("--policy_noise", default=0.2)      
parser.add_argument("--noise_clip", default=0.5)
parser.add_argument("--policy_frequency", default=2, type=int)
parser.add_argument("--model_save_path", default="./results/01_HalfCheetah/01_TD3_Best_Params.pth")
parser.add_argument("--model_load_path", default=None)

args = parser.parse_args()

## policy, env 알림 출력
print("---------------------------------------")
print(f"Policy: TD3, Env: {args.env}, Seed: {args.seed}")
print("---------------------------------------")

## 하이퍼파라미터 스페이스 저장
np.save(args.model_save_path.replace(".pth", "_param_dict.npy"), vars(args))

## policy evaluation 함수 (gymnasium의 기본 설정에 따른 return값 계산)
def eval_policy(policy, env_name, seed, eval_episodes=10):
    """
    학습된 정책을 평가하여 평균 누적 보상을 계산
    
    Args:
        policy: 평가할 정책 (Actor 네트워크)
        env_name (str): OpenAI Gym 환경 이름
        seed (int): 재현성을 위한 랜덤 시드
        eval_episodes (int): 평가할 에피소드 수 (기본값: 10)
        
    Returns:
        float: 평가 에피소드들의 평균 누적 보상
    """
    eval_env = gym.make(env_name)
    avg_reward = 0.
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset(seed=seed+1000)
        terminated = False
        truncated = False
        
        while not (terminated or truncated):  # 둘 중 하나라도 True면 종료
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward
    
    avg_reward /= eval_episodes
    
    print("########################################")
    print(f"Average return Evaluation results for {eval_episodes} episodes: {avg_reward:.3f}")
    print("########################################")
    
    return avg_reward

## policy, env 초기화
"""
env. 및 TD3 policy 초기화, replay buffer 생성

1. Gym env 생성 및 random seed 설정 (재현성 확보)
2. state/action space 차원 및 action 범위 추출
3. TD3 policy network 초기화 만약, 학습된 actor, crtic weight가 있다면 load_state_dict
4. 학습 데이터 저장을 위한 replay buffer 생성
"""

# env. 생성
env = gym.make(args.env)

# 랜덤 시드 설정 (재현성 확보)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# env 정보 추출
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

# actor, critic network 로드
actor_network = Actor(state_dim, action_dim, max_action)
critic_network = Critic(state_dim, action_dim)

# 학습된 weight load
if args.model_load_path is not None:
    actor_network.load_state_dict(torch.load(args.model_load_path.replace(".pth", "_actor.pth")))
    critic_network.load_state_dict(torch.load(args.model_load_path.replace(".pth", "_critic.pth")))

# TD3 policy 초기화
policy = TD3_train(actor_network=actor_network,
                   critic_network=critic_network,
                   state_dim=state_dim,
                   action_dim=action_dim,
                   max_action=max_action,
                   discount=args.discount,
                   tau=args.tau,
                   policy_noise=args.policy_noise * max_action,
                   noise_clip=args.noise_clip * max_action,
                   policy_frequency=args.policy_frequency)


# replay buffer 생성
replay_buffer = ReplayBuffer(state_dim, action_dim)

## random(학습되지 않은) policy의 average return 계산 후 리스트에 append
evaluations = [eval_policy(policy, args.env, args.seed)]

## 학습
"""
TD3 학습 루프

전체 프로세스:
1. 초기 random exploration으로 replay buffer add (start_timesteps)
2. exploration noise를 추가한 policy로 action 선택
3. env와 상호작용하여 경험 수집
4. replay buffer에서 샘플링하여 TD3 학습
5. 주기적으로 policy 성능 평가
"""

# 에피소드 초기화
state, _ = env.reset(seed=args.seed)
terminated, truncated = False, False

episode_reward = 0
episode_timesteps = 0
episode_num = 0

# 메인 학습 루프
for t in range(int(args.max_timesteps)):
    episode_timesteps += 1

    # 행동 선택 (Action Selection)
    if t < args.start_timesteps:
        # 초기 랜덤 탐색: 리플레이 버퍼를 다양한 경험으로 채우기
        action = env.action_space.sample()
    else:
        # 정책 기반 행동 + 탐색 노이즈
        action = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.exploration_noise, size=action_dim)).clip(-max_action, max_action)
        
    # 환경에서 행동 수행 (Environment Step)
    next_state, reward, terminated, truncated, _ = env.step(action)

    # 리플레이 버퍼에 경험 저장
    replay_buffer.add(state, action, next_state, reward, float(truncated))

    # 상태 업데이트
    state = next_state
    episode_reward += reward

    # 정책 학습 (Policy Training)
    if t >= args.start_timesteps:
        # 충분한 데이터가 수집된 후 학습 시작
        policy.train(replay_buffer, args.batch_size)

    # 에피소드 종료 처리 (Episode Termination)
    if truncated: # done 대신 truncated 사용 (Gym 0.26+ API)
        # 에피소드 정보 출력
        print(f"Total Iterations: {t+1} Episode No.: {episode_num+1} Episode Iterations: {episode_timesteps} Rewards: {episode_reward:.3f}")

        # 에피소드 리셋
        state, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        
    # 정책 평가 (Policy Evaluation) 후 모델 및 return 히스토리 저장
    if (t + 1) % args.eval_frequency == 0:
        evaluations.append(eval_policy(policy, args.env, args.seed))

        # return 히스토리 저장
        np.save(args.model_save_path.replace(".pth", "_average_return_list.npy"), evaluations)

        # network 저장
        torch.save(actor_network.state_dict(), args.model_save_path.replace(".pth", "_actor.pth"))
        torch.save(critic_network.state_dict(), args.model_save_path.replace(".pth", "_critic.pth"))