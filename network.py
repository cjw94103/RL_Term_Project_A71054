import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np

# pytorch 디바이스(GPU) 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    
    """
    TD3 Actor 네트워크: 연속적인 행동 공간에서 결정론적 정책을 학습
    
    주어진 상태에서 최적의 행동을 출력하는 신경망입니다.
    Tanh 활성화 함수를 사용하여 [-max_action, max_action] 범위의 행동을 생성합니다.
    """
    
    def __init__(self, state_dim, action_dim, max_action):

        """
        Actor 네트워크 초기화
        
        Args:
            state_dim (int): state space의 차원
            action_dim (int): action space의 차원
            max_action (float): action의 최대 절댓값 (행동 범위: [-max_action, max_action])
        """
        
        super(Actor, self).__init__()
        
        # 은닉층 (Hidden layers)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # 출력층 (Output layer)
        self.output = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):

        """
        순전파: 상태를 입력받아 행동을 출력
        
        Args:
            state (torch.Tensor): 입력 상태 (batch_size, state_dim)
            
        Returns:
            torch.Tensor: 선택된 행동 (batch_size, action_dim)
                         범위: [-max_action, max_action]
        """
        
        # 특징 추출 (Feature extraction)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Tanh activation으로 행동 출력 [-1, 1] 범위
        action = torch.tanh(self.output(x))

        # max_action으로 스케일링하여 실제 행동 범위로 변환
        return self.max_action * action

class Critic(nn.Module):

    """
    TD3 Critic 네트워크: Clipped Double Q-learning을 위한 Twin Q-networks
    
    overestimation bias을 완화하기 위해 두 개의 독립적인 Q-network를 사용
    각 Q-network는 (state, action) pair을 입력받아 Q 값을 출력
    """
    
    def __init__(self, state_dim, action_dim):

        """
        Critic 네트워크 초기화
        
        Args:
            state_dim (int): state space의 차원
            action_dim (int): action space의 차원
        """
        
        super(Critic, self).__init__()
        
        # 첫 번째 Q 네트워크 (First Q-network)
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_output = nn.Linear(256, 1)
        
        # 두 번째 Q 네트워크 (Second Q-network)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_output = nn.Linear(256, 1)
    
    def forward(self, state, action):

        """
        순전파: state와 action을 입력받아 두 Q 값을 모두 출력
        
        Args:
            state (torch.Tensor): 입력 state (batch_size, state_dim)
            action (torch.Tensor): 입력 action (batch_size, action_dim)
            
        Returns:
            tuple: (q1, q2)
                - q1 (torch.Tensor): 첫 번째 Q 네트워크의 Q 값 (batch_size, 1)
                - q2 (torch.Tensor): 두 번째 Q 네트워크의 Q 값 (batch_size, 1)
        """
        
        # 상태와 행동 결합 (Concatenate state and action)
        state_action = torch.cat([state, action], dim=1)
        
        # Q1 네트워크 순전파 (Q1 network forward pass)
        q1 = F.relu(self.q1_fc1(state_action))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_output(q1)
        
        # Q2 네트워크 순전파 (Q2 network forward pass)
        q2 = F.relu(self.q2_fc1(state_action))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_output(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        
        """
        첫 번째 Q 네트워크의 Q 값만 출력 (Actor 업데이트 시 사용)
        
        Actor의 policy gradient 계산 시 하나의 Q 네트워크만 사용하여
        계산 효율성을 높이고 학습 안정성을 향상시킴
        
        Args:
            state (torch.Tensor): 입력 state (batch_size, state_dim)
            action (torch.Tensor): 입력 action (batch_size, action_dim)
            
        Returns:
            torch.Tensor: 첫 번째 Q 네트워크의 Q 값 (batch_size, 1)
        """
        
        state_action = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.q1_fc1(state_action))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_output(q1)
        
        return q1

class TD3_train(object):
    def __init__(self,
                 actor_network,
                 critic_network,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_frequency=2):

        """
        TD3 (Twin Delayed Deep Deterministic Policy Gradient) 학습 클래스
        
        Args:
            actor : actor network
            critic : critic network
            state_dim: state space의 차원
            action_dim: action space의 차원
            max_action: action의 최대값
            discount: discount factor gamma
            tau: target network의 soft update 비율
            policy_noise: targetp policy smoothing을 위한 noise std
            noise_clip: noise clipping 범위
            policy_frequency: policy update frequency
        """

        # Actor 네트워크 초기화
        self.actor = actor_network.to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Critic 네트워크 초기화 (Twin Q-networks)
        self.critic = critic_network.to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # 하이퍼파라미터 저장
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency

        # 학습 iteration 카운터
        self.total_iter = 0
        
    def select_action(self, state):
        
        """
        현재 policy에 따라 action 선택 (evaluation mode)
        
        Args:
            state: 현재 state
            
        Returns:
            선택된 action (numpy array)
        """
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()
        
    def train(self, replay_buffer, batch_size=128):

        """
        리플레이 버퍼에서 학습 데이터를 샘플링하여 TD3 네트워크 학습
        
        Args:
            replay_buffer: experience replay buffer
            batch_size: mini-batch size
        """
        
        self.total_iter += 1
        
        # replay buffer에서 mini-batch 샘플링
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Target Policy Smoothing: target policy에 noise 추가
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Clipped Double Q-learning: 두 Q 값 중 최소값 선택
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # TD 타겟 계산
            target_Q = reward + not_done * self.discount * target_Q
        
        # 현재 Q 값 추정
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Critic loss 계산 (두 Q 네트워크 모두)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Critic 최적화
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========== Delayed Policy Update ==========
        if self.total_iter % self.policy_frequency == 0:
            # Actor loss 계산 (Q1 값 최대화)
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Actor 최적화
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ========== Target Network Soft Update ==========
            # Critic 타겟 네트워크 소프트 업데이트
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Actor 타겟 네트워크 소프트 업데이트
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)