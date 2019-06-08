import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ValueNet, PolicyNet, Config


# 참고: https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/2-actor-critic/cartpole_a2c.py


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, config):
        self.config = config
        self.losses = []

        # replay memory
        self.replay_memory = deque(maxlen=self.config.n_replay_memory)

        # 정책신경망 생성
        self.actor = PolicyNet(self.config.n_state, self.config.n_action)
        self.actor.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)

        # 가치신경망 생성
        self.critic = ValueNet(self.config.n_state, 1)
        self.critic.to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        policy = self.actor(state)
        policy = policy.detach().cpu().numpy()[0]
        return np.random.choice(self.config.n_action, 1, p=policy)[0]

    # 히스토리 추가
    def append_replay(self, state, action, reward, next_state):
        act = np.zeros(self.config.n_action)
        act[action] = 1
        self.replay_memory.append((state, act, reward, next_state))

    # 리턴값 계산
    def get_returns(self, rewards, done, next_value):
        returns = np.zeros_like(rewards)
        R = 0 if done else next_value
        for i in reversed(range(0, len(rewards))):
            R = rewards[i] + self.config.discount_factor * R
            returns[i] = R
        return returns

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):
        # 히스토리를 배열 형태로 정렬
        replay_memory = np.array(self.replay_memory)
        self.replay_memory.clear()
        states = np.vstack(replay_memory[:, 0])
        actions = list(replay_memory[:, 1])
        rewards = list(replay_memory[:, 2])
        next_states = list(replay_memory[:, 3])

        states = torch.tensor(states, dtype=torch.float).to(self.config.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.config.device)

        next_values = self.critic(next_states).view(-1).detach().cpu().numpy()

        # 리턴값 계산
        returns = self.get_returns(rewards, done, next_values[-1])

        actions = torch.tensor(actions, dtype=torch.float).to(self.config.device)
        returns = torch.tensor(returns, dtype=torch.float).to(self.config.device)
        values = self.critic(states)

        # 가치신경망 학습
        critic_loss = self.train_critic(values, returns)

        # 정책신경망 학습
        actor_loss = self.train_actor(states, actions, returns - values)

        self.losses.append((actor_loss, critic_loss))
    
    # 정책신경망을 업데이트하는 함수
    def train_actor(self, states, actions, advantages):
        policy = self.actor(states)
        action_prob = torch.sum(actions * policy, dim=1)
        cross_entropy = torch.log(action_prob + 1.e-7) * advantages.detach()
        actor_loss = -torch.mean(cross_entropy)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()
    
    # 가치신경망을 업데이트하는 states
    def train_critic(self, values, targets):
        critic_loss = torch.mean(torch.pow(targets - values, 2))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    # model의 weight를 파일로 저장
    def save(self):
        torch.save(self.actor.state_dict(), self.config.save_file + ".actor")
        torch.save(self.critic.state_dict(), self.config.save_file + ".critic")
    
    # 파일로 부터 model의 weight를 읽어 옴
    def load(self):
        self.actor.load_state_dict(torch.load(self.config.save_file + ".actor"))
        self.critic.load_state_dict(torch.load(self.config.save_file + ".critic"))
    
    # GPU 메모리 반납
    def close(self):
        del self.actor
        del self.critic


def train(env, config):
    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(config)

    scores, episodes = [], []

    for e in range(config.n_epoch):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, config.n_state])

        while not done:
            if config.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, config.n_state])
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else config.fail_reward

            # 리플레이 메모리에 step 저장
            agent.append_replay(state, action, reward, next_state)
            if done or config.n_train_step <= len(agent.replay_memory):
                agent.train_model(done)

            if 0 < reward:
                score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                losses = np.array(agent.losses)
                agent.losses.clear()
                actor_loss = np.sum(losses[:, 0]) / len(losses)
                critic_loss = np.sum(losses[:, 1]) / len(losses)
                print("episode: %4d,    score: %3d,    actor_loss: %3.2f,    critic_loss: %3.2f" % (e, score, actor_loss, critic_loss))
                scores.append(score)
                episodes.append(e)

                # 이전 500점이 n_success 이상 이면 학습 중단
                if np.mean(scores[-min(config.n_success, len(scores)):]) >= 500:
                    agent.save()
                    agent.close()
                    return True, e

    agent.close()
    return False, 0


if __name__ == "__main__":
    # 데어처 저장 폴더
    if not os.path.exists("save"):
        os.makedirs("save")
 
    # cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pytorch seed 초기화
    seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # env 초기화
    env = gym.make('CartPole-v1')

    config = Config({
        "device": device, # cpu 또는 gpu 사용
        "n_epoch": 1000, # 게임 에피소드 수
        "n_state": env.observation_space.shape[0], # 상태 개수
        "n_action": env.action_space.n, # 액션 개수
        "n_replay_memory": 2000, # 리플레이 메모리 최대 크기
        "n_train_step": 32, # 리플레이 메모리 학습 단위
        "actor_lr": 0.01, # 액터 학습률
        "critic_lr": 0.01, # 크리틱 학습률
        "discount_factor": 0.99, # 감가율
        "fail_reward": -100, # 중간에 실패할 경우 보상
        "render": False, # 화면 출력 여부
        "save_file": "save/a2c.torch", # 할습 모델 저장 위치
        "n_success": 5, # 500점을 몇번 연속하면 학습을 종료할 것인가 기준
    })

    success, e = train(env, config)
    print("sussess: %r,    epoch: %3d" % (success, e))
    
    env.close()
