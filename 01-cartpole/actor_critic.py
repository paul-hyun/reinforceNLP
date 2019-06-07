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

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    
        value = self.critic(state).item()
        next_value = self.critic(next_state).item()

        act = np.zeros([1, self.config.n_action])
        act[0][action] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = [reward - value]
            target = [reward]
        else:
            advantage = [(reward + self.config.discount_factor * next_value) - value]
            target = [reward + self.config.discount_factor * next_value]

        # 정책신경망 학습
        act = torch.tensor(act, dtype=torch.float).to(device)
        advantage = torch.tensor(advantage, dtype=torch.float).to(device)
        actor_loss = self.train_actor(state, act, advantage)

        # 가치신경망 학습
        target = torch.tensor(target, dtype=torch.float).to(device)
        critic_loss = self.train_critic(state, target)

        self.losses.append((actor_loss, critic_loss))
    
    # 정책신경망을 업데이트하는 함수
    def train_actor(self, state, action, advantage):
        policy = self.actor(state)
        action_prob = torch.sum(action * policy, dim=1)
        cross_entropy = torch.log(action_prob + 1.e-7) * advantage
        actor_loss = -torch.mean(cross_entropy)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()
    
    # 가치신경망을 업데이트하는 함수
    def train_critic(self, state, target):
        value = self.critic(state)
        critic_loss = torch.mean(torch.pow(target - value, 2))

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

            agent.train_model(state, action, reward, next_state, done)

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
