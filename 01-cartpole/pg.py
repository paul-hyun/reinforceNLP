import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import gym

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from model import CartpoleP, CartpoleConfig


# 참고: https://github.com/rlcode/reinforcement-learning-kr/blob/master/1-grid-world/7-reinforce/reinforce_agent.py
#      https://github.com/Finspire13/pytorch-policy-gradient-example/blob/master/pg.py

"""
Policy Gradient Agent
"""
class PGAgent:
    def __init__(self, config):
        self.config = config
        self.epsilon = config.epsilon

        # replay memory
        self.history_memory = []

        # train model, target model 생성
        self.model = CartpoleP(config.n_state, config.n_action)
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
    
    # 상태에 대한 Action 에측 (epsilon 확률로 탐험:exploration)
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.config.n_action)
        else:
            output = self.model(torch.tensor(state, dtype=torch.float))
            output = output.detach().numpy()[0]
            return np.random.choice(self.config.n_action, 1, p=output)[0]
    
    # 히스토리 추가
    def append_history(self, state, action, reward):
        self.history_memory.append((state, action, reward))
    
    # 학습
    def train(self):
        # 학습이 계속 될 수 록 탐험 학률을 줄여 줌
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        # 히스토리를 배열 형태로 정렬
        states = np.zeros((len(self.history_memory), self.config.n_state))
        actions, rewards = [], []

        for i in range(len(self.history_memory)):
            states[i] = self.history_memory[i][0]
            # action은 onehot으로 저장
            action = np.zeros(self.config.n_action)
            action[self.history_memory[i][1]] = 1
            actions.append(action)
            rewards.append(self.history_memory[i][2])

        # discount reward 계산
        discounted_rewards = np.zeros_like(rewards)
        R = 0
        for i in reversed(range(0, len(rewards))):
            R = rewards[i] + self.config.discount_factor * R
            discounted_rewards[i] = R
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        prob = self.model(torch.tensor(states, dtype=torch.float))
        action_prob = torch.sum(torch.tensor(actions, dtype=torch.float) * prob, dim=1) ## π(St,At): 액션을 수행할 확률
        cross_entropy = torch.log(action_prob + 1.e-10) * torch.tensor(discounted_rewards, dtype=torch.float) ## log(π(St,At)) * Gt, 1.e-10는 0 방지
        loss = -torch.mean(cross_entropy) ## 음수를 취하여 경사 하강법으로 학습 (sum 보다 mean이 더 학습이 잘 됨)

        # 학습
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 히스토리 삭제 (다시 히스토리 저장하기 위함)
        self.history_memory.clear()
    
    # model의 weight를 파일로 저장
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    # 파일로 부터 model의 weight를 읽어 옴
    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def train(config):
    # pytorch seed 초기화
    seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # env 초기화
    env = gym.make('CartPole-v1')

    # 환경
    config.n_state = env.observation_space.shape[0]
    config.n_action = env.action_space.n

    # agent 생성
    agent = PGAgent(config)

    # 그래프 출력용 데이터
    epochs = []
    scores = []

    for epoch in range(config.n_epoch):
        done = False
        score = 0

        # St: 초기 state 조회
        state = env.reset()
        state = np.reshape(state, [1, config.n_state])

        while not done:
            if config.render:
                env.render()
            
            # At: St 대한 action 예측
            action = agent.get_action(state)
            # St+1, Rt+1: action 수행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, config.n_state])
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100

            # St, At, Rt+1: 히스토리 저장
            agent.append_history(state, action, reward)

            if reward != -100:
                score += reward
            state = next_state
            
            if done:
                # 에프소드가 종료되면 한번에 학습 (몬테카를로)
                agent.train()

                print("epoch: %d, score: %d" % (epoch + 1, score))

                epochs.append(epoch + 1)
                scores.append(score)

                # 이전 5개 에피소드의 점수 평균이 500이면 학습 중단
                if np.mean(scores[-min(5, len(scores)):]) > 499:
                    agent.save(config.save_file)
                    sys.exit()
    # env 종료
    env.close()

    # 학습 스코어 화면 출력
    plt.plot(epochs, scores, label='score')
    plt.show()


def run(config):
    # env 초기화
    env = gym.make('CartPole-v1')

    # 환경
    config.n_epoch = 10
    config.render = True
    config.n_state = env.observation_space.shape[0]
    config.n_action = env.action_space.n
    config.epsilon = 0

    # agent 생성
    agent = PGAgent(config)
    agent.load(config.save_file)

    done = False
    score = 0

    # St: 초기 state 조회
    state = env.reset()
    state = np.reshape(state, [1, config.n_state])

    # 그래프 출력용 데이터
    epochs = []
    scores = []

    for epoch in range(config.n_epoch):
        done = False
        score = 0

        # St: 초기 state 조회
        state = env.reset()
        state = np.reshape(state, [1, config.n_state])

        while not done:
            if config.render:
                env.render()
            
            # At: St 대한 action 예측
            action = agent.get_action(state)
            # St+1, Rt+1: action 수행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, config.n_state])

            score += reward
            state = next_state
            
            if done:
                print("epoch: %d, score: %d" % (epoch + 1, score))

                epochs.append(epoch + 1)
                scores.append(score)
    # env 종료
    env.close()

    # 학습 스코어 화면 출력
    plt.plot(epochs, scores, label='score')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'run'], default='run', nargs='?')
    args = parser.parse_args()

    if not os.path.exists("save"):
        os.makedirs("save")

    config = CartpoleConfig({
        "n_epoch": 500,
        "n_state": 0,
        "n_action": 0,
        "n_batch": 64,
        "n_replay_memory": 2000,
        "n_train_start": 1000,
        "learning_rate": 0.01,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "render": False,
        "save_file": "save/pg.torch",
    })

    if args.mode == 'train':
        train(config)
    elif args.mode == 'run':
        run(config)
