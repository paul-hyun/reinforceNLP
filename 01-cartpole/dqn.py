import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import gym

import torch

from model import CartpoleNet, CartpoleConfig


# 참고: https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py

"""
Deep Q Network Agent
"""
class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.epsilon = config.epsilon

        # replay memory
        self.replay_memory = deque(maxlen=config.n_replay_memory)

        # train model, target model 생성
        self.model = CartpoleNet(config.n_state, config.n_action, softmax=False)
        self.target = CartpoleNet(config.n_state, config.n_action, softmax=False)
        # MSE loss 및 optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
    
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.config.n_action)
        else:
            output = self.model(torch.tensor(state, dtype=torch.float))
            return output.argmax().item()
    
    def append_replay(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
    
    def is_train(self):
        return True if self.config.n_train_start < len(self.replay_memory) else False
    
    def train(self):
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        mini_batch = random.sample(self.replay_memory, self.config.n_batch)

        states = np.zeros((self.config.n_batch, self.config.n_state))
        next_states = np.zeros((self.config.n_batch, self.config.n_state))
        actions, rewards, dones = [], [], []

        for i in range(len(mini_batch)):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
        
        values = self.model(torch.tensor(states, dtype=torch.float))
        labels = np.copy(values.detach().numpy())
        next_values = self.target(torch.tensor(next_states, dtype=torch.float)).detach().numpy()

        for i in range(len(labels)):
            if dones[i]:
                labels[i][actions[i]] = rewards[i]
            else:
                labels[i][actions[i]] = rewards[i] + self.config.discount_factor * (np.amax(next_values[i]))
        
        self.optimizer.zero_grad()
        loss = self.criterion(values, torch.tensor(labels, dtype=torch.float))
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target()


def train(config):
    # env 초기화
    env = gym.make('CartPole-v1')

    # 환경
    config.n_state = env.observation_space.shape[0]
    config.n_action = env.action_space.n

    # DQN agent 생성
    agent = DQNAgent(config)

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

            # St, At, Rt+1, St+1: 리플레이 메모리 저장
            agent.append_replay(state, action, reward, next_state, done)

            if reward != -100:
                score += reward
            state = next_state

            if agent.is_train():
                agent.train()
            
            if done:
                print("epoch: %d, score: %d" % (epoch + 1, score))
                agent.update_target()

                epochs.append(epoch + 1)
                scores.append(score)

                # 이전 5개 에피소드의 점수 평균이 500이면 학습 중단
                if np.mean(scores[-min(5, len(scores)):]) > 499:
                    agent.save(config.save_file)
                    sys.exit()
    
    env.close()
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

    # DQN agent 생성
    agent = DQNAgent(config)
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
    
    env.close()
    plt.plot(epochs, scores, label='score')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'run'], default='run', nargs='?')
    args = parser.parse_args()

    if not os.path.exists("save"):
        os.makedirs("save")

    config = CartpoleConfig({
        "n_epoch": 300,
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
        "save_file": "save/dqn.torch",
    })

    if args.mode == 'train':
        train(config)
    elif args.mode == 'run':
        run(config)
