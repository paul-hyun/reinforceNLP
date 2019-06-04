import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import gym

import torch

from model import CartpoleQ, CartpoleConfig


# 참고: https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py


"""
Deep Q Network Agent
"""
class DQNAgent:
    # 초기화
    def __init__(self, config):
        self.config = config
        self.epsilon = config.epsilon

        # replay memory
        self.replay_memory = deque(maxlen=config.n_replay_memory)

        # train model, target model 생성
        self.model = CartpoleQ(config.n_state, config.n_action)
        self.model.to(self.config.device)
        self.target = CartpoleQ(config.n_state, config.n_action)
        self.target.to(self.config.device)
        # MSE loss 및 optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
    
    # 상태에 대한 Action 에측 (epsilon 확률로 탐험:exploration)
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.config.n_action)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.config.device)
            output = self.model(state)
            return output.argmax().item()
    
    # 리플레이 메모리에 St, At, Rt+1, St+1, Done 추가
    def append_replay(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
    
    # 리플레이 메모리가 일정 수 이상 수집된 후 학습이 가능 함
    def is_train(self):
        return True if self.config.n_train_start < len(self.replay_memory) else False
    
    # 학습
    def train(self):
        # 학습이 계속 될 수 록 탐험 학률을 줄여 줌
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        # 리플레이 메모리에서 배치사이즈 만큼 랜덤 셈플링
        mini_batch = random.sample(self.replay_memory, self.config.n_batch)

        # 랜덤 샘플링 된 데이터를 배열 형태로 정렬
        mini_batch = np.array(mini_batch)
        states = np.vstack(mini_batch[:, 0])
        actions = list(mini_batch[:, 1])
        rewards = list(mini_batch[:, 2])
        next_states = list(mini_batch[:, 3])
        dones = list(mini_batch[:, 4])

        states = torch.tensor(states, dtype=torch.float).to(self.config.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.config.device)

        values = self.model(states) ## 정답
        target = np.copy(values.detach().cpu().numpy()) ## Vt
        next_values = self.target(next_states).detach().cpu().numpy() ## Vt+1

        for i in range(len(target)):
            if dones[i]:
                target[i][actions[i]] = rewards[i] # Vt = Rt+1
            else:
                target[i][actions[i]] = rewards[i] + self.config.discount_factor * (np.amax(next_values[i])) # Vt = Rt+1 + rVt+1
        target = torch.tensor(target, dtype=torch.float).to(self.config.device)
        
        # 학습
        self.optimizer.zero_grad()
        loss = self.criterion(values, target)
        loss.backward()
        self.optimizer.step()
    
    # model의 weight를 target에 저장
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
    
    # model의 weight를 파일로 저장
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    # 파일로 부터 model의 weight를 읽어 옴, target에도 동일하게 적용
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target()


def train(config):
    # agent 생성
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

            # St, At, Rt+1, St+1, Done: 리플레이 메모리 저장
            agent.append_replay(state, action, reward, next_state, done)

            if reward != -100:
                score += reward
            state = next_state

            # 리플레이 메모리에 일정량 이상의 데이터가 쌓이면 매 스텝마다 학습
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
    # env 종료
    env.close()

    # 학습 스코어 화면 출력
    plt.plot(epochs, scores, label='score')
    plt.show()


def run(config):
   # 환경
    config.n_epoch = 10
    config.render = True
    config.epsilon = 0

    # agent 생성
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
    
    # cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pytorch seed 초기화
    seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # env 초기화
    env = gym.make('CartPole-v1')

    config = CartpoleConfig({
        "device": device,
        "n_epoch": 3000,
        "n_state": env.observation_space.shape[0],
        "n_action": env.action_space.n,
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
