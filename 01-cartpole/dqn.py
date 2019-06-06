import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import gym

import torch

from model import ValueNet, Config


# 참고: https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.epsilon = config.epsilon
        self.losses = []

        # replay memory
        self.replay_memory = deque(maxlen=self.config.n_replay_memory)

        # 가치신경망 생성
        self.model = ValueNet(self.config.n_state, self.config.n_action)
        self.model.to(device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.config.n_action)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.config.device)
            output = self.model(state)
            return output.argmax().item()
        
    # 히스토리 추가
    def append_replay(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self):
        # 학습이 계속 될 수 록 탐험 학률을 줄여 줌
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

        # 히스토리를 배열 형태로 정렬
        replay_memory = np.array(random.sample(self.replay_memory, self.config.n_batch))
        states = np.vstack(replay_memory[:, 0])
        actions = list(replay_memory[:, 1])
        rewards = list(replay_memory[:, 2])
        next_states = list(replay_memory[:, 3])
        dones = list(replay_memory[:, 4])

        states = torch.tensor(states, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
    
        targets = self.model(states).detach().cpu().numpy()
        next_values = self.model(next_states).detach().cpu().numpy()

        for i in range(len(targets)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i] # Vt = Rt+1
            else:
                targets[i][actions[i]] = rewards[i] + self.config.discount_factor * (np.amax(next_values[i])) # Vt = Rt+1 + rVt+1

        targets = torch.tensor(targets, dtype=torch.float).to(device)
        ValueNet_loss = self.train_ValueNet(states, targets)

        self.losses.append(ValueNet_loss)
    
    # 가치신경망을 업데이트하는 함수
    def train_ValueNet(self, state, target):
        value = self.model(state)
        ValueNet_loss = torch.mean(torch.pow(target - value, 2))

        self.model_optimizer.zero_grad()
        ValueNet_loss.backward()
        self.model_optimizer.step()

        return ValueNet_loss.item()

    # model의 weight를 파일로 저장
    def save(self):
        torch.save(self.model.state_dict(), self.config.save_file)
    
    # 파일로 부터 model의 weight를 읽어 옴
    def load(self):
        self.model.load_state_dict(torch.load(self.config.save_file))
    
    # GPU 메모리 반납
    def close(self):
        del self.model


def train(env, config):
    # DQN 에이전트 생성
    agent = DQNAgent(config)

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
            agent.append_replay(state, action, reward, next_state, done)
            if config.n_train_start < len(agent.replay_memory):
                agent.train_model()

            if 0 < reward:
                score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                losses = np.array(agent.losses)
                agent.losses.clear()
                ValueNet_loss = 0 if len(losses) == 0 else np.sum(losses) / len(losses)
                print("episode: %4d,    score: %3d,    loss: %3.2f" % (e, score, ValueNet_loss))
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
        "n_batch": 64, # 학습 데이터 크기 (리플레이 메모리에서 셈플링 할 수)
        "n_replay_memory": 2000, # 리플레이 메모리 최대 크기
        "n_train_start": 1000, # 리플레이 메모리 취소 학습 시작 기준
        "learning_rate": 0.02, # 학습율
        "discount_factor": 0.99, # 감가율
        "fail_reward": -100, # 중간에 실패할 경우 보상
        "epsilon": 1.0, # 참험 확률
        "epsilon_decay": 0.99, # 탐험확률 감소 율
        "epsilon_min": 0.01, # 참험 확률 최소값
        "render": False, # 화면 출력 여부
        "save_file": "save/dqn.torch", # 할습 모델 저장 위치
        "n_success": 5, # 500점을 몇번 연속하면 학습을 종료할 것인가 기준
    })


    success, e = train(env, config)
    print("sussess: %r,    epoch: %3d" % (success, e))
    
    env.close()
