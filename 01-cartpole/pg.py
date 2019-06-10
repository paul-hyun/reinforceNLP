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

from model import PolicyNet, Config


# 참고: https://github.com/rlcode/reinforcement-learning-kr/blob/master/1-grid-world/7-reinforce/reinforce_agent.py


# 카트폴 예제에서의 Polocy Gradiend 에이전트
class PGAgent:
    def __init__(self, config):
        self.config = config

        # replay memory
        self.replay_memory = deque(maxlen=self.config.n_replay_memory)

        # 정책신경망 생성
        self.model = PolicyNet(self.config.n_state, self.config.n_action)
        self.model.to(device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        policy = self.model(state)
        policy = policy.detach().cpu().numpy()[0]
        return np.random.choice(self.config.n_action, 1, p=policy)[0]
    
    # 히스토리 추가
    def append_replay(self, state, action, reward, next_state):
        act = np.zeros(self.config.n_action)
        act[action] = 1
        self.replay_memory.append((state, act, reward, next_state))

    # 리턴값 계산
    def get_returns(self, rewards):
        returns = np.zeros_like(rewards)
        R = 0
        for i in reversed(range(0, len(rewards))):
            R = rewards[i] + self.config.discount_factor * R
            returns[i] = R
        if 1 < len(returns):
            returns -= np.mean(returns)
            returns /= (np.std(returns) + 1.e-7)
        return returns

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self):
        # 히스토리를 배열 형태로 정렬
        replay_memory = np.array(self.replay_memory)
        self.replay_memory.clear()
        states = np.vstack(replay_memory[:, 0])
        actions = list(replay_memory[:, 1])
        rewards = list(replay_memory[:, 2])
        next_states = list(replay_memory[:, 3])

        # 리턴값 계산
        returns = self.get_returns(rewards)

        states = torch.tensor(states, dtype=torch.float).to(self.config.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.config.device)
        returns = torch.tensor(returns, dtype=torch.float).to(self.config.device)
        loss = self.train_policy(states, actions, returns)

        return loss
    
    # 정책신경망을 업데이트하는 함수
    def train_policy(self, states, actions, returns):
        policy = self.model(states)
        action_prob = torch.sum(actions * policy, dim=1)
        cross_entropy = torch.log(action_prob + 1.e-7) * returns
        loss = -torch.mean(cross_entropy)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return loss.item()

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
    # Ploicy Gradient 에이전트 생성
    agent = PGAgent(config)

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

            if 0 < reward:
                score += reward
            state = next_state

            if done:
                # 에피소드 완료 후 학습
                loss = agent.train_model()

                # 에피소드마다 학습 결과 출력
                print("episode: %4d,    score: %3d,    loss: %3.2f" % (e, score, loss))
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
        "learning_rate": 0.01, # 학습율
        "discount_factor": 0.99, # 감가율
        "fail_reward": -100, # 중간에 실패할 경우 보상
        "render": False, # 화면 출력 여부
        "save_file": "save/pg.torch", # 할습 모델 저장 위치
        "n_success": 5, # 500점을 몇번 연속하면 학습을 종료할 것인가 기준
    })

    success, e = train(env, config)
    print("sussess: %r,    epoch: %3d" % (success, e))
    
    env.close()
