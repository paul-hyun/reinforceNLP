import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random
import queue
import time
from datetime import datetime
import threading
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ValueNet, PolicyNet, Config


# 참조: https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/2-actor-critic/cartpole_a2c.py


# A3C 글로벌신경망
class A3CGlobal:
    def __init__(self, config):
        self.config = config

        # 정책신경망 생성
        self.actor = PolicyNet(self.config.n_state, self.config.n_action)
        self.actor.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)

        # 가치신경망 생성
        self.critic = ValueNet(self.config.n_state, 1)
        self.critic.to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
    
    # 리턴값 계산
    def get_returns(self, rewards, done, next_value):
        returns = torch.zeros(len(rewards), dtype=torch.float).to(self.config.device)
        R = 0 if done else next_value
        for i in reversed(range(0, len(rewards))):
            R = rewards[i] + self.config.discount_factor * R
            returns[i] = R
        return returns
    
    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, states, actions, rewards, next_states, done):
        states = torch.tensor(states, dtype=torch.float).to(self.config.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.config.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.config.device)

        next_values = self.critic(next_states).view(-1)

        # 리턴값 계산
        returns = self.get_returns(rewards, done, next_values[-1])

        values = self.critic(states).view(-1)

        # 가치신경망 학습
        critic_loss = self.train_critic(values, returns)
        # 정책신경망 학습
        actor_loss = self.train_actor(states, actions, returns - values)

        return actor_loss, critic_loss

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
    
    # GPU 메모리 반납
    def close(self):
        del self.actor
        del self.critic


# A3C 로컬신경망
class A3CLocal:
    def __init__(self, config):
        self.config = config

        # 리플레이메모리
        self.replay_memory = deque(maxlen=self.config.n_replay_memory)

        # 정책신경망 생성
        self.actor = PolicyNet(self.config.n_state, self.config.n_action)
        self.actor.to(device)

        # 가치신경망 생성
        self.critic = ValueNet(self.config.n_state, 1)
        self.critic.to(device)
    
    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        policy = self.actor(state)
        policy = policy.detach().cpu().numpy()[0]
        return np.random.choice(self.config.n_action, 1, p=policy)[0]
    
    # 리플에이메모리 추가
    def append_replay(self, state, action, reward, next_state):
        act = np.zeros(self.config.n_action)
        act[action] = 1
        self.replay_memory.append((state, act, reward, next_state))
    
    # 리플에이메모리 조회 및 클리어
    def get_replay(self):
        # 히스토리를 배열 형태로 정렬
        replay_memory = np.array(self.replay_memory)
        self.replay_memory.clear()
        states = np.vstack(replay_memory[:, 0])
        actions = list(replay_memory[:, 1])
        rewards = list(replay_memory[:, 2])
        next_states = list(replay_memory[:, 3])

        return states, actions, rewards, next_states

    # 글로벌신병망의 wegith를 로컬신경망으로 복사
    def update_local_model(self, actor_dict, critic_dict):
        self.actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)

    # GPU 메모리 반납
    def close(self):
        del self.actor
        del self.critic


# 로컬 학습 (쓰레드로 동작)
def train_local(config, actor_dict, critic_dict, global_q, index):
    # env 초기화
    env = gym.make('CartPole-v1')

    # 로컬 에이전트
    agent = A3CLocal(config)
    agent.update_local_model(actor_dict, critic_dict)

    local_q = queue.Queue()
    result = 0 # 결과 성공한 에피소드, 0: 실패

    scores, episodes = [], []

    try:
        for e in range(config.n_epoch):
            done = False
            score = 0
            losses = []
            state = env.reset()
            state = np.reshape(state, [1, config.n_state])

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, config.n_state])
                # 에피소드가 중간에 끝나면 -100 보상
                reward = reward if not done or score == 499 else config.fail_reward

                # 리플레이메모리에 step 저장
                agent.append_replay(state, action, reward, next_state)

                if done or config.n_train_step <= len(agent.replay_memory):
                    # 리플에이메모리 조회 및 클리어
                    states, actions, rewards, next_states = agent.get_replay()
                    # 글로벌큐에 리플레이메모리 갑 전달 (글로벌 에이전트 학습 요청)
                    global_q.put((states, actions, rewards, next_states, done, agent, local_q))
                    # 로컬큐로부터 학습결과 응답 대가
                    actor_loss, critic_loss = local_q.get()
                    losses.append((actor_loss, critic_loss))
                
                if 0 < reward:
                    score += reward
                state = next_state
                
                if done:
                    # 에피소드마다 학습 결과 출력
                    losses = np.array(losses)
                    actor_loss = np.sum(losses[:, 0]) / len(losses)
                    critic_loss = np.sum(losses[:, 1]) / len(losses)
                    print("[%2d] episode: %4d,   score: %3d,   actor_loss: %3.2f,   critic_loss: %3.2f" % (index, e + 1, score, actor_loss, critic_loss))
                    scores.append(score)
                    episodes.append(e)

            if done:
                # 이전 500점이 n_success 이상 이면 학습 중단
                if np.mean(scores[-min(config.n_success, len(scores)):]) >= 500:
                    result = e
                    break

    except Exception as error:
        print(error)
        result = 0

    agent.close()
    env.close()
    # 글로벌큐에 처리결과 저장 (로컬학습 완료 알림 !!!)
    global_q.put(result)


# 최종 학습 시간 (학습 모니터링 데이터)
train_timstamp = time.time()


# global 학습
def train(config):
    global train_timstamp
    global_q = queue.Queue()

    # 글로벌 에이전트
    agent = A3CGlobal(config)

    index = 0
    for _ in range(config.n_local_thread):
        thread = threading.Thread(target=train_local, args=(config, agent.actor.state_dict(), agent.critic.state_dict(), global_q, index))
        thread.start()
        index += 1

    index = config.n_local_thread
    e_min = 999
    e_max = 0
    while 0 < index:
        value = global_q.get(timeout=10)
        train_timstamp = time.time()
        if type(value) is int: # 글로벌큐에 입력값이 숫자이면 로컬학습 완료
            index -= 1
            e_min = min(e_min, value)
            e_max = max(e_max, value)
        else: # 글로벌큐에 입력값이 숫자가 아니면 학습 요청
            (states, actions, rewards, next_states, done, local_agent, local_q) = value
            # 글로벌 에이전트 학습
            actor_loss, critic_loss = agent.train_model(states, actions, rewards, next_states, done)
            # 학습결과 로컬 에이전트에 저장
            local_agent.update_local_model(agent.actor.state_dict(), agent.critic.state_dict())
            # 학습결과 로컬 에이전틍 전달
            local_q.put((actor_loss, critic_loss))
            time.sleep(0.02) # pytorch 블럭현상 방지 (블럭현상이 발생할 경우 이 값을 크게 해 주어야 함)

    agent.close()
    return False if e_min == 0 else True, e_max


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

    # config 초기화를 위해 env 생성
    env = gym.make('CartPole-v1')
    config = Config({
        "device": device, # cpu 또는 gpu 사용
        "n_epoch": 1000, # 게임 에피소드 수
        "n_state": env.observation_space.shape[0], # 상태 개수
        "n_action": env.action_space.n, # 액션 개수
        "n_replay_memory": 2000, # 리플레이 메모리 최대 크기
        "n_train_step": 32, # 리플레이 메모리 학습 단위
        "n_local_thread": 2, # 로칼 쓰레드 개수 (쓰레드 개수가 증가한다고 학습이 잘되지 않음)
        "actor_lr": 0.01, # 액터 학습률
        "critic_lr": 0.01, # 크리틱 학습률
        "discount_factor": 0.99, # 감가율
        "fail_reward": -100, # 중간에 실패할 경우 보상
        "render": False, # 화면 출력 여부
        "save_file": "save/a3c.torch", # 할습 모델 저장 위치
        "n_success": 5, # 500점을 몇번 연속하면 학습을 종료할 것인가 기준
    })
    env.close()

    success, e = train(config)
    print("sussess: %r,    epoch: %3d" % (success, e))

