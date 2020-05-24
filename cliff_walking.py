#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from flloat.parser.ldlf import LDLfParser
from flloat.parser.ltlf import LTLfParser
from functools import reduce
import time
from monitoring_rewards.core import TraceStep
from monitoring_rewards.monitoring_specification import MonitoringSpecification
from monitoring_rewards.reward_monitor import RewardMonitor
from monitoring_rewards.multi_reward_monitor import MultiRewardMonitor


matplotlib.use('Agg')
import sys

GOAL = [3, 11]
SAFE = True

class CliffWalking:

    def __init__(self):
        self.world_dimension = [4, 12]
        self.world_height = 4
        self.world_width = 12
        # all possible actions
        self.action_up = 0
        self.action_down = 1
        self.action_left = 2
        self.action_right = 3
        self.actions = [self.action_up, self.action_down, self.action_left, self.action_right]
        # initial state action pair values
        self.start = [3, 0]
        self.goal = GOAL

    def step(self, state, action):
        goal = False
        cliff = False
        i, j = state
        if action == self.action_up:
            next_state = [max(i - 1, 0), j]
        elif action == self.action_left:
            next_state = [i, max(j - 1, 0)]
        elif action == self.action_right:
            next_state = [i, min(j + 1, self.world_width - 1)]
        elif action == self.action_down:
            next_state = [min(i + 1, self.world_height - 1), j]
        else:
            assert False

        # reward = -1  # must substitute with MONITOR
        if (action == self.action_down and i == 2 and 1 <= j <= 10) or (
                action == self.action_right and state == self.start):
            cliff = True
            # reward = -100
            # next_state = self.start  # MUST substitute with MONITOR
        """ not sure on first if, should finish episode, for reward need our pipeline"""
        return next_state, cliff


class LearningAgent:

    def __init__(self, action_space, state_dim, goal):
        self.goal = goal
        self.actions = action_space
        self.state = None
        self.epsilon = 0.2  # probability for exploration over exploitation
        self.alpha = 0.5  # step size
        self.gamma = 1  # gamma for Q-Learning
        self.epsilon = 0.2
        q_shape = state_dim
        q_shape.append(len(action_space))
        self.q_value = np.zeros(q_shape)

    def choose_action(self, use_epsilon=True):
        if np.random.binomial(1, self.epsilon) == 1 and use_epsilon:
            return np.random.choice(self.actions)
        else:
            values_ = self.q_value[self.state[0], self.state[1], :]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    def q_learning(self, action, reward, next_state):
        self.q_value[self.state[0], self.state[1], action] += self.alpha * (
                reward + self.gamma * np.max(self.q_value[next_state[0], next_state[1], :]) -
                self.q_value[self.state[0], self.state[1], action])  # this is Q learning, not value iteration

    # print optimal policy
    def print_optimal_policy(self):
        optimal_policy = []
        for i in range(0, self.q_value.shape[0]):
            optimal_policy.append([])
            for j in range(0, self.q_value.shape[1]):
                if [i, j] == self.goal:
                    optimal_policy[-1].append('G')
                    continue
                bestAction = np.argmax(self.q_value[i, j, :])

                if bestAction == self.actions[0]:
                    optimal_policy[-1].append('U')
                elif bestAction == self.actions[1]:
                    optimal_policy[-1].append('D')
                elif bestAction == self.actions[2]:
                    optimal_policy[-1].append('L')
                elif bestAction == self.actions[3]:
                    optimal_policy[-1].append('R')
        for row in optimal_policy:
            print(row)


def obs_to_trace_step(observation) -> TraceStep:

    position = observation
    trace_step = {
        'goal': False,
        'cliff': False,
        'safe': True,
    }

    goal = (position == GOAL)
    if position[0] == 3 and position[1] != 0 and position[1] != 11:
        cliff = True
    else:
        cliff = False

    if position[0] != 0 and position[1] != 0 and position[1] != 11:
        safe = False
    else:
        safe = True
    trace_step['safe'] = safe

    trace_step['goal'] = goal
    trace_step['cliff'] = cliff
    # print(trace_step)

    return trace_step


def print_video(state, is_perm=False):
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    map = []
    if not is_perm:
        for row in range(4):
            r = []
            for col in range(12):
                if state[0] == row and state[1] == col:
                    r.append('A')
                elif row == 3 and col != 0 and col != 11:
                    r.append('C')
                else:
                    r.append('-')
            map.append(r)
    else:
        map = [['*', '*', '*', '*', '*', '*', ' ', '*', ' ', '*', ' ', ' '],
               ['*', ' ', ' ', '*', ' ', '*', '*', ' ', '*', '*', ' ', ' '],
               ['*', '*', '*', '*', ' ', '*', '*', '*', '*', '*', ' ', ' '],
               ['*', '*', '*', '*', '*', '*', '*', ' ', '*', '*', '*', '*']]

    print(map[0])
    print(map[1])
    print(map[2])
    print(map[3])
    time.sleep(1)


def run():
    env = CliffWalking()
    agent1 = LearningAgent(env.actions, env.world_dimension, env.goal)

    lflf_formula = "!cliff U goal"

    monitoring_specification = MonitoringSpecification(
        ltlf_formula=lflf_formula,
        r=0,
        c=-1,
        s=1,
        f=-100
    )

    if SAFE:
        lflf_formula_safe = "G(F(safe))"
        monitoring_specification_safe = MonitoringSpecification(
            ltlf_formula=lflf_formula_safe,
            r=0,
            c=-1,
            s=0,
            f=0
        )
        monitoring_specifications = [monitoring_specification, monitoring_specification_safe]
        reward_monitor = MultiRewardMonitor(
            monitoring_specifications=monitoring_specifications,
            obs_to_trace_step=obs_to_trace_step
        )
    else:
        reward_monitor = RewardMonitor(
            monitoring_specification=monitoring_specification,
            obs_to_trace_step=obs_to_trace_step
        )
            
    episodes = 1000
    rewards_q_learning = np.zeros(episodes)

    for episode in range(0, episodes):
        is_perm = False
        agent1.state = env.start

        rewards = 0.0
        reward_monitor(env.start)  # to initialize the trace
        while not is_perm:
            action = agent1.choose_action()
            next_state, cliff = env.step(agent1.state, action)

            reward, is_perm = reward_monitor(next_state)
            #print('state', next_state, 'reward', reward)
            rewards += reward
            #print(next_state)
            #print(reward)
            # Q-Learning update
            agent1.q_learning(action, reward, next_state)
            agent1.state = next_state
        reward_monitor.reset()
        rewards_q_learning[episode] += rewards


    # draw reward curves
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()
    plt.savefig('q_learning_with_logic.png')
    plt.close()
    # display optimal policy
    print('Q-Learning Optimal Policy:')
    agent1.print_optimal_policy()

    # use the best policy
    is_perm = False
    agent1.state = env.start
    print_video(agent1.state)
    reward_monitor(agent1.state)
    while not is_perm:
        action = agent1.choose_action(use_epsilon=False)  # only exploitation
        next_state, cliff = env.step(agent1.state, action)
        agent1.state = next_state
        reward, is_perm = reward_monitor(next_state)
        print_video(agent1.state, is_perm)

    print(is_perm)


if __name__ == '__main__':
    run()
