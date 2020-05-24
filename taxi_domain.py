import gym
from flloat.parser.ltlf import LTLfParser
from functools import reduce
from time import sleep
import sys
from time import sleep
import numpy as np
import random
from monitoring_rewards.core import TraceStep
from monitoring_rewards.monitoring_specification import MonitoringSpecification
from monitoring_rewards.reward_monitor import RewardMonitor
from monitoring_rewards.multi_reward_monitor import MultiRewardMonitor

env = gym.make("Taxi-v3").env
""" define our monitor """


def obs_to_trace_step(observation) -> TraceStep:
    # print(observation)
    state = observation[0]
    action = observation[1]
    trace_step = {
        'pickup': False,
        'goal': False,
        'bad_action': False,
    }
    taxi = (state[0], state[1])
    passenger = state[2]
    destination = state[3]

    location = [(0, 0), (0, 4), (4, 0), (4, 3)]
    pickup = False
    goal = False
    bad_action = False
    if action == 4:  # wrong pickup
        if passenger < 4 and taxi == location[passenger]:
            pickup = True
        else:
            bad_action = True
    elif action == 5:  # wrong dropoff
        if (taxi == location[destination]) and passenger == 4:
            goal = True
        elif not (taxi in location and passenger == 4):
            bad_action = True

    trace_step['pickup'] = pickup
    trace_step['goal'] = goal
    trace_step['bad_action'] = bad_action
    # print(trace_step)
    return trace_step


""" specify the monitoring reward """

lflf_formula_1 = "F (pickup & F(goal))"  # eventually pick up passenger and eventually drop passenger in right location
monitoring_specification_1 = MonitoringSpecification(
    ltlf_formula=lflf_formula_1,
    r=0,
    c=-1,
    s=20,
    f=0
)

lflf_formula_2 = "F(G(!bad_action))"  # always eventually not bad action
monitoring_specification_2 = MonitoringSpecification(
    ltlf_formula=lflf_formula_2,
    r=0,
    c=-9,
    s=0,
    f=0
)
monitoring_specifications = [monitoring_specification_1, monitoring_specification_2]
reward_monitor = MultiRewardMonitor(
    monitoring_specifications=monitoring_specifications,
    obs_to_trace_step=obs_to_trace_step
)


def run():
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.2

    for i in range(1, 1000):
        state = env.reset()
        print('episode', i)
        epochs, penalties, reward, = 0, 0, 0
        is_perm = False
        reward_monitor([[state_i for state_i in env.decode(state)], None])  # init trace, no action at starting state

        while not is_perm:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, _, _, info = env.step(action)

            reward, is_perm = reward_monitor([[state_i for state_i in env.decode(state)], action])

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1
            state = next_state
            epochs += 1

        reward_monitor.reset()

        if i % 100 == 0:
            # clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")

    # save q_value
    with open('std_taxi.npy', 'wb') as f:
        np.save(f, q_table)


def print_frames(frames):
    for i, frame in enumerate(frames):
        # clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.2)


def test():
    env = gym.make("Taxi-v3").env
    with open('std_taxi.npy', 'rb') as f:
        q_table = np.load(f)
    is_perm = False
    penalties, reward = 0, 0
    frames = []  # for animation
    state = 328
    env.s = state  # set environment to a certain state
    timesteps = 0

    reward_monitor([[state_i for state_i in env.decode(state)], None])
    while not is_perm:

        action = np.argmax(q_table[state])  # Exploit learned values
        next_state, _, done, info = env.step(action)
        reward, is_perm = reward_monitor([[state_i for state_i in env.decode(state)], action])

        if reward == -10:
            penalties += 1

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': next_state,
            'action': action,
            'reward': reward
        })
        state = next_state
        timesteps += 1

    print_frames(frames)
    print("Timesteps taken: {}".format(timesteps))
    print("Penalties incurred: {}".format(penalties))
    #print(q_table)


if __name__ == '__main__':
    run()  # train the agent
    test()  # test performance
