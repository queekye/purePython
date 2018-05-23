from numpy import *
from ddpg import DDPG
from env import Env
import time

STATE_DIM = 6
ACTION_DIM = 6

NUM_EPISODE = 50000
TEST_GAP = 100
TEST_NUM = 10
STEP_LIMIT = 200


def angel_to_q(ang):
    a, b, c = ang[0], ang[1], ang[2]
    qqq = array([cos(a / 2) * cos(b / 2) * cos(c / 2) + sin(a / 2) * sin(b / 2) * sin(c / 2), sin(a / 2) * cos(b / 2) * cos(c / 2) - cos(a / 2) * sin(b / 2) * sin(c / 2), cos(a / 2) * sin(b / 2) * cos(c / 2) + sin(a / 2) * cos(b / 2) * sin(c / 2), -sin(a / 2) * sin(b / 2) * cos(c / 2) + cos(a / 2) * cos(b / 2) * sin(c / 2)])
    qqq /= sqrt(qqq.dot(qqq))
    return qqq


env = Env()
agent = DDPG(STATE_DIM, ACTION_DIM)
state_input = env.observe_state
for i in range(NUM_EPISODE):
    t_start = time.time()
    state_initialized = False
    reward = 0
    while not state_initialized:
        sd = random.randint(1, 10000000)
        state_initialized, state_input, matrix_input = env.set_state_seed(sd)
    # action实际输出的是ZYX欧拉角(-1,1)，需要转换为四元数
    for step in range(STEP_LIMIT):
        action = agent.noise_action(state_input)
        q = angel_to_q(action[0:3] * pi)
        qw = concatenate([q, action[3:6]])
        next_state, next_matrix, reward, done, stop = env.step(qw)
        agent.perceive(state_input, action, reward, next_state, done or stop)
        state_input = next_state
        if done or stop:
            break
    print("episode: %10d   reward:%10d   replay_buffer_num:%10d   cost_time:%10.5f  \n" % (i, reward, agent.replay_buffer.count(), time.time() - t_start))

    if mod(i, TEST_GAP) == 0 and i > 1:
        total_reward = 0.
        for j in range(TEST_NUM):
            state_initialized = False
            while not state_initialized:
                sd = random.randint(1, 10000000)
                state_initialized, state_input, matrix_input = env.set_state_seed(sd)
            # action实际输出的是ZYX欧拉角(-1,1)，需要转换为四元数
            for step in range(STEP_LIMIT):
                action = agent.action(state_input)
                q = angel_to_q(action[0:3] * pi)
                qw = concatenate([q, action[3:6]])
                next_state, next_matrix, reward, done, stop = env.step(qw)
                state_input = next_state
                total_reward += reward
                if done or stop:
                    break
        ave_reward = total_reward/TEST_NUM
        print("episode: %d    Evaluation Average Reward:%f  \n" % (i, ave_reward))
