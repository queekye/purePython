from numpy import *

from TerrainMap import TerrainMap

# 任务介绍：
# 小行星探测器在小行星上跳跃行走，目标点始终设定为原点。
# 输入网络的地图为32*32，每像素代表4m*4m的区域，即总共128m*128m的区域
# 为方便网络对地图的理解，设定达到原点附近的半径为8的区域就视为到达目标
# agent-env交互方案：
# agent接收探测器所有角点均离开地面时的状态和相应的global_map,local_map；输出期望的姿态角速度
# env接收agent的action后，无控仿真至碰撞，然后按此时期望姿态下是否有角点在地面以下，向前或向后搜索碰前状态，


ACTION_DIM = 7
STATE_DIM = 6

# 地图参数
MAP_DIM = 32
PIXEL_METER = 4
DONE_R = 8

# 小行星与探测器的相关参数
g = array([0, 0, -0.001])
I_star = array([[0.055, 0, 0], [0, 0.055, 0], [0, 0, 0.055]])
I_star_inv = array([[1 / 0.055, 0, 0], [0, 1 / 0.055, 0], [0, 0, 1 / 0.055]])
U = eye(3)
J_wheel = array([[0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]])
UJ_inv = array([[500, 0, 0], [0, 500, 0], [0, 0, 500]])
m = 5
l_robot = array([0.2, 0.2, 0.2])
mu = 0.45
# k:地面刚度数据，有待考证
k = 4.7384 * 10 ** 4
recovery_co = 0.95
# mu0:静摩擦系数
mu0 = 0.48
lx = l_robot[0]
ly = l_robot[1]
lz = l_robot[2]
# vertex_b : 体坐标系下初始各角点相对质心的位置，dim:3*8
vertex_b = array([[lx, lx, lx, lx, -lx, -lx, -lx, -lx],
                  [-ly, -ly, ly, ly, -ly, -ly, ly, ly],
                  [-lz, lz, lz, -lz, -lz, lz, lz, -lz]])
MIN_ENERGY = 0.0001

# RK4算法参数
STEP_LENGTH = 0.001

# 探测器空中飞行时间低于此值时认为无法控制到目标姿态角速度，按无控进行仿真
MIN_FLY_TIME = 70


# check: OK
def mat_q(q):
    mq = array([[q[0], -q[1], -q[2], -q[3]],
                [q[1], q[0], -q[3], q[2]],
                [q[2], q[3], q[0], -q[1]],
                [q[3], -q[2], q[1], q[0]]])
    return mq


# check:OK
def crossMatrix(w):
    wx = array([[0, -w[2], w[1]],
                [w[2], 0, -w[0]],
                [-w[1], w[0], 0]])
    return wx


# check:OK
# 四元数转换为姿态余弦矩阵
def q_to_DCM(q):
    a = q[0] * eye(3) - crossMatrix(q[1:4])
    DCM = dot(reshape(q[1:4], [-1, 1]), reshape(q[1:4], [1, -1])) + dot(a, a)
    return DCM


# check:OK
# 计算惯性系下的各角点参数
def vertex(terrain_map, state):
    XYZc, v, q, w = state[0:3], state[3:6], state[6:10], state[10:13]
    q /= sqrt(q.dot(q))
    DCM = q_to_DCM(q)
    vertex_s = dot(DCM.T, vertex_b) + reshape(XYZc, [-1, 1])
    vertex_high = vertex_s[2, :] - terrain_map.get_high(vertex_s[0, :], vertex_s[1, :])
    vertex_v = dot(DCM.T, dot(crossMatrix(w), vertex_b)) + reshape(v, [-1, 1])
    return vertex_s, vertex_high, vertex_v


# check:OK
# 动力学微分方程，flag标记是否发生碰撞，true为在碰撞，
def dynamic(terrain_map, state, flag=ones([8]) < 0, v0=zeros(8), normal=ones([3, 8])):
    XYZc, v, q, w, w_wheel = state[0:3], state[3:6], state[6:10], state[10:13], state[13:16]
    q /= sqrt(q.dot(q))
    d_w_wheel = array([0, 0, 0])
    F = zeros([3, 8])
    T = zeros([3, 8])
    DCM = q_to_DCM(q)
    vertex_s, vertex_high, vertex_v = vertex(terrain_map, state)
    Teq = cross(w, I_star.dot(w) + U.dot(J_wheel).dot(w_wheel)) + U.dot(J_wheel).dot(d_w_wheel)
    for i in range(0, 8):
        if flag[i]:
            r_one = vertex_b[:, i]
            high = vertex_high[i]
            normal_one = normal[:, i]
            invade_s = -dot(array([0, 0, high]), normal_one)
            if invade_s < 0:
                continue
            vertex_v_one = vertex_v[:, i]
            invade_v = -dot(vertex_v_one, normal_one)
            vt = vertex_v_one - dot(vertex_v_one, normal_one) * normal_one
            vt_value = sqrt(dot(vt, vt))
            if abs(v0[i]) < 0.00001:
                v0[i] = 0.00001*sign(v0[i])
            c = 0.75 * (1 - recovery_co ** 2) * k * (invade_s ** 1.5) / v0[i]
            Fn_value = k * (invade_s ** 1.5) + c * invade_v
            Fn = Fn_value * normal_one
            if vt_value >= 0.0006:
                Ft = -mu * Fn_value * vt / vt_value
            else:
                # 具体方程及求解过程见 笔记

                A = eye(3) / m - linalg.multi_dot([crossMatrix(r_one), I_star_inv, crossMatrix(r_one), DCM])
                A_inv = linalg.inv(A)
                b = crossMatrix(r_one).dot(I_star_inv).dot(Teq) + dot(w, r_one) * w - dot(w, w) * r_one
                alpha = (Fn.dot(Fn) + A_inv.dot(b).dot(Fn)) / (A_inv.dot(Fn).dot(Fn))
                Ft = -A_inv.dot(dot(A - alpha * eye(3), Fn) + b)
                Ft_value = sqrt(Ft.dot(Ft))
                if Ft_value >= mu0 * Fn_value:
                    Ft = mu0 * Fn_value * Ft / Ft_value
            F[:, i] = Ft + Fn
            T[:, i] = cross(r_one, DCM.dot(Ft + Fn))
    F = sum(F, 1) + m * g
    T = sum(T, 1)
    M_star = T - Teq
    d_XYZc = v
    d_v = F / m
    d_q = 0.5 * dot(mat_q(q), concatenate((zeros(1), w)))
    d_w = I_star_inv.dot(M_star)
    d_state = concatenate((d_XYZc, d_v, d_q, d_w, d_w_wheel))
    return d_state


# check:OK
def RK4(terrain_map, t, state, step_length, flag=ones([8]) < 0, v0=zeros(8), normal=ones([3, 8])):
    h = step_length
    k1 = dynamic(terrain_map, state, flag, v0, normal)
    k2 = dynamic(terrain_map, state + h * k1 / 2, flag, v0, normal)
    k3 = dynamic(terrain_map, state + h * k2 / 2, flag, v0, normal)
    k4 = dynamic(terrain_map, state + h * k3, flag, v0, normal)
    state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    state[6:10] /= linalg.norm(state[6:10])
    t += h
    return t, state


class Env:
    s_dim = STATE_DIM
    a_dim = ACTION_DIM
    a_bound = array([1, 1, 1, 1, 2, 2, 2])

    def __init__(self):
        self.t = 0
        self.observe_t = 0
        self.state = array([0, 0, 5, 0.12, -0.08, 0, 1, 0, 0, 0, 0.2, -0.1, 0.15, -1.9, 1.5, -1.2])
        self.observe_state = self.state.copy()
        self.terrain_map = TerrainMap(3, MAP_DIM, PIXEL_METER)

    # 生成地图
    def set_map_seed(self, sd=1):
        self.terrain_map = TerrainMap(sd, MAP_DIM, PIXEL_METER)

    # check:
    # 设定初始状态，即探测器与地面的撞前状态
    # 暂时不设定难度，（根据初始xy坐标与原点（目标）的距离，分10个难度，默认为最高难度）
    def set_state_seed(self, sd=1):
        random.seed(sd)
        minXY = 5 * PIXEL_METER
        maxXY = MAP_DIM * PIXEL_METER / 2 - 2 * PIXEL_METER
        minVxy = 0.1
        maxVxy = 0.3
        minZ = 2
        maxZ = 5
        XY_theta = random.random() * 2 * pi
        XY = ((maxXY - minXY) * random.random() + minXY) * array([cos(XY_theta), sin(XY_theta)])
        Z = (random.random() * (maxZ - minZ) + minZ) + self.terrain_map.get_high(XY[0], XY[1])
        v_theta = random.random() * 2 * pi
        v_xy = ((maxVxy - minVxy) * random.random() + minVxy) * array([cos(v_theta), sin(v_theta)])
        q = random.rand(4)
        q /= linalg.norm(q)
        w = random.rand(3) * 2 - 1
        w_wheel = random.rand(3) * 2 - 1
        self.state = concatenate([XY, array([Z]), v_xy, zeros(1), q, w, w_wheel])
        flag, v0, normal, t_fly = self.hypothesisSim()
        t_fly = 0
        stop_bool = False
        done_bool = False
        while t_fly < MIN_FLY_TIME and not (stop_bool or done_bool):
            self.collisionSim(flag, v0, normal)
            self.t = 0
            self.observe_state = self.state.copy()
            self.observe_t = 0
            stop_bool = self.energy() < MIN_ENERGY or linalg.norm(self.state[0:2]) > MAP_DIM*PIXEL_METER
            done_bool = linalg.norm(self.state[0:2]) < DONE_R
            flag, v0, normal, t_fly = self.hypothesisSim()

        if done_bool or stop_bool:
            return False, self.state[0:6], self.terrain_map.map_matrix
        else:
            return True, self.observe_state[0:6], self.terrain_map.map_matrix

    # check:
    # step env接收agent的action后的状态转移
    # 输入action即碰前角速度与姿态，输出新的state，reward以及标志该次仿真是否达到目的地的done，是否速度过小导致停止的stop
    # 先按假想的无控进行仿真，若探测器在空中时间小于MIN_FLY_TIME,按无控进行仿真
    # ?探测器在空中角动量守恒，应该可以计算出与action对应的飞轮转速？
    def step(self, action):
        t0 = self.observe_t
        pre_state = self.observe_state.copy()
        done_bool, flag, v0, normal = self.relocation(action)
        t_fly = 0
        stop_bool = False
        done_bool = False
        while t_fly < MIN_FLY_TIME and not (stop_bool or done_bool):
            self.collisionSim(flag, v0, normal)
            self.observe_state = self.state.copy()
            self.observe_t = self.t
            stop_bool = self.energy() < MIN_ENERGY or linalg.norm(self.state[0:2]) > MAP_DIM*PIXEL_METER
            done_bool = linalg.norm(self.state[0:2]) < DONE_R
            flag, v0, normal, t_fly = self.hypothesisSim()
        reward_value = self.reward(done_bool, stop_bool, pre_state, t0)
        return self.observe_state[0:6], self.terrain_map.map_matrix, reward_value, done_bool, stop_bool

    # check:
    # 碰撞仿真，直至所有顶点均与地面脱离
    # 更新t,state
    def collisionSim(self, flag, v0, normal):
        while flag.any():
            self.t, self.state = RK4(self.terrain_map, self.t, self.state, STEP_LENGTH, flag, v0, normal)
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
            slc1 = logical_and(flag, vertex_high > 0)
            flag[slc1] = False
            slc2 = logical_and(logical_not(flag), vertex_high <= 0)
            flag[slc2] = True
            normal[:, slc2] = self.terrain_map.get_normal(vertex_s[0, slc2], vertex_s[1, slc2])
            v0[slc2] = -sum(vertex_v[:, slc2] * normal[:, slc2], 0)

    # check:
    # 模拟无控仿真，直至有顶点与地面接触
    # 更新t,state，并返回进行碰撞仿真所需参数flag(碰撞标志),v0(碰撞点初始速度),normal(碰撞点初始法向量)
    # 额外返回这一段的仿真时长即探测器在空中的时间
    def hypothesisSim(self):
        t0 = self.t
        v0 = zeros(8)
        last_t, last_state = self.t, self.state.copy()
        flag = ones(8) < 0
        normal = ones([3, 8])
        vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        if (vertex_high <= 0).any():
            raise Exception("Invalid hypothesisSim!")
        # 先按0.1s的间隔仿真，找到碰撞的大致时间
        while (vertex_high > 0).all():
            last_t, last_state = self.t, self.state.copy()
            self.t, self.state = RK4(self.terrain_map, self.t, self.state, STEP_LENGTH * 1000)
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        # 再回到碰撞前，按0.001s的间隔仿真
        self.t, self.state = last_t, last_state.copy()
        vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        while (vertex_high > 0).all():
            self.t, self.state = RK4(self.terrain_map, self.t, self.state, STEP_LENGTH)
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        slc = vertex_high <= 0
        flag[slc] = True
        normal[:, slc] = self.terrain_map.get_normal(vertex_s[0, slc], vertex_s[1, slc])
        v0[slc] = -sum(vertex_v[:, slc] * normal[:, slc], 0)
        if (v0 < 0).any():
            raise Exception("Invalid v0!")
        return flag, v0, normal, self.t - t0

    # check:
    # 重定位，从虚拟的无控仿真得到的碰撞点定位到目标姿态下的碰撞点
    # 更新t,state,done(按碰撞点是否在设定目标区域判断)
    # 同hypothesisSim一样返回flag, v0, normal
    # 探测器在空中角动量守恒，可以计算出与action对应的飞轮转速
    def relocation(self, action):
        XYZc, v, q, w, w_wheel = self.state[0:3], self.state[3:6], self.state[6:10], self.state[10:13], self.state[
                                                                                                        13:16]
        self.state[6:10], self.state[10:13] = action[0:4].copy(), action[4:7].copy()
        self.state[13:16] = UJ_inv.dot(q_to_DCM(q).dot(q_to_DCM(q).T).dot(
            dot(I_star, w) + U.dot(J_wheel).dot(w_wheel)) - I_star.dot(w))
        vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        while (vertex_high <= 0).any() and self.t > self.observe_t:
            self.t -= STEP_LENGTH
            self.state[0:3] -= STEP_LENGTH * self.state[3:6]
            self.state[3:6] -= STEP_LENGTH * g
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        while (vertex_high <= 0).any():
            self.state[6:10] = self.observe_state[6:10].copy()
            self.t += STEP_LENGTH
            self.state[0:3] += STEP_LENGTH * self.state[3:6]
            self.state[3:6] += STEP_LENGTH * g
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        flag, v0, normal, t_fly = self.hypothesisSim()
        done_bool = linalg.norm(self.state[0:2]) < 8
        return done_bool, flag, v0, normal

    # 输出强化学习所需的输入，即【xyz，vxyz】,map_matrix
    def observe(self):
        return self.observe_state[0:6], self.terrain_map.map_matrix  # , self.map.local_map

    def reward(self, done_bool, stop_bool, pre_state, t0):
        if done_bool:
            reward_value = 1
        elif stop_bool:
            reward_value = -1
        else:
            reward_value = 0.001 * (linalg.norm(pre_state[0:2]) - linalg.norm(self.observe_state[0:2])) - 0.0001*(self.observe_t-t0)
        return reward_value

    def energy(self):
        v, w = self.state[3:6], self.state[10:13]
        eg = 0.5*m*dot(v, v) + 0.5*reshape(w, [1, 3]).dot(I_star).dot(w)
        return eg

    def test_ZeroMap(self):
        self.t = 0
        self.state = array([0, 0, 5, 0.12, -0.08, 0, 1, 0, 0, 0, 0.2, -0.1, 0.15, -1.9, 1.5, -1.2])
        self.terrain_map = TerrainMap(3, MAP_DIM, PIXEL_METER)
        flag, v0, normal, t_fly = self.hypothesisSim()
        self.collisionSim(flag, v0, normal)
        print(self.t)
        print(self.state)


if __name__ == '__main__':
    env = Env()
    sed = random.randint(1, 10000)
    env.set_map_seed(sed)
    if env.set_state_seed(sed):
        state_input, matrix_input, r, done, stop = env.step(array([1, 0, 0, 0, 1, -1, 0.5]))
        vertex_s1, vertex_high1, vertex_v1 = vertex(env.terrain_map, env.observe_state)
        print(state_input)
        print(r)
    # env.test_ZeroMap()