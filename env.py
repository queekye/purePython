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
    M_wheel = zeros(3)
    d_w_wheel = UJ_inv.dot(M_wheel)
    F = zeros([3, 8])
    T = zeros([3, 8])
    DCM = q_to_DCM(q)
    vertex_s, vertex_high, vertex_v = vertex(terrain_map, state)
    Teq = cross(w, I_star.dot(w) + U.dot(J_wheel).dot(w_wheel)) + U.dot(J_wheel).dot(d_w_wheel)
    for j in range(0, 8):
        if flag[j] and vertex_high[j] <= 0:
            r_one = vertex_b[:, j]
            high = vertex_high[j]
            normal_one = normal[:, j]
            invade_s = -dot(array([0, 0, high]), normal_one)
            vertex_v_one = vertex_v[:, j]
            invade_v = -dot(vertex_v_one, normal_one)
            vt = vertex_v_one - dot(vertex_v_one, normal_one) * normal_one
            vt_value = sqrt(dot(vt, vt))
            if abs(v0[j]) < 0.00001:
                v0[j] = 0.00001 * sign(v0[j])
            c = 0.75 * (1 - recovery_co ** 2) * k * (invade_s ** 1.5) / v0[j]
            Fn_value = k * (invade_s ** 1.5) + c * invade_v
            if Fn_value < 1e-8:
                Fn_value = 1e-8
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
            F[:, j] = Ft + Fn
            T[:, j] = cross(r_one, DCM.dot(Ft + Fn))
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
    k1 = dynamic(terrain_map, state, flag.copy(), v0.copy(), normal.copy())
    k2 = dynamic(terrain_map, state + h * k1 / 2, flag.copy(), v0.copy(), normal.copy())
    k3 = dynamic(terrain_map, state + h * k2 / 2, flag.copy(), v0.copy(), normal.copy())
    k4 = dynamic(terrain_map, state + h * k3, flag.copy(), v0.copy(), normal.copy())
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
        self.state0 = self.state.copy()
        self.terrain_map = TerrainMap(3, MAP_DIM, PIXEL_METER)

    def baoluo(self, x0, y0):
        # 0.35略大于0.2*sqrt(3)
        u = array([x0 - 0.35, x0, x0 + 0.35])
        v = array([y0 - 0.35, y0, y0 + 0.35])
        U, V = meshgrid(u, v)
        return self.terrain_map.get_high(U, V)

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
        minVxy = 0.05
        maxVxy = 0.2
        XY_theta = random.random() * 2 * pi
        XY = ((maxXY - minXY) * random.random() + minXY) * array([cos(XY_theta), sin(XY_theta)])
        v_theta = random.random() * 2 * pi
        v_xy = ((maxVxy - minVxy) * random.random() + minVxy) * array([cos(v_theta), sin(v_theta)])
        vz = 0.07 * random.random() + 0.03
        q = random.rand(4)
        q /= linalg.norm(q)
        w = random.rand(3) * 2 - 1
        w_wheel = random.rand(3) * 2 - 1
        Z = ndarray.max(self.baoluo(XY[0], XY[1])) + 0.35
        self.state = concatenate([XY, array([Z]), v_xy, array([vz]), q, w, w_wheel])
        self.state0 = self.state.copy()
        self.t = 0

    # check:
    # step env接收agent的action后的状态转移
    # 输入action即碰前角速度与姿态，输出新的state，reward以及标志该次仿真是否达到目的地的done，是否速度过小导致停止的stop
    def step(self, action):
        pre_t = self.t
        pre_state = self.state.copy()
        self.state[6:13] = action.copy()
        flag, v0, normal0 = self.hypothesisSim()
        overtime, fall = self.collisionSim(flag, v0, normal0)

        stop_bool = self.energy() < MIN_ENERGY or linalg.norm(self.state[3:5]) < 1e-2 or abs(self.state[5]) < 1e-2
        over_map = linalg.norm(self.state[0:2]) > (MAP_DIM * PIXEL_METER)
        over_speed = linalg.norm(self.state[3:5]) > 1 or linalg.norm(self.state[5]) > 0.3
        done_bool = linalg.norm(self.state[0:2]) < DONE_R

        reward_value = self.reward(done_bool, stop_bool, pre_state, pre_t, over_speed, over_map, overtime, fall)
        done = done_bool or stop_bool or over_speed or overtime or over_map or fall
        return self.observe_state(), self.terrain_map.map_matrix, reward_value, done

    # check:
    # 碰撞仿真，直至所有顶点均与地面脱离
    # 更新t,state
    def collisionSim(self, flag, v0, normal0):
        t0 = self.t
        overtime = False
        fall = False
        vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        while flag.any() and self.t - t0 < 10:
            pre_t, pre_state = self.t, self.state.copy()
            self.t, self.state = RK4(self.terrain_map, self.t, self.state, STEP_LENGTH, flag, v0, normal0)
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
            slc1 = logical_and(flag, vertex_high > 0)
            flag[slc1] = False
            v0[slc1] = 0
            normal0[0:2, slc1] = 0
            normal0[2, slc1] = 1
            slc2 = logical_and(logical_not(flag), vertex_high < 0)
            if slc2.any():
                flag[slc2] = True
                self.t, self.state = pre_t, pre_state.copy()
                vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
                normal0[:, slc2] = self.terrain_map.get_normal(vertex_s[0, slc2], vertex_s[1, slc2])
                v0[slc2] = -sum(vertex_v[:, slc2] * normal0[:, slc2], 0)
        while (vertex_high > 0).all() and (self.state[2] - self.baoluo(self.state[0], self.state[1]) < 0.35).any:
            self.state[0:2] += self.state[3:5] * 1
            self.state[2] += self.state[5] * 1 + 0.5 * g[2] * 1 ** 2
            self.state[5] += g[2] * 1
            self.t += 1
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        if self.t - t0 >= 50:
            overtime = True
        if (vertex_high < 0).any():
            fall = True
        return overtime, fall

    # check:
    # 更新t,state，并返回进行碰撞仿真所需参数flag(碰撞标志),v0(碰撞点初始速度),normal(碰撞点初始法向量)
    # 额外返回这一段的仿真时长即探测器在空中的时间
    def hypothesisSim(self):
        flag = ones(8) < 0
        v_xy = self.state[3:5].copy()
        vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        delta_t = 1
        pre_t = self.t
        pre_state = self.state.copy()
        normal0 = self.terrain_map.get_normal(vertex_s[0, :], vertex_s[1, :])
        v0 = -sum(vertex_v[:, :] * normal0[:, :], 0)
        for i in range(4):
            while (vertex_high > 0).all():
                pre_t = self.t
                pre_state = self.state.copy()
                self.state[0:2] += v_xy*delta_t
                self.state[2] += self.state[5]*delta_t + 0.5*g[2]*delta_t**2
                self.state[5] += g[2]*delta_t
                self.t += delta_t
                vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
                flag = vertex_high <= 0
            delta_t /= 10
            self.t = pre_t
            self.state = pre_state.copy()
            vertex_s, vertex_high, vertex_v = vertex(self.terrain_map, self.state)
        normal0[:, flag] = self.terrain_map.get_normal(vertex_s[0, flag], vertex_s[1, flag])
        v0[flag] = -sum(vertex_v[:, flag] * normal0[:, flag], 0)

        return flag, v0, normal0

    # 输出强化学习的state
    def observe_state(self):
        o_s = self.state.copy()
        o_s[0:2] /= (MAP_DIM * PIXEL_METER / 2)
        o_s[0:2] = minimum(maximum(o_s[0:2], -1), 1 - 1e-3)
        o_s[-3:] /= 10
        o_s[10:13] /= 2
        return o_s

    def reward(self, done_bool, stop_bool, pre_state, pre_t, over_speed, over_map, overtime, fall):
        def _cos_vec(a, b):
            f = dot(a, b) / (linalg.norm(a) * linalg.norm(b))
            return f

        if done_bool:
            reward_value = 10
        elif over_map:
            reward_value = -2.5
        elif stop_bool or overtime or fall:
            reward_value = (linalg.norm(self.state0[0:2]) - linalg.norm(self.state[0:2])) / \
                           max(linalg.norm(self.state0[0:2]), linalg.norm(self.state[0:2]))
        elif over_speed:
            reward_value = -2
        else:
            d = (linalg.norm(pre_state[0:2]) - linalg.norm(self.state[0:2])) / \
                max(linalg.norm(pre_state[0:2]), linalg.norm(self.state[0:2]))
            c_pre = _cos_vec(-pre_state[0:2], pre_state[3:5])
            c = _cos_vec(-self.state[0:2], self.state[3:5])
            v_xy = linalg.norm(self.state[3:5])
            reward_value = (c - c_pre) + (v_xy * c - v_xy * sqrt(1 - c ** 2)) + d - 0.0001 * (self.t - pre_t)
        return reward_value

    def energy(self):
        v, w = self.state[3:6], self.state[10:13]
        eg = 0.5 * m * dot(v, v) + 0.5 * reshape(w, [1, 3]).dot(I_star).dot(w)
        return eg


if __name__ == '__main__':
    env = Env()
    sed = 100
    env.set_map_seed(sed)
    for i in range(100):
        env.set_state_seed(random.randint(0, 100000))
        act = random.random_sample(7)
        act[0:4] /= linalg.norm(act[0:4])
        state_input, matrix_input, r, done = env.step(act)
        print(state_input, r, done)
    # env.test_ZeroMap()
