import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *

MAX_PEAK = 20
MIN_PEAK = 3
MAX_VALLEY = 10


def gauss_fcn(mu, sigma, x, y):
    zx = (x - mu[0]) ** 2 / (2 * sigma[0] ** 2)
    zy = (y - mu[1]) ** 2 / (2 * sigma[1] ** 2)
    z = 10 * exp(- zx - zy)
    return z


def gauss_diff(mu, sigma, x, y):
    z = gauss_fcn(mu, sigma, x, y)
    dx = -(x - mu[0]) * z / (sigma[0] ** 2)
    dy = -(y - mu[1]) * z / (sigma[1] ** 2)
    return stack([dx, dy, zeros(shape(x))], 0)


class MoonMap:
    def __init__(self, sd, map_dim, global_pixel_meter):
        # TerrianMap 构建高度图的对象
        random.seed(sd)
        self.map_dim = map_dim
        self.global_pixel_meter = global_pixel_meter
        side = map_dim * global_pixel_meter
        num_hole = array([random.poisson(8), random.poisson(4), random.poisson(0.4)])
        hole_pool = []
        for s in range(3):
            for i in range(num_hole[s]):
                # c_theta = random.random() * 2 * pi
                c = array([random.random(), random.random()]) * side - side / 2 - 10
                hole_pool.append(_hole(s, c))
        num_rock = array([random.poisson(15), random.poisson(8), random.poisson(1)])
        rock_pool = []
        for s in range(3):
            for i in range(num_rock[s]):
                rock_pool.append(_rock(s, zeros(2), map_dim * global_pixel_meter, 8))
        self.hole_pool = hole_pool
        self.rock_pool = rock_pool
        self.side = side
        u = arange(global_pixel_meter / 2, side + global_pixel_meter / 2, global_pixel_meter) - side / 2
        v = u
        [U, V] = meshgrid(u, v)
        map_matrix = zeros([map_dim, map_dim])
        for i in range(map_dim):
            for j in range(map_dim):
                map_matrix[i, j] = self.get_high(U[i, j], V[i, j])
        self.map_matrix = map_matrix

    def get_high(self, x, y):
        # get_high 求对应点的高度
        z = 0.
        for hole in self.hole_pool:
            if hole.in_or_out(x, y):
                z += hole.get_high(x, y)
        for rock in self.rock_pool:
            if rock.in_or_out(x, y):
                z += rock.get_high(x, y)
        return z

    def get_normal(self, x, y):
        n = zeros(3)
        for hole in self.hole_pool:
            if hole.in_or_out(x, y):
                dx, dy = hole.get_diff(x, y)
                n[0] += dx
                n[1] += dy
        for rock in self.rock_pool:
            if rock.in_or_out(x, y):
                dx, dy = rock.get_diff(x, y)
                n[0] += dx
                n[1] += dy
        n[2] = -1
        n /= -linalg.norm(n)
        return n

    def get_local_map(self, loc):
        side = self.side
        gpm = self.global_pixel_meter
        index = asarray((loc + side / 2) / gpm, dtype=int)
        loc_m = zeros([self.map_dim, self.map_dim])
        if (index >= 0).all():
            loc_m[index[0], index[1]] = 1
        return loc_m

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        side = self.map_dim * self.global_pixel_meter
        gap = 0.5
        u = arange(gap / 2, side + gap / 2, gap) - side / 2
        v = u
        [U, V] = meshgrid(u, v)
        Z = zeros(shape(U))
        for i in range(shape(Z)[0]):
            for j in range(shape(Z)[1]):
                Z[i, j] = self.get_high(U[i, j], V[i, j])
        ax.plot_surface(U, V, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        ax.set_zlim([-30, 30])

        # ax.contourf(U, V, Z, zdir='z', offset=-60, cmap=plt.get_cmap('rainbow'))
        plt.show()


class _rock:
    # s石块直径大小类别0，1，2；c所在坑中心坐标；d所在坑直径
    def __init__(self, s, c, d, min=0):
        if s == 2:
            self._d = abs(random.normal(0, 1)) + 16  # 石块直径
        elif s == 1:
            self._d = abs(random.normal(0, 3)) + 8  # 石块直径
        else:
            self._d = abs(random.normal(0, 1)) + 4  # 石块直径
        self._h = self._d / 2  # 石块高度
        c_theta = random.random() * 2 * pi
        self._c = (random.random() * (d / 2 - self._d / 2 - min) + min) * array([cos(c_theta), sin(c_theta)]) + c

    def get_high(self, a, b):
        x, y = a - self._c[0], b - self._c[1]
        h = self._h
        d = self._d
        z = h - 4*(x ** 2 + y ** 2) * h / d ** 2
        return z

    def get_diff(self, a, b):
        x, y = a - self._c[0], b - self._c[1]
        h = self._h
        d = self._d
        dx = -2 * h * x / d ** 2
        dy = -2 * h * y / d ** 2
        return dx, dy

    # 判断点（a，b）是否在石块范围内，true为in， false为out
    def in_or_out(self, a, b):
        x, y = a - self._c[0], b - self._c[1]
        r = sqrt(x ** 2 + y ** 2)
        return r < self._d / 2


class _hole:
    # s直径大小，分三类，0最小，2最大；c中心xy坐标
    def __init__(self, s, c):
        self.c = c
        if s == 0:
            self.d = abs(random.normal(0, 4)) + 8  # d 坑直径
            self.rock_pool = []
        elif s == 1:
            self.d = abs(random.normal(0, 4)) + 20
            num_rock_0 = random.poisson(0.01 * pi * (self.d / 2) ** 2)  # 坑内部石块数量,0表示是最小类型的石块，2最大
            self.rock_pool = []
            for i in range(num_rock_0):
                self.rock_pool.append(_rock(0, c, self.d))
        else:
            self.d = abs(random.normal(0, 4)) + 32
            num_rock_0 = random.poisson(0.01 * pi * (self.d / 2) ** 2 / 2)
            num_rock_1 = random.poisson(0.001 * pi * (self.d / 2) ** 2)
            self.rock_pool = []
            for i in range(num_rock_0):
                self.rock_pool.append(_rock(0, c, self.d))
            for i in range(num_rock_1):
                self.rock_pool.append(_rock(1, c, self.d))
        self.d_ = self.d / (random.random() * 2 + 9)  # d_坑唇宽度,d/(9--11)
        self.h = self.d * (random.random() * 0.14 + 0.11)  # h坑深,0.23--0.25d
        self.h_ = self.d * (random.random() * 0.052 + 0.008)  # h_坑唇高0.022--0.06d
        self.a = -4 * self.h_ / self.d_ ** 2  # a抛物线系数

    def get_high(self, a, b):
        z = 0.
        if self.in_or_out(a, b):
            z = self.get_hole_high(a, b)
            for rock in self.rock_pool:
                if rock.in_or_out(a, b):
                    z += rock.get_high(a, b)
        return z

    def get_diff(self, a, b):
        dx = 0.
        dy = 0.
        if self.in_or_out(a, b):
            dx, dy = self.get_hole_diff(a, b)
            for rock in self.rock_pool:
                if rock.in_or_out(a, b):
                    dx.dy += rock.get_diff(a, b)
        return dx, dy

    def get_hole_high(self, a, b):
        x, y = a - self.c[0], b - self.c[1]
        r = sqrt(x ** 2 + y ** 2)
        if r < self.d / 2:
            z = 4 * self.h * r ** 2 / self.d ** 2 - self.h
        else:
            z = self.h_ + self.a * (r - (self.d + self.d_) / 2) ** 2
        return z

    def get_hole_diff(self, a, b):
        x, y = a - self.c[0], b - self.c[1]
        r = sqrt(x ** 2 + y ** 2)
        if r < self.d / 2:
            dx = 8 * self.h * x / self.d ** 2
            dy = 8 * self.h * y / self.d ** 2
        else:
            dx = 2 * self.a * (r - (self.d + self.d_) / 2) * x / r
            dy = 2 * self.a * (r - (self.d + self.d_) / 2) * y / r
        return dx, dy

    # 判断点（a，b）是否在坑内，true为in， false为out
    def in_or_out(self, a, b):
        x, y = a - self.c[0], b - self.c[1]
        r = sqrt(x ** 2 + y ** 2)
        return r < self.d / 2 + self.d_


if __name__ == '__main__':
    sed = random.randint(1, 10000)
    rock = _rock(2, zeros(2), 64, 8)
    c = rock._c
    d = rock._d
    h1 = rock.get_high(c[0], c[1])
    h3 = rock.get_high(c[0] - d/2, c[1])
    h2 = rock.get_high(c[0] - d, c[1])
    m = MoonMap(sed, 32, 4)
    m.plot()
    print(m.map_matrix)
    print(m.get_normal(30, 40))
    loc1 = array([-33, 21.4])
    locm = m.get_local_map(loc1)
    print(locm)
