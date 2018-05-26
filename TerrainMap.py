import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *

MAX_PEAK = 10
MIN_PEAK = 3
MAX_VALLEY = 5


def gauss_fcn(mu, sigma, x, y):
    zx = (x - mu[0])**2 / (2 * sigma[0] ** 2)
    zy = (y - mu[1])**2 / (2 * sigma[1] ** 2)
    z = 10 * exp(- zx - zy)
    return z


def gauss_diff(mu, sigma, x, y):
    z = gauss_fcn(mu, sigma, x, y)
    dx = -(x - mu[0]) * z / (sigma[0] ** 2)
    dy = -(y - mu[1]) * z / (sigma[1] ** 2)
    return stack([dx, dy, zeros(shape(x))], 0)


class TerrainMap:
    def __init__(self, sd, map_dim, global_pixel_meter):
        # TerrianMap 构建高度图的对象
        random.seed(sd)
        self.map_dim = map_dim
        self.global_pixel_meter = global_pixel_meter
        self.num_peak = random.randint(MIN_PEAK, MAX_PEAK)
        self.num_valley = random.randint(0, MAX_VALLEY)
        side = map_dim * global_pixel_meter
        self.side = side
        self.center = random.rand(self.num_peak + self.num_valley, 2) * side - side/2
        self.sigma = abs(
            random.normal(0, global_pixel_meter, [self.num_peak + self.num_valley, 2])) + global_pixel_meter

        u = arange(global_pixel_meter / 2, side + global_pixel_meter / 2, global_pixel_meter) - side / 2
        v = u
        [U, V] = meshgrid(u, v)
        self.map_matrix = self.get_high(U, V)

    def get_high(self, x, y):
        # get_high 求对应点的高度
        #   x，y可为实数或列向量或矩阵，输出与其维度相同
        s = shape(x)
        z = zeros(s)
        for i in range(0, self.num_peak):
            mu = self.center[i, ...]
            sigma = self.sigma[i, ...]
            z += gauss_fcn(mu, sigma, x, y)

        for i in range(0, self.num_valley):
            mu = self.center[i + self.num_peak, ...]
            sigma = self.sigma[i + self.num_peak, ...]
            z -= gauss_fcn(mu, sigma, x, y)

        return z

    def get_normal(self, x, y):
        s = concatenate([array([3], dtype=int), array(shape(x), dtype=int)])
        n = zeros(s)
        for i in range(0, self.num_peak):
            mu = self.center[i, ...]
            sigma = self.sigma[i, ...]
            n += gauss_diff(mu, sigma, x, y)

        for i in range(0, self.num_valley):
            mu = self.center[i + self.num_peak, ...]
            sigma = self.sigma[i + self.num_peak, ...]
            n -= gauss_diff(mu, sigma, x, y)

        n[2] = -1
        n /= -linalg.norm(n, axis=0)
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
        u = arange(self.global_pixel_meter / 2, side - self.global_pixel_meter / 2) - side / 2
        v = u
        [U, V] = meshgrid(u, v)
        Z = self.get_high(U, V)
        ax.plot_surface(U, V, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        plt.show()


if __name__ == '__main__':
    sed = random.randint(1, 10000)
    m = TerrainMap(sed, 27, 9)
    m.plot()
    print(m.map_matrix)
    print(m.get_normal(30, 40))
    print(m.get_normal(array([1, 2, 3]), array([2, 4, 6])))
    loc1 = array([-33, 21.4])
    print(m.get_local_map(loc1))
    for i in range(10000):
        a = m.get_high(28, -33)
