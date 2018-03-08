import numpy
from scipy.spatial.distance import euclidean


class DMM:

    def __init__(self):
        self.__models = None

    def train(self, X, y):

        t = list(zip(X, y))

        c1 = filter(lambda x: x[1] == 0, t)
        cx1, cy1= zip(*c1)
        c1_mean = sum(cx1) / len(cx1)

        c2 = filter(lambda x: x[1] == 1, t)
        cx2, cy2= zip(*c2)
        c2_mean = sum(cx2) / len(cx2)

        c3 = filter(lambda x: x[1] == 2, t)
        cx3, cy3= zip(*c3)
        c3_mean = sum(cx3) / len(cx3)

        self.__models = [c1_mean, c2_mean, c3_mean]

    def __calc_distance(self, data, mean):
        return euclidean(data, mean)

    def __distances(self, models, X):
        for x in X:
            yield [self.__calc_distance(x, m) for m in models]

    def predict(self, X):
        distances_euclidian = list(self.__distances(self.__models, X))
        return [numpy.argmin(d) for d in distances_euclidian]


class K1NN:

    def __init__(self):
        self.__models = None

    def train(self, X, y):
        self.__models = list(zip(y, X))

    def __calc_distance(self, data, mean):
        return euclidean(data, mean)

    def predict(self, X):
        for x in X:
            distances = [(label, self.__calc_distance(x, model)) for label, model in self.__models]
            dy, dx = zip(*distances)
            idx = numpy.argmin(dx)
            yield dy[idx]