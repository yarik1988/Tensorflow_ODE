import tensorflow as tf
import numpy as np

class WrapperODE(object):
    """docstring"""

    def __init__(self, model):
        """Constructor"""
        self.__model = model
        self.__function = None
        self.__start_values = None
        self.__derivatives = []
        self.__keep_derivatives = False


    def set_function(self, function):
        self.__function = function


    def set_start_values(self, start_values):
        self.__start_values = start_values


    def residual(self, x):
        return self.__function(x)


    def d(self, x, degree = 1):

        if (len(self.__derivatives) > degree):
            return self.__derivatives[degree]

        if (degree == 0):
            y = self.__model(x)
            self.__add_to_derivitives(y)
            return y

        with tf.GradientTape() as g:
            g.watch(x)
            y = self.d(x, degree - 1)
            dy = g.gradient(y, x)
        self.__add_to_derivitives(dy)
        return dy


    def y(self, x):
        return self.d(x, 0)


    def loss(self, x):
        self.__set_keep_derivatives(True)
        res = [self.__common_loss(x), self.__start_loss(x)]
        self.__set_keep_derivatives(False)
        return res


    def __common_loss(self, x):
        return tf.math.square(self.residual(x))


    def __start_loss(self, x):
        n = x.shape[0]
        values_number = self.__start_values.size

        res = np.zeros(values_number)
        for i in range(values_number):
            res += tf.math.square(self.d(x, i)[0] - self.__start_values[i])
        res *= n
        res = tf.concat([res, tf.zeros(n - 1, dtype='float64')], 0)
        return res


    def __set_keep_derivatives(self, on):
        self.__keep_derivatives = on
        if (not on):
            self.__derivatives = []


    def __add_to_derivitives(self, dy):
        if (self.__keep_derivatives):
            self.__derivatives.insert(len(self.__derivatives), dy)