import tensorflow as tf
import numpy as np

class WrapperPDE(object):
    """docstring"""

    def __init__(self, model):
        """Constructor"""
        self.__model = model
        self.__function = None
        self.__init_cond = None
        self.__init_mask = None
        # This flag is imposed due to probable accuracy losses
        # while working on a large number of iterations.
        self.__optimization_mode = False


    def set_function(self, function):
        self.__function = function


    def set_init_cond(self, init_cond):
        self.__init_cond = init_cond


    def set_init_mask(self, init_mask):
        self.__init_mask = init_mask


    def set_optimization_mode(self, optimization_mode):
        self.__optimization_mode = optimization_mode


    def residual(self, features):
        return self.__function(features)


    def d(self, features, degree = 1):
        if (degree == 0):
            u = self.u(features)
            return u

        with tf.GradientTape() as g:
            g.watch(features)
            u = self.d(features, degree - 1)
            du = g.gradient(u, features)
        return du


    def u(self, features):
        return self.__model(features)


    def loss(self, features):
        res = [self.__common_loss(features), self.__start_loss(features)]
        return res


    def __common_loss(self, features):
        residual = self.residual(features)
        if (not self.__optimization_mode):
            residual = tf.math.reduce_sum(residual, axis=1)
        return tf.math.square(residual)


    def __start_loss(self, features):
        N = int(tf.math.reduce_sum(self.__init_mask))
        predictions = self.__model(features)
        diff = tf.math.square(tf.squeeze(predictions) - self.__init_cond)
        start_loss = tf.multiply(diff, self.__init_mask)
        start_loss = tf.expand_dims(start_loss, axis=1)*N
        return start_loss