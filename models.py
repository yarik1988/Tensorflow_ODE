import numpy as np
import tensorflow as tf


class Simple:
    """
    $x\'=sin(t)/(1+t+x)$
    """

    T = np.array([0, 10])  # interval
    pos = np.array([0])  # initial condition positions
    val = np.array([0])  # initial condition values
    dim = len(val)

    @staticmethod
    def equation(t, z):
        return [np.sin(t)/(1 + t + z)]

    @staticmethod
    def tf_equation(t, z):
        return tf.math.divide(tf.math.sin(t),(1 + z + t))


class TestModel1:
    """
    $x_0\'=x_1$
    $x_1\'=6x_0^2$
    """

    T = np.array([1, 2])  # interval
    pos = np.array([1, 1])  # initial condition positions
    val = np.array([1, -2])  # initial condition values
    dim = len(val)

    @staticmethod
    def equation(t, z):
        return [z[1], 6 * z[0] * z[0]]

    @staticmethod
    def tf_equation(t, z):
        return [z[:, 1], 6 * tf.math.square(z[:, 0])]

    @staticmethod
    def theoretical(t):
        res = np.vstack((1 / np.power(t, 2), -2 / np.power(t, 3)))
        return np.transpose(res)


class TestModel2:
    """
    $x_0\'=x_1$
    $x_1\'=x_0^2-x_0-1$
    """

    T = np.array([0, 5])  # interval
    pos = np.array([0, 0])  # initial condition positions
    val = np.array([0, 1])  # initial condition values
    dim = len(val)

    @staticmethod
    def equation(t, z):
        return [z[1],z[0]*z[0]-z[0]-1]

    @staticmethod
    def tf_equation(t, z):
        return [z[:, 1], tf.math.square(z[:, 0])-z[:, 0]-1]


class TestModel3:
    """
    $x_0\'=-x_1-x_0^2$
    $x_1\'=2x_0-x_1^3$
    """

    T = np.array([0, 20])  # interval
    pos = np.array([0, 0])  # initial condition positions
    val = np.array([1, 1])  # initial condition values
    dim = len(val)

    @staticmethod
    def equation(t, z):
        return [-z[1]-z[0]*z[0], 2*z[0]-np.power(z[1], 3)]

    @staticmethod
    def tf_equation(t, z):
        return [-z[:, 1]-tf.math.square(z[:, 0]), 2*z[:, 0]-tf.math.pow(z[:, 1], 3)]