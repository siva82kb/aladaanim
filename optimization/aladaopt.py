"""
Script defining different functions used for demonstration.

Author: Sivakumar Balasubramanian
Date: 06 March 2024
"""

import numpy as np


class Circle:
    def __init__(self, xmin: np.array):
        self.name = 'Circle'
        self.title = r"$(\mathbf{x - \mathbf{x}^*})^\top(\mathbf{x} - \mathbf{x}^*)$"
        self.xmin = xmin.reshape(-1, 1)
    
    def func(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return 0.5 * (_x1 ** 2 + _x2 ** 2)
    
    def grad(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return np.array([[_x1],
                         [_x2]])
    
    def hessian(self, x1, x2):
        return np.array([[1, 0],
                         [0, 1]])


class Ellipse:
    def __init__(self, xmin: np.array, Q: np.array):
        self.name = 'Ellipse'
        self.title = r"$\frac{1}{2}(\mathbf{x - \mathbf{x}^*})^\top\mathbf{Q}(\mathbf{x} -  - \mathbf{x}^*)$"
        self.xmin = xmin.reshape(-1, 1)
        self.Q = Q
    
    def func(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return 0.5 * (self.Q[0, 0] * _x1 ** 2
                + self.Q[1, 1] * _x2 ** 2
                + (self.Q[1, 0] + self.Q[0, 1]) * _x1 * _x2)
    
    def grad(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return 0.5 * np.array([
            [2 * self.Q[0, 0] * _x1 + (self.Q[1, 0] + self.Q[0, 1]) * _x2],
            [2 * self.Q[1, 1] * _x2 + (self.Q[1, 0] + self.Q[0, 1]) * _x1]
        ])
    
    def hessian(self, x1, x2):
        return self.Q


class Rosenbrock:
    def __init__(self, a: float, b: float):
        self.name = 'Rosenbrock'
        self.title = r"$(a - x_1)^2 + b (x_2 - x_1^2)^2$"
        self.xmin = np.array([[a], [a ** 2]])
        self.a, self.b = a, b
    
    def func(self, x1, x2):
        return (self.a - x1)**2 + self.b * (x2 - x1**2)**2
    
    def grad(self, x1, x2):
        return np.array([
            [(-2 * (self.a - x1)) - (4 * self.b * x1 * (x2 - x1**2))],
            [2 * self.b * (x2 - x1**2)]
        ])
    
    def hessian(self, x1, x2):
        return np.array([
            [2 - 4 * self.b * (x2 - 3 * x1**2), -4 * self.b * x1],
            [-4 * self.b * x1, 2 * self.b]
        ])


class Quartic:
    def __init__(self, xmin: np.array, a: float, b: float, c: float):
        self.name = 'Quartic'
        self.title = r"$a (x_1 - x_1^*)^4 + b (x_2 - x_2^*)^4 + c (x_1 - x_1^*)(x_2 - x_2^*)^3$"
        self.xmin = xmin.reshape(-1, 1)
        self.a, self.b, self.c = a, b, c
    
    def func(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return self.a * _x1 ** 4 + self.b * _x2 ** 4 + self.c * _x1 * _x2 ** 3
    
    def grad(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return np.array([
            [4 * self.a * _x1 ** 3 + self.c * _x2 ** 3],
            [4 * self.b * _x2 ** 3 + 3 * self.c * _x1 * _x2 ** 2]
        ])
            
    
    def hessian(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return np.array([
            [12 * self.a * _x1 ** 2, 3 * self.c * _x2 ** 2],
            [3 * self.c * _x2 ** 2, 12 * self.b * _x2 ** 2 + 6 * self.c * _x1 * _x2]
        ])


class FlippedGaussian:
    def __init__(self, xmin: np.array, Q: np.array):
        self.name = 'Flipped Gaussian'
        self.title = r"$1 - \exp\left(-\frac{1}{2}(\mathbf{x - \mathbf{x}^*})^\top\mathbf{Q}(\mathbf{x} - \mathbf{x}^*)\right)$"
        self.xmin = xmin.reshape(-1, 1)
        self.Q = Q

    def _v(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return 0.5 * (self.Q[0, 0] * _x1 ** 2
                    + self.Q[1, 1] * _x2 ** 2
                    + (self.Q[1, 0] + self.Q[0, 1]) * _x1 * _x2)
    
    def _dv1(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return self.Q[0, 0] * _x1 + 0.5 * (self.Q[1, 0] + self.Q[0, 1]) * _x2
    
    def _dv2(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return self.Q[1, 1] * _x2 + 0.5 * (self.Q[1, 0] + self.Q[0, 1]) * _x1

    def func(self, x1, x2):
        return 1 - np.exp(- self._v(x1, x2))
    
    def grad(self, x1, x2):
        _v = self._v(x1, x2)
        return np.exp(-_v) * np.array([
            [self._dv1(x1, x2)],
            [self._dv2(x1, x2)]
        ])
    
    def hessian(self, x1, x2):
        _v = self._v(x1, x2)
        _dv1 = self._dv1(x1, x2)
        _dv2 = self._dv2(x1, x2)
        return np.exp(-_v) * np.array([
            [self.Q[0, 0] - _dv1 ** 2, 0.5 * (self.Q[1, 0] + self.Q[0, 1]) - _dv1 * _dv2],
            [0.5 * (self.Q[1, 0] + self.Q[0, 1]) - _dv1 * _dv2, self.Q[1, 1] - _dv2 ** 2]
        ])


class RectangularHyperbola:
    def __init__(self):
        self.name = 'Rectangular Hyperbola'
        self.title = r"$-x_1 x_2$"

    def func(self, x1, x2):
        return - x1 * x2
    
    def grad(self, x1, x2):
        return np.array([
            [-x2],
            [-x1]
        ])
    
    def hessian(self, x1, x2):
        return np.array([
            [0, -1],
            [-1, 0]
        ])


# Gradient Descent Class
class GradientDescent:
    name = "Gradient Descent"
    
    @staticmethod
    def update(xk, ak, grad):
        return xk - ak * grad


# Newton-Raphson Class
class NewtonRaphson:
    name = "Newton-Raphson"
    
    @staticmethod
    def update(xk, grad, hess):
        return xk - np.linalg.solve(hess, grad)


# Levenberg-Marquardt Class
class LevenbergMarquardt:
    name = "Levenberg-Marquardt"
    
    @staticmethod
    def update(xk, grad, hess, mu):
        return xk - np.linalg.solve(hess + mu * np.eye(hess.shape[0]), grad)


# Equality contraint functions
class ParabolaEc():
    name = "Parabola"

    @staticmethod
    def func(t):
        return np.array([[t, (t / 2)**2]]).T

    @staticmethod
    def normal(x1, x2):
        return np.array([[0.5 * x1, -1]]).T
    
    @staticmethod
    def tangent(x1, x2):
        return np.array([[1, 0.5 * x1]]).T
    
    @staticmethod
    def equation():
        return r"$h(\mathbf{x}) = \frac{1}{4}x_1^2 - x_2 = 0$"
    

class CircleEc():
    name = "Circle"

    @staticmethod
    def func(t):
        return 2 * np.array([[np.cos(t), np.sin(t)]]).T

    @staticmethod
    def normal(x1, x2):
        return 2 * np.array([[x1, x2]]).T
    
    @staticmethod
    def tangent(x1, x2):
        return np.array([[-x2, x1]]).T
    
    @staticmethod
    def equation():
        return r"$h(\mathbf{x}) = \mathbf{x}^\top\mathbf{x} - 4 = 0$"
    

class EllipseEc():
    name = "Ellipse"
    
    @staticmethod
    def func(t):
        return np.array([[3 * np.cos(t), 1 * np.sin(t)]]).T

    @staticmethod
    def normal(x1, x2):
        return np.array([[(2/9) * x1, 2 * x2]]).T
    
    @staticmethod
    def tangent(x1, x2):
        return np.array([[-2 * x2, (2/9) * x1]]).T
    
    @staticmethod
    def equation():
        return r"$h(\mathbf{x}) = \mathbf{x}^\top\mathbf{Q}\mathbf{x} - 1 = 0$"
    

class ObliqueEllipseEc():
    name = "Ellipse"
    V = (1 / np.sqrt(2)) * np.array([[1,  1],
                                     [1, -1]])
    L = np.diag([1, np.sqrt(5)])
    Q = V @ L @ L @ V.T

    @staticmethod
    def func(t):
        _v = np.array([[np.cos(t), np.sin(t)]]).T
        return ObliqueEllipseEc.V @ np.linalg.inv(ObliqueEllipseEc.L) @ _v

    @staticmethod
    def normal(x1, x2):
        return 2 * ObliqueEllipseEc.Q @ np.array([[x1, x2]]).T
    
    @staticmethod
    def tangent(x1, x2):
        _n = ObliqueEllipseEc.normal(x1, x2)
        return np.array([[_n[1, 0], -_n[0, 0]]]).T
    
    @staticmethod
    def equation():
        return r"$h(\mathbf{x}) = \mathbf{x}^\top\mathbf{Q}\mathbf{x} - 1 = 0$"
    

class Ellipse2Ec():
    name = "Ellipse2"
    
    @staticmethod
    def func(t):
        return np.array([[np.cos(t), 0.5 * np.sin(t)]]).T

    @staticmethod
    def normal(x1, x2):
        return np.array([[-4 * x2, 2 * x1]]).T
    
    @staticmethod
    def tangent(x1, x2):
        return np.array([[2 * x1, 4 * x2]]).T
    
    @staticmethod
    def equation():
        return r"$h(\mathbf{x}) = x_1^2 + 2x_2^2 - 1 = 0$"


class Linear2Ec():
    name = "Ellipse2"
    c = np.array([[4., 8.]]).T

    @staticmethod
    def func(t):
        return np.array([[t, - t / 2]]).T + np.array([[0.5, 1.]]).T

    @staticmethod
    def normal(x1, x2):
        return Linear2Ec.c
    
    @staticmethod
    def tangent(x1, x2):
        return np.array([[0., -1.], [1., 0.]]) @ Linear2Ec.c
    
    @staticmethod
    def equation():
        return r"$h(\mathbf{x}) = 4x_1 + 8x_2 - 10 = 0$"


# class RectHyperbolaEc():
#     name = "RectangularHyperbola"
    
#     @staticmethod
#     def func(t):
#         return np.array([[np.cos(t), 0.5 * np.sin(t)]]).T

#     @staticmethod
#     def normal(x1, x2):
#         return np.array([[-4 * x2, 2 * x1]]).T
    
#     @staticmethod
#     def tangent(x1, x2):
#         return np.array([[2 * x1, 4 * x2]]).T
    
#     @staticmethod
#     def equation():
#         return r"$h(\mathbf{x}) = x_1^2 + 2x_2^2 - 1 = 0$"