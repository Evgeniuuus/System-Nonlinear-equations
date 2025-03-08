# Вариант № 3
# Анатолий 6 И на лабу 3

import numpy as np
import sympy as sp


# ----------------------Для метода простой итерации---------------------
def phi_1(x, y):
    return np.sqrt(y / 3)


def phi_2(x, y):
    return np.sqrt(4 - x ** 2)


# ---------------------------Для метода Ньютона--------------------------
x, y = sp.symbols('x y')

F_x_y_1 = 3 * x ** 2 - y
F_x_y_2 = x ** 2 + y ** 2 - 4

J = np.array([[sp.diff(F_x_y_1, x), sp.diff(F_x_y_1, y)],  # [[6*x -1]
              [sp.diff(F_x_y_2, x), sp.diff(F_x_y_2, y)]])  # [2*x 2*y]]


def F(x_val, y_val):
    return np.array([F_x_y_1.subs({x: x_val, y: y_val}),
                     F_x_y_2.subs({x: x_val, y: y_val})], dtype=float)


def Jacobian(x_val, y_val):
    return np.array([[J[0, 0].subs({x: x_val, y: y_val}), J[0, 1].subs({x: x_val, y: y_val})],
                     [J[1, 0].subs({x: x_val, y: y_val}), J[1, 1].subs({x: x_val, y: y_val})]], dtype=float)


def f1(arg1, arg2):
    return arg1 ** 2 + arg2 ** 2 - 4


def f2(arg1, arg2):
    return 3 * arg1 ** 2 - arg2


def check(root1, root2, epsilon):
    if abs(f1(root1, root2)) < epsilon and abs(f2(root1, root2)) < epsilon:
        print(f1(root1, root2))
        print(f2(root1, root2))
        return "Найден верно."
    else:
        print(f1(root1, root2))
        print(f2(root1, root2))
        return "Найден неверно."


'''
def EasyGauss(a: np.array):
    def div_row(a: np.array, n: int, divider: float):
        for j in range(len(a[i])):
            a[n][j] /= divider
        return a

    def sub_rows(a: np.array, n: int):
        for i in range(len(a)):
            k = a[i][n]
            for j in range(len(a[i])):
                if i != n:
                    a[i][j] -= a[n][j] * k
        return a

    for i in range(len(a)):
        # print(a)
        if a[i][i] != 0:
            a = div_row(a, i, a[i][i])
            a = sub_rows(a, i)
    return a


def NotSoEasyGauss(a: np.array):
    tmp = a
    # s = 1
    print('матрица:')
    print(tmp)
    for i in range(len(tmp)):
        k = np.argmax(abs(tmp[i:, i])) + i
        if i != k:
            tmp[[i, k]] = tmp[[k, i]]
            # s*=-1
        # print(tmp,s)

    def div_row(a: np.array, n: int, divider: float):
        for j in range(len(a[i])):
            a[n][j] /= divider
        return a

    def sub_rows(a: np.array, n: int):
        for i in range(len(a)):
            k = a[i][n]
            for j in range(len(a[i])):
                if i != n:
                    a[i][j] -= a[n][j] * k
        return a

    for i in range(len(a)):
        # print(a)
        if a[i][i] != 0:
            a = div_row(a, i, a[i][i])
            a = sub_rows(a, i)
    return a


def get_answer(a: np.array):
    answ = []
    if len(a) == len(a[0]) - 1:
        for i in range(len(a)):
            for j in range(len(a[0])):
                if i == j:
                    answ.append(a[i][-1] / a[i][j])
        return answ
    else:
        return []
'''
