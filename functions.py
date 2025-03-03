# Вариант № 3
# Анатолий 6

import numpy as np


def phi_1(x, y):
    return np.sqrt(y / 3)


def phi_2(x, y):
    return np.sqrt(4 - x ** 2)


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
