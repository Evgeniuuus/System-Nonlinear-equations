import matplotlib.pyplot as plt
import sympy as sp
from functions import *
from random import randint
import subprocess

F_x_y_1 = "3*x^2 - y = 0"
F_x_y_2 = "x^2 + y^2 - 4 = 0"

print("Функция 1: ", F_x_y_1)
print("Функция 2: ", F_x_y_2)

x_1 = np.linspace(-1, 1, 1000)
y_1 = 3 * x_1 ** 2

x_2 = np.linspace(-2, 2, 1000)
y_up = np.sqrt(4 - x_2 ** 2)
y_down = -np.sqrt(4 - x_2 ** 2)

root_x = [0.7832125644, -0.7832125644]
root_y = [1.8402657631, 1.8402657631]

plt.title('График', fontsize=14, fontname='Times New Roman')
ax = plt.gca()
ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-3, 3])
plt.plot(x_1, y_1, '--', color="green", label="3*x^2 - y = 0")
plt.plot(x_2, y_up, color="red", label="x^2 + y^2 - 4 = 0")
plt.plot(x_2, y_down, color="red")
plt.scatter(root_x, root_y, color='blue', label='Найденные корни')

plt.grid()
plt.legend()
plt.show()

# subprocess.run(["python", "R.py"])

print('\n-----------------------Метод простой итерации----------------------')

epsilon = np.double(input("Введите Эпсилон: "))
#epsilon = 0.001

x_0, y_0 = 1, 1
switch = True
k = 0

while True:

    x_next = phi_1(x_0, y_0)
    y_next = phi_2(x_0, y_0)

    k += 1

    if abs(x_next - x_0) < epsilon and abs(y_next - y_0) < epsilon:
        print(f"Метод сошелся через {k} итераций")
        break

    if np.isnan(x_next) or np.isnan(y_next):
        print("Итерации расходятся (NaN)")
        switch = False
        break

    x_0, y_0 = x_next, y_next

print(f"Решение: x = {x_0:.10f}, y = {y_0:.10f}")
if not switch:
    print("Решение ошибочно")

print("\n---------------------------Метод Ньютона----------------------------")

x, y = sp.symbols('x y')

F_x_y_1 = 3 * x ** 2 - y
F_x_y_2 = x ** 2 + y ** 2 - 4

# Частные производные (матрица Якоби)
J = np.array([[sp.diff(F_x_y_1, x), sp.diff(F_x_y_1, y)],
              [sp.diff(F_x_y_2, x), sp.diff(F_x_y_2, y)]])


# Функция для вычисления значений системы
def F(x_val, y_val):
    return np.array([F_x_y_1.subs({x: x_val, y: y_val}),
                     F_x_y_2.subs({x: x_val, y: y_val})], dtype=float)


# Функция для вычисления Якобиана
def Jacobian(x_val, y_val):
    return np.array([[J[0, 0].subs({x: x_val, y: y_val}), J[0, 1].subs({x: x_val, y: y_val})],
                     [J[1, 0].subs({x: x_val, y: y_val}), J[1, 1].subs({x: x_val, y: y_val})]], dtype=float)


# Метод Ньютона
def newton_method(x0, y0, epsilon):
    x_k, y_k = x0, y0
    k = 0
    while True:
        F_k = F(x_k, y_k)
        J_k = Jacobian(x_k, y_k)

        try:
            delta = np.linalg.solve(J_k, -F_k)  # Решение системы линейных уравнений
        except np.linalg.LinAlgError:
            print("Якобиан вырожденный, метод не применим")
            return None

        x_k1, y_k1 = x_k + delta[0], y_k + delta[1]
        k += 1

        if np.linalg.norm(delta) < epsilon:
            print(f"Метод Ньютона сошелся через {k} итераций")
            return x_k1, y_k1

        if k > 1000:
            print("Метод не сошелся")
            return None

        x_k, y_k = x_k1, y_k1


epsilon = float(input("Введите Эпсилон: "))
result = newton_method(1, 1, epsilon)

if result:
    print(f"Решение: x = {result[0]:.10f}, y = {result[1]:.10f}")


'''
n = (3, 3)
A1 = np.array([[1., 2., 3.], [3., 5., 7.], [1., 3., 4.]])
A2 = np.zeros(n)

A3 = np.zeros(n)
A4 = np.zeros(n)
A5 = np.zeros(n)
f1 = np.array([3, 0, 1])
f2 = np.zeros(n[0])
f3 = np.zeros(n[0])
f4 = np.zeros(n[0])
f5 = np.zeros(n[0])

for i in range(n[0]):
    for j in range(n[0]):
        A2[i][j] = randint(-10, 10)
        if i == j:
            A3[i][j] = 1
        A5[i][j] = 1 / (i + j + 1)
    f2[i] = randint(-10, 10)
    f3[i] = randint(-10, 10)
    f4[i] = randint(-10, 10)
    f5[i] = randint(-10, 10)

print(A1, f1)
print(A2, f2)
print(A3, f3)
print(A4, f4)
print(A5, f5)

tmp = np.c_[A5, f5]
print(tmp)

print(EasyGauss(tmp))
print(NotSoEasyGauss(np.c_[A5, f5]))
print('Решение: ', get_answer(NotSoEasyGauss(np.c_[A5, f5])))
print('Проверка: ', np.dot(A5, get_answer(NotSoEasyGauss(np.c_[A5, f5]))))
'''
