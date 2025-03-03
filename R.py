import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x, y = sp.symbols('x y')

f1 = x ** 2 + y ** 2 - 4
f2 = 3 * x ** 2 - y

phi2 = 3 * x ** 2

phi1_pos = sp.sqrt(4 - y ** 2)
phi1_neg = -sp.sqrt(4 - y ** 2)

phi1_x = sp.diff(phi1_pos, x)  # ∂φ1/∂x = 0, так как x отсутствует в выражении
phi1_y = sp.diff(phi1_pos, y)
phi2_x = sp.diff(phi2, x)
phi2_y = sp.diff(phi2, y)  # ∂φ2/∂y = 0, так как y отсутствует в выражении

phi1_y_func = sp.lambdify(y, phi1_y, 'numpy')
phi2_x_func = sp.lambdify(x, phi2_x, 'numpy')

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)

Q1 = np.abs(phi1_y_func(Y)) + np.abs(phi2_x_func(X))
Q2 = np.abs(phi2_x_func(X)) + np.abs(phi1_y_func(Y))
R = (Q1 < 1) | (Q2 < 1)

plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.set_xlim([-0.2, 0.2])
ax.set_ylim([-1.5, 1.5])
plt.title("Область R")
plt.xlabel("x")
plt.ylabel("y")
plt.contourf(X, Y, R, alpha=0.3, cmap='Blues', levels=[0.5, 1])
plt.grid()
plt.show()

