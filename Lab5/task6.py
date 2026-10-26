import numpy as np
import matplotlib.pyplot as plt


# Вариант 7: исследуемая функция
def f(x):
    return x ** 4 - 10 * x ** 3 + 20 * x ** 2 - 100 * x


x = np.arange(0, 10.05, 0.1)  # x ∈ [0; 10] с шагом 0.1, 101 отсчет
y = f(x)
N = 13                        # степень полинома

X_all = np.array([[a ** n for n in range(N + 1)] for a in x])  # строки (1, x, x^2, ..., x^13)
X = X_all[::2]                # обучающая выборка: четные индексы x0, x2, ..., x100
Y = y[::2]
print("Отсчетов всего: %d, в обучающей выборке: %d" % (len(x), len(Y)))
print("Число обусловленности X^T X = %.2e" % np.linalg.cond(X.T @ X))


def mse(a, b):
    return np.mean((a - b) ** 2)


# 1) без регуляризации: w = (X^T X)^-1 X^T Y
w = np.linalg.inv(X.T @ X) @ X.T @ Y
a = X_all @ w                 # прогноз на всем диапазоне
print()
print("Без регуляризации:")
print("  w =", np.array2string(w, precision=4))
print("  MSE на обучающих отсчетах = %.2f, на остальных = %.2f" % (mse(a[::2], y[::2]), mse(a[1::2], y[1::2])))

# 2) с L2-регуляризатором: w = (X^T X + lambda*I)^-1 X^T Y, подбор lambda
print()
print("  lambda  | MSE обуч. | MSE остальн. |   |w|")
for lam_i in [0.01, 1, 10, 100, 1000, 10000, 100000]:
    w_i = np.linalg.inv(X.T @ X + lam_i * np.eye(N + 1)) @ X.T @ Y
    a_i = X_all @ w_i
    print("%9g | %9.2f | %12.2f | %.2e" % (lam_i, mse(a_i[::2], y[::2]), mse(a_i[1::2], y[1::2]), np.linalg.norm(w_i)))

lam = 1000
w_l2 = np.linalg.inv(X.T @ X + lam * np.eye(N + 1)) @ X.T @ Y
a_l2 = X_all @ w_l2
print()
print("С L2-регуляризатором, lambda = %g:" % lam)
print("  w =", np.array2string(w_l2, precision=4))
print("  MSE на обучающих отсчетах = %.2f, на остальных = %.2f"
      % (mse(a_l2[::2], y[::2]), mse(a_l2[1::2], y[1::2])))
print("  Число обусловленности X^T X + lambda*I = %.2e" % np.linalg.cond(X.T @ X + lam * np.eye(N + 1)))

# проверка: тот же МНК без явного обращения X^T X (через SVD матрицы X)
w_ls = np.linalg.lstsq(X, Y, rcond=None)[0]
a_ls = X_all @ w_ls
print()
print("Проверка (lstsq, без обращения X^T X): MSE на обучающих = %.4f, на остальных = %.4f"
      % (mse(a_ls[::2], y[::2]), mse(a_ls[1::2], y[1::2])))

# графики
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for k, (pred, title) in enumerate([(a, "без регуляризации"), (a_l2, "L2-регуляризатор, λ = %g" % lam)]):
    ax[k].plot(x, y, c="black", label="f(x)")
    ax[k].scatter(x[::2], Y, s=12, c="blue", label="обучающая выборка")
    ax[k].plot(x, pred, c="red", label="модель a(x)")
    ax[k].set_ylim(-1500, 1500)
    ax[k].set_title(title)
    ax[k].set_xlabel("x")
    ax[k].set_ylabel("y")
    ax[k].legend()
    ax[k].grid(True)
plt.tight_layout()
plt.savefig("result_task6.png")
plt.show()
