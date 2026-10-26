import numpy as np
import matplotlib.pyplot as plt

# Вариант 7: обучающая выборка
data_x = [(3.0, 1.3), (3.4, 1.6), (3.4, 0.4), (3.7, 0.2), (3.5, 0.2), (3.4, 0.2), (3.4, 0.4), (3.9, 0.4),
          (3.4, 0.3), (3.2, 0.2), (2.8, 1.3), (3.5, 0.3), (2.4, 1.0), (3.0, 0.1), (3.6, 0.2), (3.2, 0.2),
          (2.9, 0.2), (2.9, 1.3), (2.3, 1.3), (3.8, 0.2), (3.2, 1.5), (2.3, 1.0), (3.0, 1.7), (3.3, 0.2),
          (3.4, 0.2), (3.8, 0.3), (2.0, 1.0), (3.1, 0.2), (2.5, 1.3), (2.4, 1.1), (3.2, 0.2), (2.2, 1.0),
          (3.1, 1.4), (3.0, 0.2), (3.0, 0.2), (3.4, 0.2), (3.7, 0.2), (2.8, 1.2), (2.9, 1.4), (4.0, 0.2),
          (3.2, 1.4), (3.2, 0.2), (2.9, 1.3), (2.9, 1.3), (3.5, 0.2), (3.3, 1.6), (2.9, 1.3), (2.7, 1.0),
          (2.9, 1.3), (3.4, 0.2), (3.2, 0.2), (4.1, 0.1), (3.5, 0.6), (2.7, 1.4), (2.3, 0.3), (2.9, 1.5),
          (3.1, 1.5), (3.5, 0.2), (2.7, 1.6), (3.3, 0.5), (3.0, 1.4), (3.6, 0.2), (3.0, 1.2), (2.8, 1.3),
          (2.5, 1.1), (3.0, 1.5), (3.1, 0.2), (2.6, 1.0), (2.7, 1.2), (2.2, 1.5), (3.7, 0.4), (3.4, 0.2),
          (3.5, 0.3), (3.6, 0.1), (2.5, 1.5), (2.6, 1.2), (2.8, 1.3), (3.1, 0.1), (2.4, 1.0), (3.1, 1.5),
          (2.3, 1.3), (2.8, 1.5), (3.0, 0.3), (3.0, 0.2), (2.5, 1.1), (3.0, 1.5), (3.2, 1.8), (3.9, 0.4),
          (2.8, 1.4), (4.2, 0.2), (3.4, 0.2), (2.7, 1.3), (3.8, 0.3), (3.0, 1.4), (2.6, 1.2), (4.4, 0.4),
          (3.8, 0.4), (3.1, 0.2), (3.0, 0.1), (3.0, 1.5)]

data_y = [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1,
          1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1,
          1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1,
          -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1]

x_train = np.array(data_x)
y_train = np.array(data_y)
classes = np.array([-1, 1])

# обучение: априорные вероятности классов, средние и дисперсии признаков в каждом классе
prior = np.array([np.mean(y_train == c) for c in classes])
mean = np.array([x_train[y_train == c].mean(axis=0) for c in classes])
var = np.array([x_train[y_train == c].var(axis=0) for c in classes])  # оценка max правдоподобия (1/n)

print("Класс | P(y) | mu1    mu2    | sigma1^2  sigma2^2")
for k, c in enumerate(classes):
    print("%+4d  | %.2f | %.4f %.4f | %.5f   %.5f" % (c, prior[k], mean[k, 0], mean[k, 1], var[k, 0], var[k, 1]))


# логарифм гауссовской плотности одного признака
def log_gauss(x, mu, s2):
    return -0.5 * np.log(2 * np.pi * s2) - (x - mu) ** 2 / (2 * s2)


# оценки ln P(y) + sum_j ln p(x_j | y) для каждого объекта и каждого класса
def scores(x):
    return np.array([np.log(prior[k]) + log_gauss(x, mean[k], var[k]).sum(axis=1)
                     for k in range(len(classes))]).T


def predict(x):
    return classes[np.argmax(scores(x), axis=1)]


y_pred = predict(x_train)
errors = np.sum(y_pred != y_train)
print()
print("Число неверных классификаций = %d из %d" % (errors, len(y_train)))
print("Доля неверных классификаций = %.2f" % (errors / len(y_train)))

# апостериорные вероятности P(y|x) для 5 объектов, ближайших к границе классов
s = scores(x_train)
post = np.exp(s - s.max(axis=1, keepdims=True))
post = post / post.sum(axis=1, keepdims=True)
print()
print("Объект       | y  | P(-1|x)  P(+1|x) | a(x)")
for i in np.argsort(np.abs(post[:, 1] - 0.5))[:5]:
    print("(%.1f, %.1f)   | %+d | %.4f   %.4f   | %+d"
          % (x_train[i, 0], x_train[i, 1], y_train[i], post[i, 0], post[i, 1], y_pred[i]))

# график: объекты выборки и граница между классами (где оценки классов равны)
g1, g2 = np.meshgrid(np.linspace(1.8, 4.6, 300), np.linspace(0, 2.2, 300))
grid = np.c_[g1.ravel(), g2.ravel()]
s = scores(grid)
diff = (s[:, 1] - s[:, 0]).reshape(g1.shape)

pos = y_train == 1
neg = y_train == -1
plt.contourf(g1, g2, diff > 0, levels=1, colors=["#dde6ff", "#ddf5dd"])
plt.contour(g1, g2, diff, levels=[0], colors="red")
plt.scatter(x_train[pos, 0], x_train[pos, 1], c="green", marker="o", label="класс +1")
plt.scatter(x_train[neg, 0], x_train[neg, 1], c="blue", marker="x", label="класс -1")
plt.plot([], [], c="red", label="граница классов")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("result.png")
plt.show()
