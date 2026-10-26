import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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

# сравнение разных значений параметра регуляризации C
print("   C  |   w1      w2      w0   | ошибок | опорных | ширина полосы")
for c in [0.1, 1, 10, 100, 1000]:
    model = SVC(kernel="linear", C=c).fit(x_train, y_train)
    w = model.coef_[0]
    errors = np.sum(model.predict(x_train) != y_train)
    print("%5g | %6.3f  %6.3f  %6.3f | %6d | %7d | %.4f"
          % (c, w[0], w[1], model.intercept_[0], errors, len(model.support_), 2 / np.linalg.norm(w)))

# основная модель: выборка линейно разделима, поэтому берем большое C (жесткий зазор)
model = SVC(kernel="linear", C=1000)
model.fit(x_train, y_train)

w = model.coef_[0]        # w1, w2
w0 = model.intercept_[0]  # свободный член
y_pred = model.predict(x_train)
errors = np.sum(y_pred != y_train)

print()
print("w1 = %.4f, w2 = %.4f, w0 = %.4f" % (w[0], w[1], w0))
print("Разделяющая линия: %.4f*x1 %+.4f*x2 %+.4f = 0" % (w[0], w[1], w0))
print("Ширина разделяющей полосы 2/|w| = %.4f" % (2 / np.linalg.norm(w)))
print("Опорные векторы (по классам %s):" % model.n_support_)
for i in model.support_:
    print("   x = %s, y = %+d, M = %.4f" % (x_train[i], y_train[i], y_train[i] * (x_train[i] @ w + w0)))
print("Число неверных классификаций = %d из %d" % (errors, len(y_train)))
print("Доля неверных классификаций = %.2f" % (errors / len(y_train)))

# график
line_x = np.linspace(1.8, 4.6, 2)
line_y = -(w0 + w[0] * line_x) / w[1]         # w1*x1 + w2*x2 + w0 = 0
margin_up = -(w0 - 1 + w[0] * line_x) / w[1]  # граница полосы для класса +1
margin_dn = -(w0 + 1 + w[0] * line_x) / w[1]  # граница полосы для класса -1

pos = y_train == 1
neg = y_train == -1
plt.scatter(x_train[pos, 0], x_train[pos, 1], c="green", marker="o", label="класс +1")
plt.scatter(x_train[neg, 0], x_train[neg, 1], c="blue", marker="x", label="класс -1")
sv = x_train[model.support_]
plt.scatter(sv[:, 0], sv[:, 1], s=150, facecolors="none", edgecolors="black", label="опорные векторы")
plt.plot(line_x, line_y, c="red", label="разделяющая линия")
plt.plot(line_x, margin_up, "r--", linewidth=0.8, label="границы полосы")
plt.plot(line_x, margin_dn, "r--", linewidth=0.8)
plt.ylim(0, 2.4)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc="upper center", ncol=3, fontsize="small")
plt.grid(True)
plt.savefig("result.png")
plt.show()
