import numpy as np
import matplotlib.pyplot as plt

# Вариант 7
# Разделяющая линия на графике - вертикальная прямая x1 = 6
# w1*x1 + w2*x2 + w0 = 0  ->  -x1 + 6 = 0
w = np.array([6, -1, 0])  # (w0, w1, w2)

# точки с графика (x1, x2) и метки классов
points = np.array([
    [2, 6],  # зеленая
    [7, 3],  # зеленая
    [4, 2],  # синяя
    [9, 2],  # синяя
])
y = np.array([1, 1, -1, -1])

# добавляем фиктивный признак x0 = 1
X = np.hstack([np.ones((len(points), 1)), points])

dot = X @ w                                  # <w, x> для всех точек сразу
M = y * dot                                  # отступы
dist = np.abs(dot) / np.linalg.norm(w[1:])   # расстояние до линии

print("Разделяющая линия: %d*x1 %+d*x2 %+d = 0" % (w[1], w[2], w[0]))
print("w = (w0, w1, w2) =", w)
print()
for (x1, x2), yi, d, m, r in zip(points, y, dot, M, dist):
    color = "зеленая" if yi == 1 else "синяя"
    print("%-8s x=(%d, %d), y=%2d, <w,x> = %2d, M = %2d, расстояние = %g"
          % (color, x1, x2, yi, d, m, r))

# график
fig, ax = plt.subplots(figsize=(7, 5))
colors = np.where(y == 1, "green", "blue")
ax.scatter(points[:, 0], points[:, 1], c=colors, s=60, zorder=3)
ax.axvline(-w[0] / w[1], color="red", linewidth=2, label="x1 = %g" % (-w[0] / w[1]))

for (x1, x2), m in zip(points, M):
    ax.annotate("M = %d" % m, (x1, x2), textcoords="offset points", xytext=(6, 6))

ax.scatter([], [], c="green", label="класс +1")
ax.scatter([], [], c="blue", label="класс −1")
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.set_xticks(range(12))
ax.set_yticks(range(9))
ax.grid(True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Вариант 7: разделяющая линия и отступы")
ax.legend()
plt.savefig("lab2_plot.png", dpi=120)
plt.show()
