import matplotlib.pyplot as plt

data = [
    {"config": "[35, 30, 20, 2, 20, 30, 35]", "err0": 1, "err1": 4},
    {"config": "[35, 32, 28, 2, 28, 32, 35]", "err0": 3, "err1": 4},
    {"config": "[35, 50, 25, 2, 25, 50, 35]", "err0": 3, "err1": 3},
    {"config": "[35, 80, 50, 2, 50, 80, 35]", "err0": 9, "err1": 1},
    {"config": "[35, 30, 20, 10, 2, 10, 20, 30, 35]", "err0": 2, "err1": 5},
    {"config": "[35, 32, 28, 20, 2, 20, 28, 32, 35]", "err0": 4, "err1": 3},
    {"config": "[35, 40, 30, 20, 2, 20, 30, 40, 35]", "err0": 4, "err1": 4},
    {"config": "[35, 50, 30, 20, 2, 20, 30, 50, 35]", "err0": 7, "err1": 2},
    {"config": "[35, 80, 50, 30, 2, 30, 50, 80, 35]", "err0": 8, "err1": 2},
]

strings = [
    "35-30-20-2-20-30-35",
    "35-32-28-2-28-32-35",
    "35-50-25-2-25-50-35",
    "35-80-50-2-50-80-35",
    "35-30-20-10-2-10-20-30-35",
    "35-32-28-20-2-20-28-32-35",
    "35-40-30-20-2-20-30-40-35"
    "35-50-30-20-2-20-30-50-35"
    "35-80-50-30-2-30-50-80-35"
]



total_runs = 10

labels = [f"C{i+1}" for i in range(len(data))]   # etiquetas cortas en X
err0_perc = [d["err0"] / total_runs * 100 for d in data]
err1_perc = [d["err1"] / total_runs * 100 for d in data]
totals_perc = [e0 + e1 for e0, e1 in zip(err0_perc, err1_perc)]

fig, ax = plt.subplots()
ax.bar(labels, err0_perc, label="Error 0", color='darkturquoise')
ax.bar(labels, err1_perc, bottom=err0_perc, label="Error 1", color='paleturquoise')

for x, total in zip(labels, totals_perc):
    ax.text(x, total + 1, f"{total:.0f}%", ha="center", va="bottom")

ax.set_ylabel("Porcentaje de corridas con error ≤ 1")
ax.set_title("Proporción de corridas con error 0 y 1 por configuración")
ax.legend()
plt.tight_layout()
plt.savefig("error_by_config.png")
plt.show()

for label, d in zip(labels, data):
    print(f"{label}: {d['config']}")
