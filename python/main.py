import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import __db as db


class end_error(Exception):
    pass


def histoogramm(data):
    plt.subplot(3, 1, 1)
    plt.plot(data["time_ms"], linewidth=2.0)
    plt.xlabel("Число ядер")
    plt.ylabel("Время")
    plt.grid(True)
    plt.title(data["name"])

    plt.subplot(3, 1, 3)
    plt.plot(data["speed"], linewidth=2.0)
    plt.ylabel("Скорость")
    plt.xlabel("Число ядер")
    plt.grid(True)
    if not os.path.exists("results"):
        os.mkdir("results")
    name = "results/%s" % data["name"]
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    data1 = db.read("1.json", "graphics")
    histoogramm(data1)
