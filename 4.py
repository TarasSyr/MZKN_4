from main import signal_intensity_new, measure_number
import matplotlib.pyplot as plt
import numpy as np


filename="3_2"
third=__import__(filename)
a=third.a
measure_number=measure_number[:len(a)]

def zeroCrossings():
    zero_crossings = np.where(np.diff(np.sign(a)))[0]
    # Побудова графіку
    plt.plot(measure_number, a)
    plt.scatter(np.array(measure_number)[zero_crossings], np.zeros_like(zero_crossings), color='red', label='Нулі')
    plt.show()

#zeroCrossings()

from scipy.optimize import root_scalar

# Функція для знаходження нуля в заданому інтервалі
def find_zero(f, x0, x1):
    res = root_scalar(f, bracket=[x0, x1], method='brentq')
    if res.converged:
        return res.root
    return None

# Масив для зберігання точок нульових перетинів
zeros = []
for i in range(len(measure_number) - 1):
    x0, x1 = measure_number[i], measure_number[i+1]
    y0, y1 = a[i], a[i+1]
    if y0 * y1 < 0:  # Є зміна знаку між y0 та y1
        zero = find_zero(lambda x: np.interp(x, [x0, x1], [y0, y1]), x0, x1)
        if zero is not None:
            zeros.append(zero)

# Побудова графіку з нулями
plt.grid(True)
plt.plot(measure_number[:len(a)], a, linestyle="--", color='lightgreen')
plt.scatter(zeros, [0] * len(zeros), color='red', label='Нулі')
plt.show()