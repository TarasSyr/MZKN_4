from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
from main import signal_intensity_new, measure_number

filename="3_2"
third=__import__(filename)
a=third.a
measure_number=measure_number[:len(a)]

filename="5_5"
fifth=__import__(filename)
real_signal_max=fifth.real_signal_max
real_signal_min=fifth.real_signal_min
inflections=fifth.inflections

number_min_max=len(real_signal_max)+len(real_signal_min)
number_inflections=len(inflections)

with open("inflections.txt", "w", encoding="utf-8") as f:
    f.write(f"Точки перегинів: {inflections}")
    f.close()

plt.title("Сигнал з перегинами")
plt.plot(measure_number, a,label="Сигнал")
plt.plot(inflections, a[inflections], "ro", label="Перегини")

plt.show()

print(f"Кількість максимальних та мінімальних значень(без перегинів): {number_min_max}")
print(f"Кількість перегинів: {number_inflections}")

if number_min_max > number_inflections:
    print(f"Максимумів і мінімумів більше ніж перегинів на {number_min_max-number_inflections}")
elif number_min_max==number_inflections:
    print("Кількість (максимумів і мінімумів) та перегинів однакова")
else:
    print(f"Перегинів більше ніж максимумів і мінімумів на {number_inflections-number_min_max}")