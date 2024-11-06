from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
from main import signal_intensity_new, measure_number

filename="3_2"
third=__import__(filename)
a=third.a
measure_number=measure_number[:len(a)]


def zPeregynamu():
    # Знаходимо перегини за зміною величини сигналу
    inflections = []
    for i in range(len(a) - 170):
        if abs(a[i + 170] - a[i]) < 0.1:
            inflections.append(i)

    # Тепер для знайдених перегинів знаходимо максимуми та мінімуми
    max_peaks, _ = find_peaks(a)
    min_peaks, _ = find_peaks(-a)

    # Відбираємо максимуми та мінімуми в межах знайдених перегинів
    final_max_peaks = []
    final_min_peaks = []

    for inflection in inflections:
        # Зберігаємо діапазон для перегину
        start = max(0, inflection)
        end = min(len(a), inflection + 5)

        # Знаходимо максимуми і мінімуми в межах цього діапазону
        max_in_range = [peak for peak in max_peaks if start <= peak < end]
        min_in_range = [peak for peak in min_peaks if start <= peak < end]

        # Якщо є максимуми, зберігаємо їх
        if max_in_range:
            final_max_peaks.extend(max_in_range)

        # Якщо є мінімуми, зберігаємо їх
        if min_in_range:
            final_min_peaks.extend(min_in_range)

    # Виводимо результати
    print("Фінальні максимуми (після фільтрації):", final_max_peaks)
    print("Фінальні мінімуми (після фільтрації):", final_min_peaks)

    # Візуалізація
    max_measure_points = [measure_number[i] for i in final_max_peaks]
    max_signal_points = [a[i] for i in final_max_peaks]

    min_measure_points = [measure_number[i] for i in final_min_peaks]
    min_signal_points = [a[i] for i in final_min_peaks]

    plt.scatter(max_measure_points, max_signal_points, color="red", label="Максимуми перегинів")
    plt.scatter(min_measure_points, min_signal_points, color="green", label="Мінімуми перегинів")


def bezPeregyniv():
    max_peaks, _ = find_peaks(a)
    min_peaks, _ = find_peaks(-a)

    # Відображення максимумів і мінімумів
    plt.scatter(max_peaks, a[max_peaks], color="red", label="Максимум")
    plt.scatter(min_peaks, a[min_peaks], color="green", label="Мінімум")


#zPeregynamu()

def peregunu(measure_number, a, threshold=6e-2):
    first_derivative = np.gradient(a, measure_number)
    second_derivative = np.gradient(first_derivative, measure_number)
    inflection_points_x = []
    inflection_points_y = []
    inflection_points=[]
    for i in range(1, len(second_derivative) - 1):
    # Зміна знака другої похідної
        if (second_derivative[i - 1] > 0 and second_derivative[i] < 0) or (second_derivative[i - 1] < 0 and second_derivative[i] > 0):
        # Перевірка на близькість до нуля першої похідної
            if np.isclose(first_derivative[i], 0, atol=threshold):
                inflection_points_x.append(measure_number[i])
                inflection_points_y.append(a[i])
                inflection_points.append((measure_number[i],a[i]))

    return [inflection_points_x, inflection_points_y]

plt.plot(measure_number, a)
#plt.scatter(peregunu(measure_number,a)[0],peregunu(measure_number,a)[1], color="red")
zPeregynamu()
plt.legend()
plt.show()
