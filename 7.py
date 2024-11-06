from matplotlib import pyplot as plt
from scipy.stats import alpha

from main import signal_intensity_new, measure_number
from scipy.signal import butter, filtfilt


filename="3_2"
third=__import__(filename)
a_from_3=third.a
measure_number=measure_number[:len(a_from_3)]


def applyFilterBand():
    order = 4  # Порядок фільтра
    lowcut = 0.1  # Нижня частота зрізу
    highcut = 0.5  # Верхня частота зрізу
    fs = 10
    b, a = butter(order, [lowcut, highcut], fs=fs, btype="band")

    # Фільтрація сигналу
    filtered_signal = filtfilt(b, a, signal_intensity_new)  # signal_intensity - ваш сигнал

    # Візуалізація
    # Тримайте x-ві координати такої ж довжини, як і signal_intensity_new
    # Обрізаємо signal_intensity_new до довжини measure_number
    #plt.plot(measure_number, signal_intensity_new[:len(measure_number)], label="Оригінальний сигнал", color="blue")

    # Перший підплан (2x2, перший графік)
    plt.subplot(2, 2, 1)
    plt.plot(measure_number, filtered_signal[:len(measure_number)], label="Фільтрований сигнал по діапазону частот",
             color="red")
    plt.title("Фільтрований сигнал по діапазону частот")

    # Другий підплан (2x2, другий графік)
    plt.subplot(2, 2, 2)
    plt.plot(measure_number, a_from_3, label="Відфільтрований сигнал по фільтру з 3 завдання")
    plt.title("Відфільтрований сигнал з 3 завдання")

    # Третій підплан (2x1, другий графік)
    plt.subplot(2, 1, 2)
    plt.plot(measure_number, signal_intensity_new[:len(measure_number)], label="Оригінальний сигнал", color="blue")
    plt.plot(measure_number, filtered_signal[:len(measure_number)], label="Фільтрований сигнал по діапазону частот", color="red")
    plt.plot(measure_number, a_from_3, "green",alpha=0.5, label="Відфільтрований сигнал по фільтру з 3 завдання")
    plt.legend()
    plt.title("Оригінальний та фільтровані сигнали")

    plt.show()


applyFilterBand()