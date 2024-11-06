import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, filtfilt, butter

from main import signal_intensity_new, measure_number


def adaptiveFilterWithAmplitudeCheck(signal_intensity_new):

    # Функції для фільтрів низьких та високих частот
    def lowpass_filter(data, normal_cutoff, fs, order=5):
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def highpass_filter(data, normal_cutoff, fs, order=5):
        if normal_cutoff > 1:  # Перевірка на перевищення діапазону
            normal_cutoff = 0.15  # Обмеження на максимальну частоту
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    sampling_rate = 10  # Частота дискретизації
    window_size = 170 # Зменшене вікно для покращення деталізації

    smoothed_signal = np.zeros(len(signal_intensity_new))  # Ініціалізація масиву для відфільтрованого сигналу

    for i in range(0, len(signal_intensity_new) - window_size + 1, window_size):
        y_segment = signal_intensity_new[i:i + window_size]

        # Перевірка на нульові значення
        if np.all(y_segment == 0):
            smoothed_signal[i:i + window_size] = 0  # Записуємо нулі
            continue  # Пропустіть обробку цього сегмента

        # Розрахунок частоти сегмента за допомогою дискретного перетворення Фур'є
        freq_spectrum = np.fft.fft(y_segment)
        freqs = np.fft.fftfreq(len(y_segment), d=1 / sampling_rate)

        # Залишаємо тільки позитивні частоти
        pos_mask = freqs >= 0
        freq_magnitudes = np.abs(freq_spectrum[pos_mask])
        freqs = freqs[pos_mask]

        # Знайдемо частоту з максимальною амплітудою
        segment_freq = freqs[np.argmax(freq_magnitudes)]

        # Перевірка амплітуди
        mean_amplitude = np.mean(np.abs(y_segment))
        significant_fluctuations = [y for y in y_segment if np.abs(y) > mean_amplitude * 1.5]

        # Якщо значущих коливань менше 10 або всі амплітуди менші за певний поріг, пропустіть сегмент
        if len(significant_fluctuations) < 5 or mean_amplitude < 0.01:
            continue  # Пропустіть сегмент

        # Визначення порогової частоти на основі частоти сегмента
        if segment_freq < 0.7:
            cutoff = 0.20  # Нижня частота для фільтра низьких частот
            filtered_segment = lowpass_filter(y_segment, cutoff, sampling_rate)
        else:
            cutoff = segment_freq * 1.8  # Вища частота для фільтра високих частот
            filtered_segment = highpass_filter(y_segment, cutoff, sampling_rate)

        # Додавання відфільтрованого сегмента до масиву
        smoothed_signal[i:i + window_size] = filtered_segment

    # Візуалізація результатів
    #plt.figure(figsize=(12, 6))
    #plt.plot(smoothed_signal, label='Сигнал з адаптивною фільтрацією', color='red')
    #plt.title('Адаптивна фільтрація сигналу з перевіркою амплітуди')
    return smoothed_signal

def sinusoidApprox(signal_intensity_new):
    def sinusoidal_func(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    # Параметри ковзного вікна
    window_size = 10  # Розмір вікна
    smoothed_signal = []

    for i in range(0, len(signal_intensity_new) - window_size, window_size):
        # Ділянка сигналу
        x_segment = measure_number[i:i + window_size]
        y_segment = signal_intensity_new[i:i + window_size]

        try:
            # Підбір параметрів для синусоїдної функції на даній ділянці
            params, _ = curve_fit(sinusoidal_func, x_segment, y_segment, p0=[1, 0.1, 0, 0], maxfev=5000)

            # Апроксимовані значення для поточної ділянки
            smoothed_segment = sinusoidal_func(np.array(x_segment), *params)
            smoothed_signal.extend(smoothed_segment)

        except RuntimeError:
            # Якщо не вдалось підібрати параметри, додаємо середнє значення сегмента
            smoothed_segment = [np.mean(y_segment)] * window_size
            smoothed_signal.extend(smoothed_segment)


    #plt.plot(measure_number[:len(smoothed_signal)], smoothed_signal, label='Синусоїдна апроксимація з ковзним вікном', color='red')
    return smoothed_signal

a=sinusoidApprox(adaptiveFilterWithAmplitudeCheck(signal_intensity_new))
plt.subplot(2,2,1)
plt.plot(measure_number, signal_intensity_new, label="Original Signal")

plt.subplot(2,2,2)
plt.plot(measure_number[:len(a)], a)

plt.subplot(2,1,2)
plt.plot(measure_number, signal_intensity_new, label="Original and Approx signal")
plt.plot(measure_number[:len(a)], a)

a=np.array(a)
#adaptiveFilterWithAmplitudeCheck(signal_intensity_new)
#sinusoidApprox(adaptiveFilterWithAmplitudeCheck(signal_intensity_new))

plt.show()
