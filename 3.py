import matplotlib.pyplot as plt
import numpy as np
from fontTools.ttLib.tables.otTables import splitMarkBasePos
from numpy.ma.extras import polyfit
from scipy.interpolate import make_interp_spline, interp1d, Rbf, BSpline, UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, filtfilt, butter

from main import signal_intensity_new, measure_number

measure_number = np.array(measure_number)
plt.subplot(2,2,1)
plt.plot(measure_number, signal_intensity_new, label="Original Signal")

plt.subplot(2,1,2)
plt.plot(measure_number, signal_intensity_new, label="Original and Approx signal")

def chooseApprox(a):

    if a==1:
        plt.subplot(2, 2, 2)
    elif a==2:
        plt.subplot(2, 1, 2)
    x_new = np.linspace(measure_number.min(), measure_number.max(), 1000)

    def polinomialnaAprox():
        degree = 60
        polyf = np.polyfit(measure_number, signal_intensity_new, degree)
        # Обчислюємо значення полінома для кожного значення measure_number
        poly_values = np.polyval(polyf, measure_number)
        return poly_values

    #plt.plot(measure_number, polinomialnaAprox(), label="Polynomial Fit", linestyle="--")

    def cubicSpline():
        spline=make_interp_spline(measure_number,signal_intensity_new,k=3)
        splineApprox=spline(x_new)
        return splineApprox

    #plt.plot(x_new, cubicSpline(), label="Cubic spline")

    def linear():
        linear_interp = interp1d(measure_number, signal_intensity_new, kind='linear')
        linearApprox = linear_interp(x_new)
        return linearApprox

    #plt.plot(x_new, linear(), label="Linear Approx")

    def cubic():
        cubic_interp = interp1d(measure_number, signal_intensity_new, kind='cubic')
        cubicApprox = cubic_interp(x_new)
        return cubicApprox

    #plt.plot(x_new, cubic(), label="Cubic Approx")

    def smoothSpline():
        knots = np.linspace(measure_number.min(), measure_number.max(), 100)  # Кількість вузлів
        coeffs = np.random.rand(len(knots) + 2)  # Випадкові коефіцієнти
        degree = 3  # Ступінь B-сплайна

        # B-сплайн
        b_spline = BSpline(knots, coeffs, degree)
        b_spline_fit = b_spline(x_new)
        return b_spline_fit

    #plt.plot(x_new, smoothSpline(), label="Smooth Spline Approx")

    def alsoSmoothSpline():
        spline = UnivariateSpline(measure_number, signal_intensity_new, s=0.05)  # s контролює гладкість
        spline_fit = spline(x_new)
        return spline_fit

    #plt.plot(x_new, alsoSmoothSpline(), label="Also smooth spline Approx")

    def radialBasusna():
        rbf = Rbf(measure_number, signal_intensity_new, function='multiquadric')  # Можливі функції: 'multiquadric', 'inverse', 'gaussian', тощо.
        rbf_fit = rbf(x_new)
        return rbf_fit

    #plt.plot(x_new, radialBasusna(), label="Radial-Basis Approx")

#chooseApprox(1)
#chooseApprox(2)

def dynamicFreqApproxFilter(signal_intensity_new):


    def sinusoidal_func(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    # Функції для фільтрів низьких та високих частот
    def lowpass_filter(data, normal_cutoff, fs, order=5):
        #nyquist = 0.5 * fs
        #normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def highpass_filter(data, normal_cutoff, fs, order=5):
        #nyquist = 0.5 * fs
        #normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    # Параметри ковзного вікна
    window_size = 30  # Кількість точок у кожному сегменті
    sampling_rate = 10  # Частота дискретизації для фільтрів
    low_cutoff = 0.03  # Порогова частота для фільтра низьких частот
    high_cutoff = 0.5  # Порогова частота для фільтра високих частот

    smoothed_signal = []

    for i in range(0, len(signal_intensity_new) - window_size, window_size):
        x_segment = measure_number[i:i + window_size]
        y_segment = signal_intensity_new[i:i + window_size]

        # Проводимо синусоїдну апроксимацію для сегмента
        try:
            params, _ = curve_fit(sinusoidal_func, x_segment, y_segment, p0=[1, 0.1, 0, 0], maxfev=10000)
            smoothed_segment = sinusoidal_func(np.array(x_segment), *params)
        except RuntimeError:
            smoothed_segment = np.mean(y_segment) * np.ones(window_size)

        # Розрахунок частоти сегмента
        segment_freq = params[1] / (2 * np.pi)  # Перетворюємо на частоту в Гц

        # Вибір фільтра на основі частоти
        if segment_freq < 0.1:
            # Використовуємо фільтр низьких частот
            filtered_segment = lowpass_filter(smoothed_segment, low_cutoff, sampling_rate)
        else:
            # Використовуємо фільтр високих частот
            filtered_segment = highpass_filter(smoothed_segment, high_cutoff, sampling_rate)

        smoothed_signal.extend(filtered_segment)

    # Візуалізація результатів
    plt.plot(measure_number[:len(smoothed_signal)], signal_intensity_new[:len(smoothed_signal)],
             label='Оригінальний сигнал', color='blue')
    plt.plot(measure_number[:len(smoothed_signal)], smoothed_signal,
             label='Синусоїдна апроксимація з адаптивною фільтрацією', color='red')


#dynamicFreqApproxFilter()

def adaptiveFilter():

    # Функції для фільтрів низьких та високих частот
    def lowpass_filter(data, normal_cutoff, fs, order=5):
        #nyquist = 0.5 * fs
        #normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def highpass_filter(data, normal_cutoff, fs, order=5):
        #nyquist = 0.5 * fs
        #normal_cutoff = cutoff / nyquist
        if normal_cutoff > 1:  # Перевірка на перевищення діапазону
            normal_cutoff = 0.15  # Обмеження на максимальну частоту
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    # Параметри ковзного вікна
    window_size = 100  # Кількість точок у кожному сегменті
    sampling_rate = 10  # Частота дискретизації для фільтрів

    smoothed_signal = []

    for i in range(0, len(signal_intensity_new) - window_size, window_size):
        y_segment = signal_intensity_new[i:i + window_size]

        # Розрахунок частоти сегмента за допомогою дискретного перетворення Фур'є
        freq_spectrum = np.fft.fft(y_segment)
        freqs = np.fft.fftfreq(len(y_segment), d=1 / sampling_rate)
        # Залишаємо тільки позитивні частоти
        pos_mask = freqs >= 0
        freq_magnitudes = np.abs(freq_spectrum[pos_mask])
        freqs = freqs[pos_mask]

        # Знайдемо частоту з максимальною амплітудою
        segment_freq = freqs[np.argmax(freq_magnitudes)]

        # Визначаємо порогову частоту на основі частоти сегмента
        if segment_freq < 0.7:
            cutoff = 0.20  # Нижня частота для фільтра низьких частот
            filtered_segment = lowpass_filter(y_segment, cutoff, sampling_rate)
        else:
            cutoff = segment_freq * 1.8  # Вища частота для фільтра високих частот
            filtered_segment = highpass_filter(y_segment, cutoff, sampling_rate)

        smoothed_signal.extend(filtered_segment)

    # Візуалізація результатів
    plt.plot(measure_number[:len(smoothed_signal)], signal_intensity_new[:len(smoothed_signal)],
             label='Оригінальний сигнал', color='blue')
    plt.plot(measure_number[:len(smoothed_signal)], smoothed_signal, label='Сигнал з адаптивною фільтрацією',
             color='red')


# Викликаємо функцію для фільтрації та малювання графіка
#adaptiveFilter()

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
    plt.figure(figsize=(12, 6))
    plt.plot(signal_intensity_new, label='Оригінальний сигнал', color='blue', alpha=0.5)
    plt.plot(smoothed_signal, label='Сигнал з адаптивною фільтрацією', color='red')
    plt.title('Адаптивна фільтрація сигналу з перевіркою амплітуди')
    plt.xlabel('Час (вибірки)')
    plt.ylabel('Амплітуда')
    plt.legend()
    plt.grid()
    plt.show()
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
            params, _ = curve_fit(sinusoidal_func, x_segment, y_segment, p0=[1, 0.1, 0, 0], maxfev=10000)

            # Апроксимовані значення для поточної ділянки
            smoothed_segment = sinusoidal_func(np.array(x_segment), *params)
            smoothed_signal.extend(smoothed_segment)

        except RuntimeError:
            # Якщо не вдалось підібрати параметри, додаємо середнє значення сегмента
            smoothed_segment = [np.mean(y_segment)] * window_size
            smoothed_signal.extend(smoothed_segment)

    # Побудова оригінального та згладженого сигналу
    plt.plot(measure_number[:len(smoothed_signal)], signal_intensity_new[:len(smoothed_signal)],
             label='Оригінальний сигнал', color='blue')
    plt.plot(measure_number[:len(smoothed_signal)], smoothed_signal, label='Синусоїдна апроксимація з ковзним вікном',
             color='red')


adaptiveFilterWithAmplitudeCheck(signal_intensity_new)
sinusoidApprox(adaptiveFilterWithAmplitudeCheck(signal_intensity_new))
plt.show()

