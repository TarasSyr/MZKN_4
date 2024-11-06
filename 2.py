import numpy as np
from matplotlib import *
from matplotlib import pyplot as plt
from scipy import *
from scipy.signal import butter, filtfilt

from main import measure_number, signal_intensity_new

sampling_rate = 100  # Частота дискретизації у Гц (приклад значення)
cutoff_freq = 15  # Частота зрізу у Гц
order = 10  # Порядок фільтра

# Розрахунок нормалізованої частоти зрізу
nyq = 0.5 * sampling_rate
normal_cutoff = cutoff_freq / nyq

# Створення низькочастотного фільтра
b, a = butter(order, normal_cutoff, btype='highpass', analog=False)

# Застосування фільтра до сигналу
filtered_data = filtfilt(b, a, signal_intensity_new)

# Побудова графіку
plt.figure(figsize=(16, 8))
plt.plot(measure_number, signal_intensity_new, label='Original Signal')
plt.plot(measure_number, filtered_data, label='Filtered Signal', color='orange')
plt.xlabel('Measure Number')
plt.ylabel('Signal Intensity')
plt.legend()
plt.grid(True)
plt.show()