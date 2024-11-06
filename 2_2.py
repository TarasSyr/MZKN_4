from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from main import signal_intensity_new, measure_number

# Припустимо значення частоти дискретизації
#sampling_rate = 10  # Приклад: 10 Гц (можна коригувати залежно від інтервалу між вимірами)

# Функція для створення фільтра і фільтрування сигналу
def apply_filter(signal_intensity_new, cutoff, filter_type, order=9):       #def apply_filter(signal_intensity_new, cutoff, fs, filter_type, order=9):
    #nyquist = 0.5 * fs  # Частота Найквіста
    #normal_cutoff = cutoff / nyquist  # Нормалізована частота зрізу

    b, a = butter(order, cutoff, btype=filter_type, analog=False)
    filtered_data = filtfilt(b, a, signal_intensity_new)
    return filtered_data

# Налаштування частот зрізу
low_cutoff_freq = 0.1 # Частота зрізу для низькочастотного фільтра(від 0.03 до 0.1)
high_cutoff_freq = 0.6  # Частота зрізу для високочастотного фільтра(від 0.2 до 0.6)

# Застосування низькочастотного фільтра
low_filtered_data = apply_filter(signal_intensity_new, low_cutoff_freq, 'low')
#low_filtered_data = apply_filter(signal_intensity_new, low_cutoff_freq, sampling_rate, 'low')
# Застосування високочастотного фільтра
high_filtered_data = apply_filter(signal_intensity_new, high_cutoff_freq, 'high')
#high_filtered_data = apply_filter(signal_intensity_new, high_cutoff_freq, sampling_rate, 'high')

# Візуалізація початкового та відфільтрованих сигналів
plt.figure(figsize=(12, 6))
plt.ylim(-2,10) #МОЖНА ДЛЯ КРАЩОГО(стиснутого графіка) ВИГЛЯДУ ФІЛЬТРІВ
plt.plot(measure_number, signal_intensity_new, label='Початковий сигнал')
plt.plot(measure_number, high_filtered_data, label='Високо-частотний фільтр', color='orange')
plt.plot(measure_number, low_filtered_data, label='Низько-частотний фільтр', color='red')
plt.xlabel('Виміри')
plt.ylabel('Інтенсивність сигналу')
plt.title('Початковий сигнал та фільтри')
plt.legend()
plt.grid(True)
plt.show()
