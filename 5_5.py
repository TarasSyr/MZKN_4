from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
from main import signal_intensity_new, measure_number

filename="3_2"
third=__import__(filename)
a=third.a
measure_number=measure_number[:len(a)]


def checkPeaks():
    """all_peaks=find_peaks(a)
    all_lows=find_peaks(-a)
    segment=10               #діапазон
    diffFromMean=0.05       #наскільки пік(максимум) відрізняється від середньої амплітуди
    real_signal_max=[]         #ініціалізую список з максимумами сигналів

    for i in range(len(a)):
        if a[i]>0:
            for i in range(len(all_peaks)):
                part_of_peaks = all_peaks[i:i + segment]
                signal_amplitude=np.mean(part_of_peaks)/segment        #середня амплітуда

                for i in range(i+segment):
                    if all_peaks[i]<(signal_amplitude+diffFromMean):
                        real_signal_max.append(all_peaks[i])
                    else:
                        continue
        """
    segment = 10  # кількість точок у сегменті
    diffFromMean = 0.025  # значення, що додається до середньої амплітуди
    real_signal_max = []  # список для збереження піків, що відповідають критерію
    real_signal_min=[]
    inflections=[]

    # Знаходимо всі максимуми
    all_peaks, _ = find_peaks(a)
    all_lows, _ = find_peaks(-a)


    # Перебираємо піки з кроком, рівним довжині сегмента
    for i in range(0, len(all_peaks), segment):             #МАКСИМУМИ
        # Беремо частину піків у поточному сегменті
        part_of_peaks = all_peaks[i:i + segment]

        # Перевірка, що сегмент не порожній
        if len(part_of_peaks) == 0:
            continue

        # Розраховуємо середню амплітуду для піків у сегменті
        signal_amplitude = np.mean(a[part_of_peaks])

        # Зберігаємо тільки ті піки, що перевищують поріг (середнє + diffFromMean)
        for peak in part_of_peaks:
            if a[peak] >= signal_amplitude + diffFromMean:
                real_signal_max.append(peak)
            else:
                inflections.append(peak)

    for i in range(0, len(all_lows), segment):             #МІНІМУМИ
        # Беремо частину піків у поточному сегменті
        part_of_lows = all_lows[i:i + segment]

        # Перевірка, що сегмент не порожній
        if len(part_of_lows) == 0:
            continue

        # Розраховуємо середню амплітуду для піків у сегменті
        signal_amplitude = np.mean(a[part_of_lows])

        # Зберігаємо тільки ті піки, що перевищують поріг (середнє + diffFromMean)
        for low in part_of_lows:
            if a[low] <= signal_amplitude + diffFromMean:
                real_signal_min.append(low)
            else:
                inflections.append(low)


    # Візуалізація результату
    import matplotlib.pyplot as plt
    plt.plot(a, label='Сигнал')
    plt.plot(real_signal_max, a[real_signal_max], 'ro', label='Максимуми')
    plt.plot(real_signal_min, a[real_signal_min], "go", label="Мінімуми")
    #plt.plot(inflections, a[inflections], "yo", label="Перегини")
    plt.show()

    with open("min_max.txt","w", encoding="utf-8") as f:
        f.write(f"Точки максимумів: {real_signal_max}\n")
        f.write(f"Точки мінімумів: {real_signal_min}")
        f.close()

    return real_signal_max,real_signal_min,inflections


#checkPeaks()
real_signal_max,real_signal_min,inflections=checkPeaks()