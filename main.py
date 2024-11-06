import numpy as np
from matplotlib import *
from matplotlib import pyplot as plt
from scipy import *


signal_intensity_new=[]
with open("Variant_12.txt", "r") as f:
    all_data=f.readlines()
    measure_number = []
    encoder_angle = []
    leg_angle = []
    signal_intensity = []

    for line in all_data:
        values=line.split()
        measure_number.append(float(values[0].replace(",", ".")))
        encoder_angle.append(float(values[1].replace(",", ".")))
        leg_angle.append(float(values[2].replace(",", ".")))
        signal_intensity.append(float(values[3].replace(",", ".")))

    """print(measure_number)
    print(encoder_angle)
    print(leg_angle)
    print(signal_intensity)"""

def lowerFunction():
    global signal_intensity_new
    shiftY = np.mean(signal_intensity)  # Розрахунок середнього значення інтенсивності
    signal_intensity_new = [x - shiftY for x in signal_intensity]  # Зсув кожного значення
    return signal_intensity_new

lowerFunction()

def show():

    plt.figure(figsize=(12, 6))
    #plt.ylim(-2, 10)  # МОЖНА ДЛЯ КРАЩОГО(стиснутого графіка) ВИГЛЯДУ ФІЛЬТРІВ
    plt.plot(measure_number, signal_intensity_new)
    plt.grid(True)
    plt.title("Початковий сигнал")
    plt.xlabel('Виміри')
    plt.ylabel('Інтенсивність сигналу')
    plt.show()


measure_number=np.array(measure_number)
signal_intensity_new=np.array(signal_intensity_new)


show()
