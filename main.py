import numpy as np
from matplotlib import *
from matplotlib import pyplot as plt
from scipy import *

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

    print(measure_number)
    print(encoder_angle)
    print(leg_angle)
    print(signal_intensity)

# leg_angle=y, signal_intensity=x
def show():

    #plt.xlim(min(measure_number), max(measure_number))
    plt.ylim(0,5)
    """plt.plot(measure_number,signal_intensity)
    plt.plot(measure_number, leg_angle)
    plt.plot(measure_number, encoder_angle)"""
    plt.plot(leg_angle,signal_intensity)
    plt.plot(encoder_angle, signal_intensity)
    plt.grid(True)
    plt.show()

show()