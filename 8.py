from matplotlib import pyplot as plt
from scipy.stats import alpha
from main import signal_intensity_new, measure_number
from scipy.signal import butter, filtfilt, find_peaks

filename="5_5"
fifth=__import__(filename)
real_signal_max=fifth.real_signal_max
real_signal_min=fifth.real_signal_min
inflections=fifth.inflections


filename="3_2"
third=__import__(filename)
a_from_3=third.a
measure_number=measure_number[:len(a_from_3)]


def number_of_maximums():
    nmaxfilter=len(real_signal_max)
    peaks,_=find_peaks(signal_intensity_new)
    nmax=len(peaks)

    print(f"Кількість максимумів без фільтрування: {nmax}")
    print(f"Кількість максимумів з фільтруванням: {nmaxfilter}")
    if nmax>nmaxfilter:
        print(f"Кількість максимумів без фільтру більша за кількість максимумів з фільтром на {nmax-nmaxfilter}")
    elif nmax==nmaxfilter:
        print(f"Кількість максимумів без фільтру та з однакова")
    else:
        print(f"Кількість максимумів з фільтром більша ніж кількість максимумів без фільтра на {nmaxfilter-nmax}")

number_of_maximums()
