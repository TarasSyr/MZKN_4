from matplotlib import pyplot as plt
from numpy.ma.extras import average

from main import leg_angle, measure_number, signal_intensity_new

a=0
dublicatesList=[]
for i in range(len(leg_angle)):

    if leg_angle[i-1]==leg_angle[i]:
        a+=1
    elif leg_angle[i-1]!=leg_angle[i]:
        dublicatesList.append(a)
        a=0

for i in dublicatesList:
    medium=average(signal_intensity_new[i],signal_intensity_new[i+1])

print(medium)
plt.grid(True)
plt.plot(measure_number,dublicatesList)
plt.show()
print(dublicatesList)
