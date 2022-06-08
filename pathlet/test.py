import numpy as np
import pickle
import matplotlib.pyplot as plt

critical_points = pickle.load(open('./critical_points', 'rb'))
for i in critical_points:
    plt.plot(i[0], i[1], 'xb')
plt.pause(0.001)
plt.show()
