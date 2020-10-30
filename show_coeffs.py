import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

plt.figure(figsize=(20,10), dpi=100)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

coeffs = []

with open('output/coeffs_y.txt') as f:
    lines = f.readlines()

    for line in lines:
        line_split = line.strip("\n").split(',')
        line_split = [float(split) for split in line_split]
        coeffs += line_split
coeffs = np.asarray(coeffs, dtype=float)
counter = Counter(coeffs)
print(counter.keys())
print(counter.values())
keys = list(counter.keys())
keys.sort()
print(keys)
for coeff in keys:
    print("%f:%d" % (coeff, counter[coeff]))
# plt.hist(coeffs, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.