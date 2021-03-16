import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import csv

file_path = '4N_E-7.499033653778218_g10000_p60.csv'
# Reading csv
with open(file_path, newline='') as f:
    reader = csv.reader(f)
    _positions = list(reader)
    _positions.pop(0)

# Solution length
M = len(positions)

# Cleaning
positions = []

for i in range(len(_positions)):
    positions.append(float(_positions[i][0]))

# Creating dataset
X = []
for x in range(0,M,3):
    X.append(positions[x])

Y = []
for y in range(1,M,3):
    Y.append(positions[y])

Z = []
for z in range(2,M,3):
    Z.append(positions[z])

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.gca(projection ="3d")
# Creating plot
ax.scatter3D(X,Y,Z, c='g',s=80)
ax.plot(X,Y,Z, c='r')
ax.set_xlabel(r'x [$10^{-14}$ m]')
ax.set_ylabel(r'y [$10^{-14}$ m]')
ax.set_zlabel(r'z [$10^{-14}$ m]')
plt.title("Optimum molecule with N=4")
plt.show()
