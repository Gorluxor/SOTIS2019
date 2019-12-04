import pandas as pd
import numpy as np


def readcsv(filename):
    data = pd.read_csv(filename, usecols=[i for i in range(6)])
    return np.array(data)

def pretprocessing(data):
    for t in data:
        if t[6] == 0:
            np.delete()


first = readcsv("User 1_all_gaze.csv")
print("Done")


pretprocessing(first)

def to_region(data_array, region_filename):
    regions = pd.read_csv(region_filename)
    for data in data_array:


