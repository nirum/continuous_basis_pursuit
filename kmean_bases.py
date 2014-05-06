import numpy as np
from scipy.signal import find_peaks_cwt
from sklearn.cluster import KMeans
from peakdetect import peakdet

def peak_windows(data, width=80):
    highs, lows = peakdet(data, 0.75)
    peak_centers = np.array(np.r_[highs[:,0], lows[:,0]], dtype=np.uint32)

    windows = np.zeros((peak_centers.shape[0], width))
    for ind, center in enumerate(peak_centers):
        windows[ind, :] = data[center-width/2:center+width/2]
    return windows

def initialization(data, units):
    km = KMeans(n_clusters=units, n_jobs=-2)
    km.fit(peak_windows(data))
    return km.cluster_centers_.T

if __name__ == "__main__":
    from simulate import *
    import matplotlib.pyplot as plt
    n_true_units = 3
    waveforms, spiketimes, v, data = record(1e4, N=n_true_units)
    plt.plot(initialization(data, n_true_units))
    plt.show()
