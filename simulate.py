"""
Simulate electrode recordings
"""

import numpy as np

def getspiketimes(r, T):
    """
    generate spike times for a single neuron

    rate: 0 <= r <= 1, number of time points: T
    """
    return np.random.binomial(1,r,T)

def getwaveform(dim):
    """
    generate waveform for a single neuron
    """

    # waveform time sampling
    t = np.linspace(0,1,dim)

    # draw time constant and frequency
    omega = 1 + 0.25*np.random.randn()
    tau = 0.25 + 0.1*np.random.randn()
    amp = np.random.randn()*1.5

    return amp * np.exp(-t/tau) * np.sin(2*np.pi*omega*t)

def record(T, N=1, dim=50, rate=0.005, sigma=0.1):
    """
    Simulates a neural recording

    returns:
        neurons, a list of waveforms
        spiketimes, a list of spike times
        v, the 'true' voltage a [T x 1] array (time points by # electrodes)
        data, the 'noisy' voltage a [T x 1] array (time points by # electrodes)
    """

    waveforms = list()
    spiketimes = list()
    v = np.zeros(T)

    # for each neuron
    for j in range(N):

        # generate waveform and spike times for this neuron
        w = getwaveform(dim)
        spk = getspiketimes(rate, T)

        # add to list
        waveforms.append(w)
        spiketimes.append(spk)

        # convolve
        v += np.convolve(spk, w, 'same')

    # add some noise yo
    data = v + sigma * np.random.randn(T)

    return waveforms, spiketimes, v, data

if __name__ == "__main__":

    # demo
    waveforms, spiketimes, v, data = record(10000, N=3)

    pass
