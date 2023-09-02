import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt

from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter

import mne


def main():
    sfreq = 1000.0
    f_p = 40.0
    # limits for plotting
    flim = (1.0, sfreq / 2.0)  # frequencies
    dlim = (-0.2, 0.2)  # delays

    nyq = sfreq / 2.0  # the Nyquist frequency is half our sample rate
    freq = [0, f_p, f_p, nyq]
    gain = [1, 1, 0, 0]

    third_height = np.array(plt.rcParams["figure.figsize"]) * [1, 1.0 / 3.0]
    ax = plt.subplots(1, figsize=third_height)[1]
    plot_ideal_filter(freq, gain, ax, title="Ideal %s Hz lowpass" % f_p, flim=flim)
    plt.show()


if __name__ == '__main__':
    main()
