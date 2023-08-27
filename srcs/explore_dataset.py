import argparse

import mne
from matplotlib import pyplot as plt
from mne.datasets import eegbci

from preprocess import standardize


def main(eeg_data_path: str):
    raw = mne.io.read_raw_edf(eeg_data_path)
    raw.plot(scalings="auto", show=False)

    standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.plot_sensors(show_names=True, sphere=(0, 0.015, 0, 0.095))

    events_from_annot, event_dict = mne.events_from_annotations(raw)
    print("Event type: ", event_dict)
    print("Event time and type: ", events_from_annot)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_data_path")
    args = parser.parse_args()
    main(args.eeg_data_path)
