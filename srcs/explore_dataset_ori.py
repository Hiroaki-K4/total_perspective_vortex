import argparse

import mne
from matplotlib import pyplot as plt
from mne.datasets import eegbci
from preprocess import standardize


def main(eeg_data_path: str):
    raw = mne.io.read_raw_edf(eeg_data_path, preload=True)
    # raw.plot(scalings="auto", show=False)
    raw.plot(scalings="auto", show=False)

    standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.filter(7.0, 30.0)
    raw.plot(scalings="auto", show=False)

    # raw.plot_sensors(show_names=True, sphere=(0, 0.015, 0, 0.095))

    events, event_dict = mne.events_from_annotations(raw)
    print("Event type: ", event_dict)
    print("Event time and type: ", events)

    picks = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    event_id = dict(hands=2, feet=3)
    tmin, tmax = -1.0, 4.0
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    print(epochs)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    print(epochs_train)
    labels = epochs.events[:, -1] - 2
    print(epochs.events)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_data_path")
    args = parser.parse_args()
    main(args.eeg_data_path)
