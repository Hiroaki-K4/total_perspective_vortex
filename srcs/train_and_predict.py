import argparse

import matplotlib.pyplot as plt
import numpy as np
from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit

from csp import CSP_ORI


def main(subjects_num: int):
    if subjects_num < 1 or subjects_num > 109:
        raise ValueError("The number of subjects is wrong.")

    tmin, tmax = -1.0, 4.0
    event_id = dict(hands=2, feet=3)
    runs = [6, 10, 14]  # motor imagery: hands vs feet
    excluded_subs = [88, 89, 92, 100]

    all_raw_files = []
    for subject in range(1, subjects_num + 1):
        if subject in excluded_subs:
            continue
        raw_fnames = eegbci.load_data(subject, runs)
        for raw_fname in raw_fnames:
            all_raw_files.append(raw_fname)

    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in all_raw_files])
    raw.plot(scalings="auto", show=False)
    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # Apply band-pass filter
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(
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
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    labels = epochs.events[:, -1] - 2

    # Define a monte-carlo cross-validation generator (reduce variance):
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    csp.fit_transform(epochs_data, labels)
    csp.plot_filters(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

    sfreq = raw.info["sfreq"]
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    original_csp = CSP_ORI(n_components=4)
    scores_windows = []
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = original_csp.fit_transform(epochs_data_train[train_idx], y_train)
        lda.fit(X_train, y_train)
        score_window = []
        for n in w_start:
            X_test = original_csp.transform(
                epochs_data[test_idx][:, :, n : (n + w_length)]
            )
            score_window.append(lda.score(X_test, y_test))

        scores_windows.append(score_window)

    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.xlabel("time [s]")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects_num", type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.subjects_num)
    if args.show:
        plt.show()
    else:
        print("It shows nothing")
