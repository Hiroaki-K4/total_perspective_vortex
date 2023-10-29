import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from csp import CSP_ORI


def main(output_model_path: str, subjects_num: int):
    # #############################################################################
    # # Set parameters and read data

    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.
    if subjects_num > 109 or subjects_num < 1:
        raise ValueError("The number of subjects is wrong.")

    tmin, tmax = -1.0, 4.0
    event_id = dict(hands=2, feet=3)
    subject = 3  # 109 volunteers
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
    eegbci.standardize(raw)  # set channel names
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
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    csp_cust = CSP_ORI(n_components=4)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

    # Printing the results
    # class_balance = np.mean(labels == labels[0])
    class_balance = np.mean(labels)
    class_balance = max(class_balance, 1.0 - class_balance)

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
    csp.plot_filters(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

    sfreq = raw.info["sfreq"]
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        # csp_cust.fit(epochs_data_train[train_idx], y_train)
        # X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        # Test data shape (Split per label, channel, data point)
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
            pick_filters = csp.filters_[:4]
            X = np.asarray(
                [
                    np.dot(pick_filters, epoch)
                    for epoch in epochs_data[test_idx][:, :, n : (n + w_length)]
                ]
            )
            X = (X**2).mean(axis=2)
            # X_test shape (Split per label, extracted components)
            score_this_window.append(lda.score(X_test, y_test))
            X_r = lda.predict(X_test)
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    print()
    print(
        "Classification accuracy: %f / Chance level: %f"
        % (np.mean(scores), class_balance)
    )
    print("Sampling frequency: ", sfreq)
    print("Window length: ", w_length)
    print("Window step size: ", w_step)
    print("Window start position: ", w_start)

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")

    dump(clf, output_model_path)
    load_clf = load(output_model_path)
    load_scores = cross_val_score(
        load_clf, epochs_data_train, labels, cv=cv, n_jobs=None
    )
    print(
        "Classification accuracy by loaded model: %f / Chance level: %f"
        % (np.mean(load_scores), class_balance)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_path", type=str)
    parser.add_argument("--subjects_num", type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.output_model_path, args.subjects_num)
    if args.show:
        plt.show()
    else:
        print("It shows nothing")
