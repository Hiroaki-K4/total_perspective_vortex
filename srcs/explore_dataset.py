import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from csp import CSP_ORI


def main():
    # #############################################################################
    # # Set parameters and read data

    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.
    tmin, tmax = -1.0, 4.0
    event_id = dict(hands=2, feet=3)
    subject = 3 # 109 volunteers
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    raw.plot(scalings="auto", show=False)
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # Apply band-pass filter
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

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
    print(
        "Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance)
    )

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
        csp_cust.fit(epochs_data_train[train_idx], y_train)
        input()
        # X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        # Test data shape (Split per label, channel, data point)
        for n in w_start:
            print("Input test shape: ", epochs_data[test_idx][:, :, n : (n + w_length)].shape)
            X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
            print("pick filters: ", csp.filters_[: 4].shape)
            pick_filters = csp.filters_[: 4]
            X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data[test_idx][:, :, n : (n + w_length)]])
            print("X_shape: ", X.shape)
            X = (X**2).mean(axis=2)
            print("Mean: ", X.shape)
            # X_test shape (Split per label, extracted components)
            print("X_test shape: ", X_test.shape)
            print(X_test)
            input()
            score_this_window.append(lda.score(X_test, y_test))
            X_r = lda.predict(X_test)
            print("X_r: ", X_r)
            print("y_test: ", y_test)
            print("score: ", lda.score(X_test, y_test))
            input()
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin
    # print("w_times: ", w_times)
    # input()
    
    print()
    print("Sampling frequency: ", sfreq)
    print("Window length: ", w_length)
    print("Window step size: ", w_step)
    print("Window start position: ", w_start)

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")


if __name__ == '__main__':
    main()
    plt.show()
