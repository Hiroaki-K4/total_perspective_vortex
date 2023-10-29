import numpy as np
from csp import CSP_ORI
from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline


def main():
    tmin, tmax = -1.0, 4.0
    event_id = dict(hands=2, feet=3)
    subject = 3  # 109 volunteers
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
        "Classification accuracy: %f / Chance level: %f"
        % (np.mean(scores), class_balance)
    )

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    csp.fit(epochs_data, labels)
    csp_cust.fit(epochs_data, labels)
    csp_data_log = csp_cust.transform(epochs_data, True)
    print("csp_data_log: ", csp_data_log)
    csp_data_mean = csp_cust.transform(epochs_data, False)
    print("csp_data_mean: ", csp_data_mean)
    csp_ori = csp.transform(epochs_data)
    print("csp_ori: ", csp_ori)


if __name__ == "__main__":
    main()
