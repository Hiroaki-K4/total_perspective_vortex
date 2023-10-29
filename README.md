# total_perspective_vortex
This project aims to create a brain computer interface based on electroencephalographic data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG reading, we will have to infer what he or she is thinking about or doing - (motion) A or B in a t0 to tn timeframe.

<br></br>

## Goals
- Process EEG datas (parsing and filtering)
- Implement a dimensionality reduction algorithm
- Use the pipeline object from scikit-learn
- Classify a data stream in "real time"

<br></br>

## Dataset
Subjects performed different motor/imagery tasks while 64-channel EEG were recorded using the [BCI2000 system](http://www.bci2000.org). Each subject performed 14 experimental runs: two one-minute baseline runs (one with eyes open, one with eyes closed), and three two-minute runs of each of the four following tasks:

```
1. A target appears on either the left or the right side of the screen. The subject opens and closes the corresponding fist until the target disappears. Then the subject relaxes.

2. A target appears on either the left or the right side of the screen. The subject imagines opening and closing the corresponding fist until the target disappears. Then the subject relaxes.

3. A target appears on either the top or the bottom of the screen. The subject opens and closes either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears. Then the subject relaxes.

4. A target appears on either the top or the bottom of the screen. The subject imagines opening and closing either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears. Then the subject relaxes.
```

You can know details and get the EEG dataset from [here](https://physionet.org/content/eegmmidb/1.0.0/).

Let's explore dataset by running below command.

```bash
python3 srcs/explore_dataset.py
```

This is the result of visualizing the raw data.

<img src='images/events.png' width='700'>

The position of the sensors are as follows.

<img src='images/sensor.png' width='600'>

```bash
python3 srcs/train.py --output_model_path model/pipeline.joblib --subjects_num 109 --show
```

<br></br>

## References
- [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [Optimizing Spatial Filters for Robust EEG Single-Trial Analysis](https://doc.ml.tu-berlin.de/bbci/publications/BlaTomLemKawMue08.pdf)
- [Blind Source Separation via Generalized Eigenvalue Decomposition](https://www.jmlr.org/papers/volume4/parra03a/parra03a.pdf)
