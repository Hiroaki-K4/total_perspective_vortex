# total_perspective_vortex
This project aims to create a brain computer interface based on electroencephalographic data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG reading, we will have to infer what he or she is thinking about or doing - (motion) A or B in a t0 to tn timeframe.

<img src='images/events.png' width='800'>

<br></br>

## Goals
- Process(parsing and filtering) EEG data
- Implement a dimensionality reduction algorithm
- Train and predict EEG data

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

<img src='images/sensor.png' width='800'>

<br></br>

## CSP(Common Spatial Pattern) algorithm
Common Spatial Pattern is a technique to analyze multi-channel data based on recordings from two classes (conditions). CSP yields a data-driven supervised decomposition of the signal parameterized by a matrix $W(C\times C)$ ($C$ being the number of channels) that projects the signal $x(t)$ in the original sensor space to $x_{CSP}(t)$, which lives in the surrogate sensor space, as follows:

$$
x_{CSP}(t)=W^\intercal x(t) \tag{1}
$$

### How to calculate



<img src='images/csp.png' width='800'>

<br></br>

## Training and prediction
You can train and predict EEG data(Task 4) by running below command. `--subject_num` is the number of subjects used training and prediction. `--show` is the flag whether program shows the result.

```bash
python3 srcs/train_and_predict.py --subjects_num 1 --show
```

<img src='images/result.png' width='800'>

<br></br>

## References
- [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [Optimizing Spatial Filters for Robust EEG Single-Trial Analysis](https://doc.ml.tu-berlin.de/bbci/publications/BlaTomLemKawMue08.pdf)
- [Blind Source Separation via Generalized Eigenvalue Decomposition](https://www.jmlr.org/papers/volume4/parra03a/parra03a.pdf)
