# eeg-motor-imagery

EEG motor imagery analysis: left vs right hand classification and BCI aptitude prediction, using the PhysioNet Motor Movement/Imagery Dataset.

## About

This project explores two questions using EEG data from 109 subjects performing motor imagery tasks:

1. **Can we classify left vs right hand motor imagery from EEG features?** 
Using time series features extracted from motor cortex electrodes (C3, C4 and neighbors), we train classifiers to distinguish which hand a subject was imagining moving.

2. **Can we predict how well a subject will perform at BCI?** 
Not all subjects produce separable EEG patterns during motor imagery: a phenomenon known as BCI illiteracy (Blankertz et al., 2010). We use aggregated EEG features per subject to predict their classification accuracy via regression.

## Dataset

[EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) from PhysioNet. 109 subjects, 64 EEG channels, motor execution and imagery tasks.

Raw data is not included in the repo due to size (~several GB). To download:

### Linux / macOS (recommended)

```bash
wget -r -N -c -np -nH --cut-dirs=3 --reject "index.html*" https://physionet.org/files/eegmmidb/1.0.0/
```

### Windows
```powershell
# requires Amazon.AWSCLI
winget install Amazon.AWSCLI
```
```powershell
# from project root
aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ data/raw/
```

Place the downloaded files in `data/raw/`.

## Project structure

```
eeg-motor-imagery/
├── data/
│   ├── raw/                     # EDF files from PhysioNet (not tracked)
│   └── processed/
│       ├── trial_features.csv   # per-trial features for classification
│       └── subject_features.csv # per-subject aggregated features for regression
├── notebooks/
│   ├── 01_exploration.Rmd
│   └── 02_preprocessing.Rmd
├── src/
│   └── preprocess.R
├── report/
│   └── final_report.Rmd
├── references.bib
├── .gitignore
└── README.md
```

## Tasks

| Task | Input | Target | Models |
|---|---|---|---|
| Classification | Per-trial EEG features (6 channels x ~15 tsfeatures) | left / right (binary) | Logistic regression, Random forest |
| Regression | Per-subject aggregated features (mean, std across trials) | Classification accuracy (0.5-1.0) | Linear regression, Random forest |

## Channel selection

We focus on electrodes over the motor cortex: C3, C4, FC3, FC4, CP3, CP4. Motor imagery of the right hand produces desynchronization over C3 (left hemisphere) and vice versa (Pfurtscheller & Neuper, 1997), making these channels the most informative for left vs right discrimination.

## References

- Pfurtscheller, G. & Neuper, C. (1997). Motor imagery activates primary sensorimotor area in humans. *Neuroscience Letters*, 239, 65-68.
- Pfurtscheller, G., Brunner, C., Schlogl, A., & Lopes da Silva, F.H. (2006). Mu rhythm (de)synchronization and EEG single-trial classification of different motor imagery tasks. *NeuroImage*, 31, 153-159.
- Blankertz, B. et al. (2010). Neurophysiological predictor of SMR-based BCI performance. *NeuroImage*, 51(4), 1303-1309.
- Ahn, M. et al. (2013). High Theta and Low Alpha Powers May Be Indicative of BCI-Illiteracy in Motor Imagery. *PLOS ONE*.