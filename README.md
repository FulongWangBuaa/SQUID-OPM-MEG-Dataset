# SQUID-OPM-MEG-Dataset

# Analysis Code for the Paired SQUID-MEG / OPM-MEG Dataset

Analysis pipeline for a paired magnetoencephalography (MEG) dataset in which the same
participants were recorded with two systems — a conventional **SQUID-MEG** system and a
wearable **OPM-MEG** system — across six paradigms:

| # | Paradigm | Folder | Description |
|---|----------|--------|-------------|
| 1 | Resting state | `1-resting_state` | Eyes-open / eyes-closed spontaneous activity |
| 2 | Auditory | `2-auditory` | Pure-tone stimulation |
| 3 | Visual | `3-checkerboard` | Checkerboard stimulation |
| 4 | Somatosensory | `4-electrical` | Electrical stimulation |
| 5 | Motor | `5-finger_tapping` | Left / right finger tapping |
| 6 | Cognitive | `6-2back` | 2-back working-memory task |

Six participants (`S1`–`S6`) completed all paradigms on both systems.
The pipeline covers preprocessing, sensor-level analysis (PSD, SNR, TFR, evoked responses),
and source-level analysis (BEM/forward modelling, minimum-norm, DICS and pseudo-T beamformers,
morphing to a common template brain).

> **Note on data.** Raw and preprocessed MEG data are large (tens of GB) and are **not**
> intended to be tracked by Git. See [Data layout](#data-layout) and
> [GitHub size limits](#github-size-limits).

---

## Table of contents

- [Repository structure](#repository-structure)
- [Data layout](#data-layout)
- [Environment setup](#environment-setup)
- [How to run the code](#how-to-run-the-code)
- [Paradigm pipelines](#paradigm-pipelines)
- [Module reference (`my_code`)](#module-reference-my_code)
- [Output reference](#output-reference)
- [Notes and troubleshooting](#notes-and-troubleshooting)
- [Citation](#citation)

---

## Repository structure

```
Analysis/
├── 1-resting_state_preprocess.py      # preprocessing (one subject/system per run)
├── 1-resting_state_analysis.py        # PSD, TFR, source PSD
├── 2-main_audio_preprocess.py         # preprocessing
├── 2-main_audio_analysis.py           # SNR, evoked, dSPM source, source SNR
├── 3-main_checkerboard_preprocess.py  # preprocessing
├── 3-main_checkerboard_analysis.py    # SNR, evoked, dSPM source, source SNR
├── 4-main_electrical_preprocess.py    # preprocessing
├── 4-main_electrical_analysis.py      # SNR, evoked, dSPM source, source SNR
├── 5-main_finger_tapping_preprocess.py# preprocessing
├── 5-main_finger_tapping_analysis.py  # MRCP (DSS), beta ERD/ERS, beta SNR, DICS
├── 6-main_2back_preprocess.py         # preprocessing
├── 6-main_2back_analysis.py           # pseudo-T beamformer, source PSD, Hilbert envelope
│
├── my_code/                           # shared library (imported by the scripts)
│   ├── plot/
│   │   ├── wfl_plot_alignment.py      # 3-D sensor/helmet alignment figure
│   │   └── wfl_plot_cloudrain.py      # raincloud plots and significance markers
│   ├── tfr/
│   │   └── wfl_tfr_multitaper.py      # multitaper spectrogram for MNE objects
│   └── utils/
│       ├── auto_reject_trial.py       # variance-based bad-trial detection
│       ├── snr.py                     # bootstrap SNR confidence intervals
│       ├── wfl_preproc_dss1.py        # DSS (denoising source separation)
│       ├── utils.py                   # amplitude spectrum helper
│       └── compute_3d.py              # 3-D geometry helpers
│
├── MEGDataset/                        # dataset (not tracked; see below)
│   ├── 0-MRI/                         # FreeSurfer reconstructions, BEM, head models
│   ├── 1-Raw data/                    # raw FIF recordings
│   └── 2-Preprocessed data/           # ICA-cleaned continuous FIF recordings
│
├── result/                            # intermediate and final numerical results
├── fig/                               # exported figures (PNG / PDF / SVG)
└── supports/                          # Info objects used for plotting topographies
```

All scripts use **paths relative to this `Analysis/` folder**, so always start your
session from the repository root.

---

## Data layout

### `MEGDataset/0-MRI/`

FreeSurfer output for every participant (`S1`–`S6`): `mri/`, `surf/`, `label/`, `stats/`,
`bem/` (including `*-head.fif` head models) and `morph-maps/` (participant-to-participant
morphing maps). Source analysis scripts read subject anatomy and BEM surfaces from here.

### `MEGDataset/1-Raw data/`

One folder per participant, then one folder per paradigm, then one folder per system:

```
MEGDataset/1-Raw data/
└── S1/
    ├── 1-resting_state/
    │   ├── OPM-MEG/   S1_resting_state_opm-raw.fif
    │   └── SQUID-MEG/ S1_resting_state_squid-raw.fif
    ├── 2-auditory/
    ├── 3-checkerboard/
    ├── 4-electrical/
    ├── 5-finger_tapping/
    │   ├── OPM-MEG/   S1_finger_tapping_opm-raw.fif
    │   │              S1_finger_tapping_opm.json     # trial-wise behavioural log
    │   └── SQUID-MEG/ S1_finger_tapping_squid-raw.fif
    │                  S1_finger_tapping_squid.json
    └── 6-2back/
```

Naming convention:

```
S<participant>_<paradigm>_<opm|squid>-raw.fif
```

The optional JSON file belonging to the finger-tapping paradigm stores the
trial-by-trial behavioural responses (`responses_left` / `responses_right`) that are used
to label left- and right-hand trials.

### `MEGDataset/2-Preprocessed data/`

Same folder structure as the raw data; files are continuous, ICA-cleaned recordings named:

```
S<participant>_<paradigm>_<opm|squid>_preprocessed-raw.fif
```

Preprocessing is done by the `*_preprocess.py` scripts; the `*_analysis.py` scripts never
write to `MEGDataset/`, only to `result/` and `fig/`.

---

## Environment setup

The pipeline was developed with Python 3.9 / 3.10 and MNE-Python. A typical environment:

```bash
conda create -n meg python=3.9
conda activate meg
pip install numpy scipy matplotlib pandas seaborn scikit-learn \
            mne nilearn colorcet joblib tqdm
```

Additional requirements:

- **FreeSurfer** — only if you want to re-create the surfaces/BEM files in `MEGDataset/0-MRI`.
  The provided reconstructions can be used directly, in which case FreeSurfer is not needed
  for the analysis scripts.
- **`my_code`** is imported as a local package. Run the scripts from the `Analysis/` folder
  (or add it to `PYTHONPATH`) so that `from my_code.utils...` resolves correctly.

---

## How to run the code

The scripts are **cell-based Python files** (`# %%` markers) that were developed in the
VS Code interactive window (Jupyter-compatible). They use IPython magics such as
`%matplotlib auto` and `%autoreload 2`, so run them cell by cell rather than with
`python script.py`:

1. Open the repository root (`Analysis/`) in VS Code.
2. Open the script and send each `# %%` cell to the interactive window in order.
3. Start with the `*_preprocess.py` script of a paradigm, then run the matching
   `*_analysis.py` script.

Typical control flow inside a preprocess script:

```python
sub_ids  = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
paradigm = 'auditory'

sub_id     = 'S2'
experiment = 'SQUID'          # 'OPM' or 'SQUID'

data_path = (f'./MEGDataset/1-Raw data/{sub_id}/2-{paradigm}/'
             f'{experiment}-MEG/{sub_id}_{paradigm}_{experiment.lower()}-raw.fif')
raw = mne.io.read_raw_fif(data_path, preload=True)

raw_filt = raw.copy().filter(l_freq=2, h_freq=40)
ica = mne.preprocessing.ICA(n_components=30, method='picard',
                            max_iter='auto', random_state=97)
ica.fit(raw_filt)
ica.plot_components()          # inspect and reject artefact components
raw_ica = ica.apply(raw_filt.copy())

save_path = data_path.replace('1-Raw data', '2-Preprocessed data') \
                     .replace('-raw.fif', '_preprocessed-raw.fif')
raw_ica.save(save_path, overwrite=False)
```

The analysis scripts loop over all participants and both systems internally, so one run of
the script produces the group-level results and figures.

### Caching and overwriting

Most steps are cached: results are written to `result/…` and re-used on the next run.
Several cells explicitly raise `FileExistsError` when the output file already exists to
prevent silent overwriting, for example:

```python
if not os.path.exists(f'./result/2-{paradigm}/SNRs.pkl'):
    pickle.dump(SNRs, open(f'./result/2-{paradigm}/SNRs.pkl', 'wb'))
else:
    raise FileExistsError(f'./result/2-{paradigm}/SNRs.pkl')
```

When you want to recompute a result, delete the corresponding file in `result/` (or comment
out the saving cell). The following cells usually load the cached file and continue.

---

## Paradigm pipelines

| Paradigm | Preprocess | Analysis | Key analyses | Main figures |
|----------|-----------|----------|--------------|--------------|
| Resting state | `1-resting_state_preprocess.py` | `1-resting_state_analysis.py` | Welch PSD for eyes-open/closed; multitaper TFR normalised to the eyes-open baseline; band-wise source PSD (theta 4–8, alpha 8–12, beta 13–30, gamma 30–40 Hz) | `psd_4conditions`, `tfr_relative_change`, `source` |
| Auditory | `2-main_audio_preprocess.py` | `2-main_audio_analysis.py` | Epoching and bad-trial rejection; bootstrap SNR (`SNRLB`); grand-average evoked; dSPM source estimates; source-space SNR | `SNR_dist`, `evoked`, `source`, `source_snr`, `source_amplitude` |
| Checkerboard | `3-main_checkerboard_preprocess.py` | `3-main_checkerboard_analysis.py` | Same pipeline as auditory | `SNR_dist`, `evoked`, `source`, `source_snr`, `source_amplitude` |
| Electrical | `4-main_electrical_preprocess.py` | `4-main_electrical_analysis.py` | Same pipeline as auditory | `SNR_dist`, `evoked`, `source`, `source_snr`, `source_amplitude` |
| Finger tapping | `5-main_finger_tapping_preprocess.py` | `5-main_finger_tapping_analysis.py` | Left/right MRCP with DSS (`nt_dss1`); beta-band ERD/ERS time-frequency maps; beta SNR (MRBD vs PMBR windows); DICS beamformer source maps morphed to the template subject | `evoked`, `Beta_power_change`, `stc` |
| 2-back | `6-main_2back_preprocess.py` | `6-main_2back_analysis.py` | Long (0–60 s) epochs; pseudo-T beamformer contrast (task 0–40 s vs baseline 40–60 s); morphing to a common volume source space (template subject, 5 mm grid); source time series, Hilbert envelope and source PSD | `stc_pseudoT`, `hilbert_envelope`, `source_psd` |

Common conventions across paradigms:

- Two systems: `experiment` is `'OPM'` or `'SQUID'`; triggers come from the `Trigger`
  channel for OPM and from `UPPT001` for SQUID.
- Evoked paradigms use baselines of a few hundred ms before stimulus onset and reject bad
  trials with `get_bad_trials(..., thresh_val=3)`.
- Source modelling uses the `oct6` surface source space, an `ico=4` single-layer BEM and an
  identity `head`→`mri` transform (`mne.transforms.Transform('head', 'mri')`), because
  sensor positions were digitised in the MRI coordinate frame.
- Group-level source maps are morphed to a common template (subject `S4`, the only participant
  with a full volume source space) before averaging.

---


## Output reference

### `result/`

```
result/
├── 1-resting_state/
│   ├── psd/<OPM|SQUID>/<S#>/{eyes_open,eyes_closed}_psd.npz   # power, amplitude, freqs, channels
│   ├── tfr_group.pkl                                          # group TFR per system/condition
│   ├── stcs_all.pkl                                           # band-wise source power
│   └── source/<OPM|SQUID>/<S#>_{oct6_surface-src,ico4-bem,oct6_surface-fwd}.fif
├── 2-auditory/  ├── 3-checkerboard/  └── 4-electrical/
│   ├── SNRs.pkl              # bootstrap SNR distributions (SNRLB)
│   ├── evokeds_mean.pkl      # grand-average evoked responses
│   ├── stcs_all.pkl          # dSPM source estimates
│   ├── stcs_snr_all.pkl      # source-space SNR
│   └── source/…              # src / bem / fwd per participant and system
├── 5-finger_tapping/
│   ├── evokeds_mean.pkl      # left/right MRCPs
│   ├── tfr_sub/*.pkl         # per-participant beta ERD/ERS maps
│   ├── tfr_all.pkl           # group beta maps
│   ├── stcs_all.pkl          # DICS source estimates
│   └── source/…
└── 6-2back/
    ├── epochs/*-epo.fif      # 0–60 s epochs
    ├── stc/*_pseudoT-stc.pkl # pseudo-T source maps
    ├── stc/*_ws.pkl          # beamformer weights
    ├── stcs_fsaverage.pkl    # morphed group source maps
    ├── source_psd.pkl        # source-level spectra
    ├── volume_source_space-src.fif
    └── source/…
```



## Citation

If you use this code or the accompanying dataset, please cite the dataset descriptor
(reference to be added after publication):

```bibtex
@article{wang2026paired,
  title   = {A paired SQUID-MEG and OPM-MEG dataset across resting-state, auditory,
             motor, and cognitive paradigms},
  author  = {Wang, Fulong and Kong, Xiangrui and Yue, Zhibang and Yu, Chenxuan and
             Jiang, Miaowen and Wang, Dawei and Cao, Fuzhi},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

Please also cite the software this pipeline builds on, in particular
[MNE-Python](https://mne.tools/) and, where used, FreeSurfer and Nilearn.

## License

Add the license of your choice (for example MIT for the code) before publishing the
repository. The MEG data are released separately together with the dataset descriptor.
