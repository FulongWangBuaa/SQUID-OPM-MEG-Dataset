'''
处理纯音听觉数据
'''
# %%
%reload_ext autoreload
%autoreload 2
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.linalg
%matplotlib auto
import pickle
import pandas as pd

from mne.io.pick import _picks_to_idx
from my_code.plot.wfl_plot_alignment import wfl_plot_alignment


# %%
sub_ids = ['S1','S2','S3','S4','S5','S6']
paradigm = 'checkerboard'

# %%
sub_id = 'S6'
experiment = 'OPM' # OPM or SQUID

data_path = f'./MEGDataset/1-Raw data/{sub_id}/3-{paradigm}/{experiment}-MEG/{sub_id}_{paradigm}_{experiment.lower()}-raw.fif'

raw = mne.io.read_raw_fif(data_path, preload=True)


# %%
# 查看数据PSD
psd = raw.compute_psd(fmin=0,fmax=100,n_fft=1024*8)
psd.plot()
plt.show()


# %%
# 查看配准情况
subjects_dir = './MEGDataset/0-MRI'
trans = mne.transforms.Transform('head', 'mri')
wfl_plot_alignment(info=raw.info,
                   trans=trans,
                   subject=sub_id,
                   subjects_dir=subjects_dir,
                   surfaces=['head-dense'],
                   coord_frame='mri',
                   meg=[],
                   ch_pos=True,pos_color='r',pos_scale=0.005,
                   ch_ori=True,ori_color='b',ori_scale=0.01,
                   )


# %%
raw_filt = raw.copy().filter(l_freq=2,h_freq=40)

# %%
ica = mne.preprocessing.ICA(n_components=30, method="picard", max_iter="auto", random_state=97)
ica.fit(raw_filt)

# %%
ica.plot_components()
plt.show()

ica.plot_sources(raw_filt)


# %%
raw_ica = raw_filt.copy()
raw_ica = ica.apply(raw_ica)



# %%
psd = raw_ica.compute_psd(fmin=0,fmax=100,n_fft=1024*8)
psd.plot()
plt.show()



# %%
# 保存ICA后的数据
save_path = data_path.replace(f'1-Raw data', f'2-Preprocessed data')
save_path = save_path.replace('-raw.fif', '_preprocessed-raw.fif')

raw_ica.save(save_path, overwrite=False)


# %%
save_path = data_path.replace(f'1-Raw data', f'2-Preprocessed data')
save_path = save_path.replace('-raw.fif', '_preprocessed-raw.fif')
raw_ica = mne.io.read_raw_fif(save_path, preload=True)

# %%
if experiment == 'SQUID':
    events = mne.find_events(raw_ica, stim_channel='UPPT001',min_duration = 10/raw.info['sfreq'])
elif experiment == 'OPM':
    events = mne.find_events(raw_ica, stim_channel='Trigger')

epochs = mne.Epochs(raw_ica, events, tmin=-0.2, tmax=0.8, preload=True, baseline=(-0.2, 0))

from my_code.utils.auto_reject_trial import get_bad_trials
bad_trials = get_bad_trials(epochs, thresh_val=3)
epochs = epochs.drop(bad_trials)


# %%
evoked = epochs.average()
evoked.plot(gfp=True)
plt.show()


# %%
