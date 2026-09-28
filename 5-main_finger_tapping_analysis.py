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

# %%
sub_ids = ['S1','S2','S3','S4','S5','S6']
paradigm = 'finger_tapping'

# %%
import json
def extract_hand_list(json_path):
    """
    Extract hand response per trial from a finger tapping JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    hand_list : list of int
        1 = left hand
        2 = right hand
        0 = no response or ambiguous response
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hand_list = []

    for i, trial in enumerate(data["trials"]):
        left = len(trial.get("responses_left", []))
        right = len(trial.get("responses_right", []))

        if left > 0 and right == 0:
            hand_list.append(1)
        elif right > 0 and left == 0:
            hand_list.append(2)
        else:
            # no response or both hands responded
            hand_list.append(255)

    return hand_list


raws_all = dict()
epochs_all = dict()
for experiment in ['OPM', 'SQUID']:
    raws_experiment = []
    epochs_experiment = []
    for sub_id in sub_ids:
        data_path = f'./MEGDataset/2-Preprocessed data/{sub_id}/5-{paradigm}/{experiment}-MEG/{sub_id}_{paradigm}_{experiment.lower()}_preprocessed-raw.fif'
        trigger_path = f'./MEGDataset/1-Raw data/{sub_id}/5-{paradigm}/{experiment}-MEG/{sub_id}_{paradigm}_{experiment.lower()}.json'

        raw = mne.io.read_raw_fif(data_path, preload=True)
        trigger = extract_hand_list(trigger_path)

        if experiment == 'SQUID':
            events = mne.find_events(raw, stim_channel='UPPT001')
        elif experiment == 'OPM':
            events = mne.find_events(raw, stim_channel='Trigger')

        if len(events) != len(trigger):
            print('trigger数量不匹配')
            events[:,-1] = trigger[0:len(events)]
        else:
            events[:,-1] = trigger
            
        epochs = mne.Epochs(raw, events, tmin=-1, tmax=6, preload=True, baseline=(-0.2, 0)).resample(200)

        raws_experiment.append(raw)
        epochs_experiment.append(epochs)

    raws_all[experiment] = raws_experiment
    epochs_all[experiment] = epochs_experiment




# %%
# MRCP
import copy
from my_code.utils.wfl_preproc_dss1 import nt_dss1
evokeds_all = dict()
for experiment in ['OPM', 'SQUID']:
    evokeds_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        raw = raws_all[experiment][sub_idx]
        raw_filt = raw.copy().filter(2, 5, fir_design='firwin')

        # proj = mne.preprocessing.compute_proj_hfc(raw_filt.info)
        # raw_hfc = raw_filt.copy().add_proj(proj).apply_proj()

        events = copy.deepcopy(epochs_all[experiment][sub_idx].events)
        if experiment == 'OPM':
            events[:,0] = events[:,0] + 0.09*raw_filt.info['sfreq']
        epochs = mne.Epochs(raw_filt, events, tmin=-1, tmax=6, preload=True, baseline=(-0.2, 0))

        del raw, raw_filt

        epochs_left = nt_dss1(epochs['1'],nremove=5)
        epochs_right = nt_dss1(epochs['2'],nremove=5)

        evoked_left = epochs['1'].average()
        evoked_right = epochs['2'].average()

        evokeds_experiment.append([evoked_left, evoked_right])

    evokeds_all[experiment] = evokeds_experiment



# %%
evokeds_mean = dict()

for experiment in ['OPM', 'SQUID']:
    evokeds_mean[experiment] = []
    for event_idx in range(2):  # 0: left, 1: right
        bads_all = []
        evoked_data_all = []
        for sub_idx, sub_id in enumerate(sub_ids):
            # 当前被试对应的 evoked
            evoked = evokeds_all[experiment][sub_idx][event_idx]
            # 收集 bad channels
            bads_ch = evoked.info['bads']
            for ch in bads_ch:
                if ch not in bads_all:
                    bads_all.append(ch)

            # SQUID取反，OPM保持原始方向
            if experiment == 'SQUID':
                evoked_data_all.append(-evoked.get_data())
            elif experiment == 'OPM':
                evoked_data_all.append(evoked.get_data())

        # 所有被试共同有效的通道
        goods_ch = [ch for ch in evoked.info['ch_names'] if ch not in bads_all]
        goods_ch_idx = [evoked.info['ch_names'].index(ch)for ch in goods_ch]

        # 被试间平均
        evoked_data_all_mean = np.mean(evoked_data_all,axis=0)

        # 创建平均后的 Evoked
        evoked_mean = evoked.copy()
        evoked_mean._data = evoked_data_all_mean

        # 设置 bad channels
        evoked_mean.info['bads'] = bads_all

        evokeds_mean[experiment].append(evoked_mean)


# %%
if not os.path.exists(f'./result/5-{paradigm}/evokeds_mean.pkl'):
    with open(f'./result/5-{paradigm}/evokeds_mean.pkl','wb') as f:
        pickle.dump(evokeds_mean,f)
else:
    print(f'./result/5-{paradigm}/evokeds_mean.pkl already exists')



# %%
if os.path.exists(f'./result/5-{paradigm}/evokeds_mean.pkl'):
    with open(f'./result/5-{paradigm}/evokeds_mean.pkl','rb') as f:
        evokeds_mean = pickle.load(f)
else:
    raise FileNotFoundError(f'./result/5-{paradigm}/evokeds_mean.pkl')


# %%
evoked_left_opm = evokeds_mean['OPM'][0]
evoked_right_opm = evokeds_mean['OPM'][1]

evoked_left_squid = evokeds_mean['SQUID'][0]
evoked_right_squid = evokeds_mean['SQUID'][1]

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(10, 3.5),constrained_layout=True,
                         gridspec_kw={'width_ratios': [4,2,0.6,4,2,0.6], 'height_ratios': [1,1]})

experiments = ['OPM', 'SQUID']

axes_todo = [axes[0,0],axes[0,3],axes[1,0],axes[1,3]]
evokeds_mean_todo = [evoked_left_opm,evoked_right_opm,evoked_left_squid,evoked_right_squid]

ylims = [[-120,120],[-120,120],[-120,120],[-120,120]]
for i in range(len(axes_todo)):
    ax = axes_todo[i]
    evokeds_mean_todo[i].plot(axes=ax,gfp=True)
    for text in ax.texts:
        text.remove()
    ax.set_title('')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (fT)')
    ax.set_xlim([-0.2, 1.5])
    ax.set_ylim([ylims[i][0]*1.08,ylims[i][1]*1.08])
    ax.set_yticks([ylims[i][0],ylims[i][0]/2,0,ylims[i][1]/2,ylims[i][1]])

    # ax.text(0.8,0.85,experiments[i],fontsize=12,fontweight='bold',transform=ax.transAxes)



vlims = [[-100,100],[-100,100],[-100,100],[-100,100]]

axes_todo = [axes[0,1],axes[0,4],axes[1,1],axes[1,4]]

for i in range(len(axes_todo)):
    ax = axes_todo[i]
    ax.axis('off')
    # ax = fig.add_axes([0.55-i*0.3, 0.6, 0.4, 0.4])
    ax = fig.add_axes([0.22+0.5*(i%2), 0.67-0.49*(i//2), 0.3, 0.3])
    info = mne.io.read_info(f'./supports/raw_{experiments[i//2].lower()}_info_plot.fif')

    ch_names = evokeds_mean_todo[i].ch_names
    bad_chs = evokeds_mean_todo[i].info['bads']
    good_chs_idx = [i for i,ch in enumerate(ch_names) if ch not in bad_chs]
    info = mne.pick_info(info, sel=good_chs_idx)
    if i == 0 or i == 1: 
        t_ori = 0.217
        t_ori_idx = np.argmin(np.abs(evokeds_mean_todo[i].times - t_ori))
        topo,_=mne.viz.plot_topomap(evokeds_mean_todo[i].data[good_chs_idx,t_ori_idx]*1e15, info, axes=ax, sensors=True,sphere=0.115,
                         contours=5,cmap='RdBu_r',extrapolate='head',vlim=vlims[i])
    else:
        t_ori = 0.217
        t_ori_idx = np.argmin(np.abs(evokeds_mean_todo[i].times - t_ori))
        topo,_=mne.viz.plot_topomap(evokeds_mean_todo[i].data[good_chs_idx,t_ori_idx]*1e15, info, axes=ax, sensors=True,sphere=0.115,
                         contours=5,cmap='RdBu_r',vlim=vlims[i])

axes_todo = [axes[0,2],axes[0,5],axes[1,2],axes[1,5]]
for i in range(len(axes_todo)):
    ax = axes_todo[i]
    ax.axis('off')
    cax = fig.add_axes([0.43+0.5*(i%2), 0.71-0.5*(i//2), 0.01, 0.2])

    cb = plt.colorbar(topo, cax=cax)
    # cb.set_label('Amplitude (fT)',labelpad=-20)


for i in range(4):
    plt.delaxes(fig.axes[12])

plt.show()


fig.savefig(f'./fig/5-{paradigm}/evoked.png',dpi=900)
fig.savefig(f'./fig/5-{paradigm}/evoked.pdf')
fig.savefig(f'./fig/5-{paradigm}/evoked.svg')





















# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import mne
from mne.time_frequency import tfr_multitaper

# %%
tfr_all = dict()
for experiment in ['OPM','SQUID']:
    tfr_all[experiment] = dict()
    for event_idx,event_key in enumerate(['1','2']):
        tfr_all[experiment][event_key] = []
        for sub_idx, sub_id in enumerate(sub_ids):
            if os.path.exists(f'./result/5-{paradigm}/tfr_sub/tfr_{experiment}_{sub_id}_{event_key}.pkl'):
                with open(f'./result/5-{paradigm}/tfr_sub/tfr_{experiment}_{sub_id}_{event_key}.pkl','rb') as file:
                    tfr_bc_ave = pickle.load(file)
                print(f'{sub_id}_{event_key} already exists')
            else:
                epochs_tf = epochs_all[experiment][sub_idx][event_key]
                # multitaper
                freqs = np.arange(2, 40)
                tfr = tfr_multitaper(epochs_tf,
                            freqs=freqs,
                            n_cycles=freqs/2,
                            use_fft=True,
                            return_itc=False,
                            average=True,
                            decim=2,
                            )

                # 基线校正
                baseline = (-0.5, 0)
                t = tfr.times
                f = tfr.freqs
                baseline_idx = np.where((t<=baseline[1]) & (t>=baseline[0]))[0]

                tfr_bc_ave = []
                for ch in tfr.ch_names:
                    tfr_ch = np.squeeze(tfr.copy().pick(picks=ch)._data)
                    tfr_baseline_mean_ave = np.mean(tfr_ch[:, baseline_idx],axis=1)
                    tfr_baseline_mean = np.tile(tfr_baseline_mean_ave[:,np.newaxis],(1,len(t)))
                    tfr_bc = (tfr_ch - tfr_baseline_mean) / tfr_baseline_mean
                    tfr_bc_ave.append(tfr_bc)
                tfr_bc_ave = np.array(tfr_bc_ave)

                with open(f'./result/5-{paradigm}/tfr_sub/tfr_{experiment}_{sub_id}_{event_key}.pkl','wb') as file:
                    pickle.dump(tfr_bc_ave, file)

            tfr_all[experiment][event_key].append(tfr_bc_ave)



# %%
if not os.path.exists(f'./result/5-{paradigm}/tfr_all.pkl'):
    with open(f'./result/5-{paradigm}/tfr_all.pkl','wb') as file:
        pickle.dump([f,t,tfr_all],file)
else:
    raise FileExistsError(f'./result/5-{paradigm}/tfr_all.pkl already exists')


# %%
if os.path.exists(f'./result/5-{paradigm}/tfr_all.pkl'):
    with open(f'./result/5-{paradigm}/tfr_all.pkl','rb') as file:
        f,t,tfr_all = pickle.load(file)
else:
    raise FileNotFoundError(f'./result/5-{paradigm}/tfr_all.pkl not found')


# %%
# 计算所有通道β频段的SNR：MRBD(movemen related beta decrease)窗口, PMBR(post movement beta rebound)
# SNR定义为: 两个窗口之间的平均信号幅度差除以MRBD窗口中信号的标准偏差
beta = [13,30]
t_mrbd = [0.5,3]
t_pmbr = [3,5]
f_ori_idx = [np.argmin(np.abs(f-beta[0])), np.argmin(np.abs(f-beta[1]))]
t_mrbd_idx = [np.argmin(np.abs(t-t_mrbd[0])), np.argmin(np.abs(t-t_mrbd[1]))]
t_pmbr_idx = [np.argmin(np.abs(t-t_pmbr[0])), np.argmin(np.abs(t-t_pmbr[1]))]
SNR_all = dict()
tfr_pick_all = dict()
for experiment in ['OPM','SQUID']:
    SNR_all[experiment] = dict()
    tfr_pick_all[experiment] = dict()
    for event_idx,event_key in enumerate(['1','2']):
        SNR_all[experiment][event_key] = []
        tfr_pick_all[experiment][event_key] = []
        for sub_idx, sub_id in enumerate(sub_ids):
            tfr = tfr_all[experiment][event_key][sub_idx] # (通道数, 频率数, 时间点数)
            tfr_beta = tfr[:,f_ori_idx[0]:f_ori_idx[1],:] # (通道数, 频率数, 时间点数)
            tfr_beta_mrbd_mean = np.mean(tfr_beta[:,:,t_mrbd_idx[0]:t_mrbd_idx[1]],axis=(1,2))
            tfr_beta_pmbr_mean = np.mean(tfr_beta[:,:,t_pmbr_idx[0]:t_pmbr_idx[1]],axis=(1,2))

            tfr_beta_mrbd_std = np.std(tfr_beta[:,:,t_mrbd_idx[0]:t_mrbd_idx[1]],axis=(1,2))

            SNR = (tfr_beta_pmbr_mean - tfr_beta_mrbd_mean) / tfr_beta_mrbd_std
            SNR_max_idx = np.argmax(SNR)
            tfr_pick = tfr[SNR_max_idx,:,:]

            SNR_all[experiment][event_key].append(SNR)
            tfr_pick_all[experiment][event_key].append(tfr_pick)



# %%
tfr_opm_left_mean = np.mean(tfr_pick_all['OPM']['1'],axis=0)
tfr_opm_left_std = np.std(tfr_pick_all['OPM']['1'],axis=0)
tfr_opm_right_mean = np.mean(tfr_pick_all['OPM']['2'],axis=0)
tfr_opm_right_std = np.std(tfr_pick_all['OPM']['2'],axis=0)

tfr_squid_left_mean = np.mean(tfr_pick_all['SQUID']['1'],axis=0)
tfr_squid_left_std = np.std(tfr_pick_all['SQUID']['1'],axis=0)
tfr_squid_right_mean = np.mean(tfr_pick_all['SQUID']['2'],axis=0)
tfr_squid_right_std = np.std(tfr_pick_all['SQUID']['2'],axis=0)


# %%
f_ori = [13,30]
f_ori_idx = [np.argmin(np.abs(f-f_ori[0])), np.argmin(np.abs(f-f_ori[1]))]

tfr_squid_left_mean_erds = tfr_squid_left_mean[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)
tfr_squid_left_std_erds = tfr_squid_left_std[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)
tfr_squid_right_mean_erds = tfr_squid_right_mean[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)
tfr_squid_right_std_erds = tfr_squid_right_std[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)

tfr_opm_left_mean_erds = tfr_opm_left_mean[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)
tfr_opm_left_std_erds = tfr_opm_left_std[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)
tfr_opm_right_mean_erds = tfr_opm_right_mean[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)
tfr_opm_right_std_erds = tfr_opm_right_std[f_ori_idx[0]:f_ori_idx[1],:].mean(axis=0)


# %%
tfr_mean_erds = [tfr_opm_left_mean_erds, tfr_opm_right_mean_erds,
                 tfr_squid_left_mean_erds, tfr_squid_right_mean_erds]

tfr_std_erds = [tfr_opm_left_std_erds, tfr_opm_right_std_erds,
                tfr_squid_left_std_erds, tfr_squid_right_std_erds,]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(2, 2, figsize=(5, 4), constrained_layout=True)
colors = ['#E9657F','#007ACC','#2ECC40','#F1C40F','#E74C3C','#00FFFF']
labels = ['OPM left', 'OPM right', 'SQUID left', 'SQUID right']
for i,ax in enumerate(axes.flatten()):
    ax.plot(t,tfr_mean_erds[i].T, color=colors[i//2],linewidth=1)
    ax.fill_between(t, tfr_mean_erds[i].T-tfr_std_erds[i].T, tfr_mean_erds[i].T+tfr_std_erds[i].T, alpha=0.3, color=colors[i//2])
    ax.hlines(0, t[0], t[-1], color='k', linestyle='--',linewidth=0.8)
    ax.vlines(0, -0.5, 1.5, color='k', linestyle='--',linewidth=0.8)
    ax.set_xlim(-0.5, 5.5)
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_ylim([-0.5,1.2])
    ax.set_yticks([-0.5,0,0.3,0.6,0.9,1.2])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Beta power change')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.12, 0.95, labels[i], transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontweight='normal', color=colors[i//2])
plt.show()

fig.savefig(f'./fig/5-{paradigm}/Beta_power_change.png',dpi=900)
fig.savefig(f'./fig/5-{paradigm}/Beta_power_change.svg')
fig.savefig(f'./fig/5-{paradigm}/Beta_power_change.pdf')




























# %%
def _gen_dics(active_win, baseline_win, epochs, fwd):
    freqs = np.logspace(np.log10(15), np.log10(30), 9)
    csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-1, tmax=5.5, decim=20)
    csd_baseline = mne.time_frequency.csd_morlet(epochs, freqs, tmin=baseline_win[0], tmax=baseline_win[1], decim=20)
    csd_ers = mne.time_frequency.csd_morlet(epochs, freqs, tmin=active_win[0], tmax=active_win[1], decim=20)
    filters = mne.beamformer.make_dics(
        epochs.info,
        fwd,
        csd.mean(),
        pick_ori="max-power",
        reduce_rank=True,
        real_filter=True,
        rank=rank,
    )
    stc_base, freqs = mne.beamformer.apply_dics_csd(csd_baseline.mean(), filters)
    stc_act, freqs = mne.beamformer.apply_dics_csd(csd_ers.mean(), filters)
    stc_act /= stc_base
    return stc_act



subjects_dir = './MEGDataset/0-MRI'
trans = mne.transforms.Transform('head', 'mri')

stcs_all = dict()
for experiment in ['OPM','SQUID']:
    stcs_all_experiment = []
    for sub_idx,sub_id in enumerate(sub_ids):
        stcs_sub = []
        for event_key in ['1','2']:
            epochs = epochs_all[experiment][sub_idx][event_key]

            if not os.path.exists(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif'):
                src = mne.setup_source_space(sub_id, spacing='oct6', add_dist='patch',
                                    subjects_dir=subjects_dir)
                src.save(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')
            else:
                src = mne.read_source_spaces(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')

            if not os.path.exists(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif'):
                conductivity = (0.3,)
                model = mne.make_bem_model(subject=sub_id, ico=4,
                                        conductivity=conductivity,
                                        subjects_dir=subjects_dir)
                bem = mne.make_bem_solution(model)
                mne.write_bem_solution(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif',bem)
            else:
                bem = mne.read_bem_solution(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif')

            if not os.path.exists(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif'):
                fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem,
                                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                                verbose=True)
                fwd.save(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')
            else:
                fwd = mne.read_forward_solution(f'./result/5-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')

            rank = mne.compute_rank(epochs, tol=1e-6, tol_kind="relative")
            active_win = (3.5, 5.5)
            baseline_win = (-1, 0)
            baseline_cov = mne.cov.compute_covariance(
                epochs,
                tmin=baseline_win[0],
                tmax=baseline_win[1],
                method="shrunk",
                rank=rank,
                verbose=True,
            )
            active_cov = mne.cov.compute_covariance(
                epochs,
                tmin=active_win[0],
                tmax=active_win[1],
                method="shrunk",
                rank=rank,
                verbose=True,
            )

            # Weighted averaging is already in the addition of covariance objects.
            common_cov = baseline_cov + active_cov
            # baseline_cov.plot(epochs.info)

            stc_dics = _gen_dics(active_win, baseline_win, epochs, fwd)

            stcs_sub.append(stc_dics)
        stcs_all_experiment.append(stcs_sub)
    stcs_all[experiment] = stcs_all_experiment


# %%
if not os.path.exists(f'./result/5-{paradigm}/stcs_all.pkl'):
    with open(f'./result/5-{paradigm}/stcs_all.pkl','wb') as f:
        pickle.dump(stcs_all,f)
else:
    raise FileExistsError(f'./result/5-{paradigm}/stcs_all.pkl')



# %%
if os.path.exists(f'./result/5-{paradigm}/stcs_all.pkl'):
    with open(f'./result/5-{paradigm}/stcs_all.pkl','rb') as f:
        stcs_all = pickle.load(f)
else:
    raise FileNotFoundError(f'./result/5-{paradigm}/stcs_all.pkl')

subjects_dir = './MEGDataset/0-MRI'




# %%
src_to = mne.setup_source_space("S4", spacing='oct6', add_dist='patch',
                                    subjects_dir=subjects_dir)
# src_to.save(f'./result/{paradigm}/source/wangfulong1_oct6_surface-src.fif')
# 投影到平均脑模版
stcs_fsaverage = dict()
for experiment in ['OPM','SQUID']:
    stc_fsaverage_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        stc_fsaverage_sub = []
        for event_idx, event_key in enumerate(['1','2']):
            stc = stcs_all[experiment][sub_idx][event_idx]

            morph = mne.compute_source_morph(
                stc,
                subject_from=sub_id,
                subject_to="S4",
                src_to=src_to,
                subjects_dir=subjects_dir)
            
            stc_fsaverage = morph.apply(stc)

            stc_fsaverage_sub.append(stc_fsaverage)

        stc_fsaverage_experiment.append(stc_fsaverage_sub)

    stcs_fsaverage[experiment] = stc_fsaverage_experiment



# %%
import matplotlib.colors as mcolors
colors = [
    (0.7, 0.7, 0.7),  # 灰色
    (1.0, 0.0, 0.0),  # 红
    (1.0, 0.5, 0.0),  # 橙
    (1.0, 1.0, 0.0),  # 黄
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
# clim = dict(kind="percent", lims=[90,95,100])
fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))
brain = stcs_fsaverage['OPM'][1][0].plot(
    hemi="lh",
    subject='S4',
    subjects_dir=subjects_dir,
    colormap=cmap,
    initial_time=0.6,
    time_viewer=True,
    show_traces=True,
    colorbar=True,
    # clim=clim,
    background='white',
    smoothing_steps=10,
    figure=fig,
)


# %%
# 计算平均源空间
stcs_mean = []
for experiment in ['OPM','SQUID']:
    stcs_mean_experiment = []
    for event_idx, event_key in enumerate(['1','2']):
        stcs_data_event = []
        for sub_idx, sub_id in enumerate(sub_ids):
            stc = stcs_fsaverage[experiment][sub_idx][event_idx]
            stc_data = stc.data
            # 归一化
            data_min = stc_data.min()
            data_max = stc_data.max()
            data_norm = (stc_data - data_min) / (data_max - data_min)
            stcs_data_event.append(data_norm)

        stcs_data_event = np.array(stcs_data_event) # subjects x sources x times
        # 跨受试者平均
        stcs_data_event_avg = np.mean(stcs_data_event, axis=0)  # sources × times
        # 归一化
        data_min = stcs_data_event_avg.min()
        data_max = stcs_data_event_avg.max()
        data_norm = (stcs_data_event_avg - data_min) / (data_max - data_min)

        stc_mean = stc.copy()
        stc_mean._data = data_norm

        stcs_mean_experiment.append(stc_mean)

    stcs_mean.append(stcs_mean_experiment)









# %%
import matplotlib.colors as mcolors
colors = [
    (0.7, 0.7, 0.7),  # 灰色
    (1.0, 0.0, 0.0),  # 红
    (1.0, 0.5, 0.0),  # 橙
    (1.0, 1.0, 0.0),  # 黄
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

screenshots = []
for i in range(2):
    if i == 0:
        clim = dict(kind="value", lims=[0.8,0.8,1])
    else:
        clim = dict(kind="value", lims=[0.9,0.9,1])
    fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))
    brain = stcs_mean[i][1].plot(
        hemi="lh",
        subject='S4',
        subjects_dir=subjects_dir,
        colormap=cmap,
        initial_time=0.6,
        time_viewer=False,
        show_traces=False,
        colorbar=False,
        clim=clim,
        background='white',
        smoothing_steps=10,
        figure=fig,
    )

    fig = brain._renderer
    fig.plotter.remove_all_lights()
    mne.viz.set_3d_view(brain, azimuth=0, elevation=-70,focalpoint='auto', distance=500)
    screenshot0=fig.plotter.screenshot(scale=10)
    screenshots.append(screenshot0)

    if i == 0:
        clim = dict(kind="value", lims=[0.95,0.95,1])
    else:
        clim = dict(kind="value", lims=[0.85,0.85,1])
    fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))
    brain = stcs_mean[i][0].plot(
        hemi="rh",
        subject='S4',
        subjects_dir=subjects_dir,
        colormap=cmap,
        initial_time=0.6,
        time_viewer=False,
        show_traces=False,
        colorbar=False,
        clim=clim,
        background='white',
        smoothing_steps=10,
        figure=fig,
    )

    fig = brain._renderer
    fig.plotter.remove_all_lights()
    mne.viz.set_3d_view(brain, azimuth=-180, elevation=-70,focalpoint='auto', distance=500)
    screenshot1=fig.plotter.screenshot(scale=10)
    screenshots.append(screenshot1)

# %%
cropped_screenshot_all = []
for screenshot in screenshots:
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    cropped_screenshot_all.append(cropped_screenshot)


# %%
import matplotlib.colors as mcolors
import matplotlib.cm as cm
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(5, 4),constrained_layout=True,
                         gridspec_kw={'width_ratios': [1,1], 'height_ratios': [1,0.3,0.05,1,0.3]})
# plt.subplots_adjust(top=0.97,bottom=0.02,left=0,right=0.995,wspace=0.1,hspace=0.3)
colors = ['#E9657F','#007ACC','#2ECC40','#F1C40F','#E74C3C','#00FFFF']

colors = [
    (1.0, 0.0, 0.0),  # 红
    (1.0, 0.5, 0.0),  # 橙
    (1.0, 1.0, 0.0),  # 黄
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

ax = axes[0,0]
im = ax.imshow(cropped_screenshot_all[1])
ax.axis('off')
ax = axes[1,0]
ax.axis('off')
cax = fig.add_axes([0.15, 0.6, 0.2, 0.02])
norm = mcolors.Normalize(vmin=0.9, vmax=1.0)
im = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = plt.colorbar(im, cax=cax, orientation='horizontal')
cb.set_ticks([0.9, 1])
cb.set_label('Normalized source power',labelpad=0)



ax = axes[0,1]
ax.imshow(cropped_screenshot_all[0])
ax.axis('off')
ax = axes[1,1]
ax.axis('off')
cax = fig.add_axes([0.65, 0.6, 0.2, 0.02])
norm = mcolors.Normalize(vmin=0.9, vmax=1.0)
im = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = plt.colorbar(im, cax=cax, orientation='horizontal')
cb.set_ticks([0.9, 1])
cb.set_label('Normalized source power',labelpad=0)



axes[2,0].axis('off')
axes[2,1].axis('off')




ax = axes[3,0]
im = ax.imshow(cropped_screenshot_all[3])
ax.axis('off')
ax = axes[4,0]
ax.axis('off')
cax = fig.add_axes([0.15, 0.1, 0.2, 0.02])
norm = mcolors.Normalize(vmin=0.8, vmax=1.0)
im = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = plt.colorbar(im, cax=cax, orientation='horizontal')
cb.set_ticks([0.8, 1])
cb.set_label('Normalized source power',labelpad=0)



ax = axes[3,1]
ax.imshow(cropped_screenshot_all[2])
ax.axis('off')
ax = axes[4,1]
ax.axis('off')
cax = fig.add_axes([0.65, 0.1, 0.2, 0.02])
norm = mcolors.Normalize(vmin=0.9, vmax=1.0)
im = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = plt.colorbar(im, cax=cax, orientation='horizontal')
cb.set_ticks([0.9, 1])
cb.set_label('Normalized source power',labelpad=0)


fig.text(0.5,0.97,'OPM', ha='center', va='center',fontsize=11)
fig.text(0.5,0.45,'SQUID', ha='center', va='center',fontsize=11)

plt.show()

fig.savefig(f'./fig/5-{paradigm}/stc.png', dpi=900)
