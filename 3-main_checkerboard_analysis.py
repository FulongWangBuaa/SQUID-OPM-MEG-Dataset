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
paradigm = 'checkerboard'

# %%
raws_all = dict()
epochs_all = dict()
for experiment in ['OPM', 'SQUID']:
    raws_experiment = []
    epochs_experiment = []
    for sub_id in sub_ids:
        data_path = f'./MEGDataset/2-Preprocessed data/{sub_id}/3-{paradigm}/{experiment}-MEG/{sub_id}_{paradigm}_{experiment.lower()}_preprocessed-raw.fif'

        raw = mne.io.read_raw_fif(data_path, preload=True)

        if experiment == 'SQUID':
            events = mne.find_events(raw, stim_channel='UPPT001',min_duration = 10/raw.info['sfreq'])
            if sub_id == 'S1':
                events[:,0] = events[:,0] - 0.013*raw.info['sfreq']
            elif sub_id == 'S2':
                events[:,0] = events[:,0] + 0.043*raw.info['sfreq']
            elif sub_id == 'S3':
                events[:,0] = events[:,0] - 0.003*raw.info['sfreq']
            elif sub_id == 'S5':
                events[:,0] = events[:,0] - 0.009*raw.info['sfreq']
            elif sub_id == 'S6':
                events[:,0] = events[:,0] - 0.005*raw.info['sfreq']

        elif experiment == 'OPM':
            events = mne.find_events(raw, stim_channel='Trigger')
            if sub_id == 'S1':
                events[:,0] = events[:,0] + 0.067*raw.info['sfreq']
            elif sub_id == 'S2':
                events[:,0] = events[:,0] + 0.103*raw.info['sfreq']
            elif sub_id == 'S4':
                events[:,0] = events[:,0] + 0.11*raw.info['sfreq']
            elif sub_id == 'S5':
                events[:,0] = events[:,0] + 0.067*raw.info['sfreq']
            

        epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, preload=True, baseline=(-0.2, 0))
        from my_code.utils.auto_reject_trial import get_bad_trials
        bad_trials = get_bad_trials(epochs, thresh_val=3)
        epochs = epochs.drop(bad_trials)

        raws_experiment.append(raw)
        epochs_experiment.append(epochs)

    raws_all[experiment] = raws_experiment
    epochs_all[experiment] = epochs_experiment




# %%
from my_code.utils.snr import SNRLB
SNRs = dict()
for experiment in ['OPM', 'SQUID']:
    snrs_lb_experiment = []
    snrs_ub_experiment = []
    snrs_mean_experiment = []
    snrs_dist_sorted_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        epochs = epochs_all[experiment][sub_idx]
        snrs_lb,snrs_ub,snrs_mean,snrs_dist_sorted = SNRLB(epochs, 
                                                           pre_time=-0.1,
                                                           post_time=0.3, 
                                                           percent_ci=0.9, 
                                                           num_bootstraps=9999,
                                                           return_all_snr=True)
        snrs_lb_experiment.append(snrs_lb)
        snrs_ub_experiment.append(snrs_ub)
        snrs_mean_experiment.append(snrs_mean)
        snrs_dist_sorted_experiment.append(snrs_dist_sorted)
    SNRs[experiment] = dict()
    SNRs[experiment]['lb'] = snrs_lb_experiment
    SNRs[experiment]['ub'] = snrs_ub_experiment
    SNRs[experiment]['mean'] = snrs_mean_experiment
    SNRs[experiment]['dist_sorted'] = snrs_dist_sorted_experiment


# %%
if not os.path.exists(f'./result/3-{paradigm}/SNRs.pkl'):
    with open(f'./result/3-{paradigm}/SNRs.pkl','wb') as f:
        pickle.dump(SNRs,f)
else:
    raise FileExistsError(f'./result/3-{paradigm}/SNRs.pkl')




# %%
if os.path.exists(f'./result/3-{paradigm}/SNRs.pkl'):
    with open(f'./result/3-{paradigm}/SNRs.pkl','rb') as f:
        SNRs = pickle.load(f)
else:
    raise FileNotFoundError(f'./result/3-{paradigm}/SNRs.pkl')

SNRs_temp = dict()
for e_idx, experiment in enumerate(['OPM', 'SQUID']):
    SNRs_temp[experiment] = dict()
    SNRs_temp[experiment]['lb'] = []
    SNRs_temp[experiment]['ub'] = []
    SNRs_temp[experiment]['mean'] = []
    SNRs_temp[experiment]['dist_sorted'] = []
    for sub_idx, sub_id in enumerate(sub_ids):
        epochs = epochs_all[experiment][sub_idx]

        ch_names_good = epochs.copy().pick(picks='data',exclude='bads').ch_names
        # temporal_chs = [ch for ch in ch_names_good if 'T' in ch]
        # temporal_chs_idx = [ch_names_good.index(ch) for ch in temporal_chs]
        n = int(len(ch_names_good)*0.2)
        idx = np.argpartition(np.array(SNRs[experiment]['lb'][sub_idx]), -n)[-n:]

        SNRs_temp[experiment]['lb'].append(np.array(SNRs[experiment]['lb'][sub_idx])[idx])
        SNRs_temp[experiment]['ub'].append(np.array(SNRs[experiment]['ub'][sub_idx])[idx])
        SNRs_temp[experiment]['mean'].append(np.array(SNRs[experiment]['mean'][sub_idx])[idx])
        SNRs_temp[experiment]['dist_sorted'].append(np.array(SNRs[experiment]['dist_sorted'][sub_idx])[idx])



# %%
from scipy.stats import gaussian_kde, norm
values_opm = np.array(SNRs_temp['OPM']['dist_sorted']).flatten()
values_squid = np.array(SNRs_temp['SQUID']['dist_sorted']).flatten()  # (n_sources,)

# 直方图
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2),constrained_layout=True,
                         gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})
colors = ['#E9657F','#007ACC','#2ECC40','#F1C40F','#E74C3C','#00FFFF']
for i,(f_values, label) in enumerate(zip([values_opm,values_squid], ['OPM', 'SQUID'])):
    counts, bins, _ = plt.hist(f_values, bins=80, alpha=0.3, density=True, label=label,
                            color=colors[i],rwidth=1, edgecolor='k', linewidth=0.5)

    x = np.linspace(f_values.min(), f_values.max(), 500)

    mu, std = norm.fit(f_values)
    pdf = norm.pdf(x, mu, std)
    ax.plot(x, pdf, color=colors[i], lw=2, linestyle='-')

    # -----------------------
    # ax.set_yticks([0,5000,10000,15000,20000,25000])
    ax.set_xlim(-5,25)
    ax.set_ylim(0,0.16)
    ax.set_yticks([0,0.04,0.08,0.12,0.16])
    ax.set_xlabel('SNR')
    ax.set_ylabel('Density')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc=(0.7,0.5),frameon=False)
plt.show()


fig.savefig(f'./fig/3-{paradigm}/SNR_dist.png',dpi=900)
fig.savefig(f'./fig/3-{paradigm}/SNR_dist.pdf')
fig.savefig(f'./fig/3-{paradigm}/SNR_dist.svg')




# %%
# 平均诱发波形
evokeds_mean = []
for experiment in ['OPM', 'SQUID']:
    bads_all = []
    evoked_data_all = []
    for sub_idx, sub_id in enumerate(sub_ids):
        epochs = epochs_all[experiment][sub_idx]
        evoked = epochs.average()
        
        bads_ch = evoked.info['bads']
        for ch in bads_ch:
            if ch not in bads_all:
                bads_all.append(ch)

        if experiment == 'SQUID' and sub_id in ['S2','S4','S5']:
            evoked_data_all.append(evoked.get_data())
        elif experiment == 'SQUID' and sub_id in ['S1','S3','S6']:
            evoked_data_all.append(-evoked.get_data())

        if experiment == 'OPM' and sub_id in ['S1']:
            evoked_data_all.append(evoked.get_data())
        elif experiment == 'OPM' and sub_id in ['S2','S3','S4', 'S5', 'S6']:
            evoked_data_all.append(-evoked.get_data())

        
    goods_ch = [ch for ch in evoked.info['ch_names'] if ch not in bads_all]
    goods_ch_idx = [evoked.info['ch_names'].index(ch) for ch in goods_ch]
    evoked_data_all_mean = np.mean(evoked_data_all,axis=0)

    evoked_mean = evoked.copy()
    evoked_mean._data = evoked_data_all_mean
    evoked_mean.info['bads'] = bads_all

    evokeds_mean.append(evoked_mean)

if not os.path.exists(f'./result/3-{paradigm}/evokeds_mean.pkl'):
    with open(f'./result/3-{paradigm}/evokeds_mean.pkl','wb') as f:
        pickle.dump(evokeds_mean,f)
else:
    print(f'./result/3-{paradigm}/evokeds_mean.pkl already exists')



# %%
if os.path.exists(f'./result/3-{paradigm}/evokeds_mean.pkl'):
    with open(f'./result/3-{paradigm}/evokeds_mean.pkl','rb') as f:
        evokeds_mean = pickle.load(f)
else:
    raise FileNotFoundError(f'./result/3-{paradigm}/evokeds_mean.pkl')


# %%
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 4),constrained_layout=True,
                         gridspec_kw={'width_ratios': [6,4,0.6], 'height_ratios': [1,1]})

experiments = ['OPM', 'SQUID']

ylims = [[-150,150],[-120,120]]
for i in range(len(experiments)):
    ax = axes[i,0]
    evokeds_mean[i].plot(axes=ax,gfp=False)
    for text in ax.texts:
        text.remove()
    ax.set_title('')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (fT)')
    ax.set_xlim([-0.2, 0.8])
    ax.set_ylim([ylims[i][0]*1.08,ylims[i][1]*1.08])
    ax.set_yticks([ylims[i][0],ylims[i][0]/2,0,ylims[i][1]/2,ylims[i][1]])

    ax.text(0.8,0.85,experiments[i],fontsize=12,fontweight='bold',transform=ax.transAxes)



vlims = [[-100,100],[-100,100]]

for i in range(len(experiments)):
    ax = axes[i,1]
    ax.axis('off')
    ax = fig.add_axes([0.55, 0.6-i*0.5, 0.4, 0.4])
    info = mne.io.read_info(f'./supports/raw_{experiments[i].lower()}_info_plot.fif')

    ch_names = evokeds_mean[i].ch_names
    bad_chs = evokeds_mean[i].info['bads']
    good_chs_idx = [i for i,ch in enumerate(ch_names) if ch not in bad_chs]
    info = mne.pick_info(info, sel=good_chs_idx)
    if i == 0: 
        t_ori = 0.147
        t_ori_idx = np.argmin(np.abs(evokeds_mean[i].times - t_ori))
        topo,_=mne.viz.plot_topomap(evokeds_mean[i].data[good_chs_idx,t_ori_idx]*1e15, info, axes=ax, sensors=True,sphere=0.115,
                         contours=5,cmap='RdBu_r',extrapolate='head',vlim=vlims[i])
    else:
        t_ori = 0.147
        t_ori_idx = np.argmin(np.abs(evokeds_mean[i].times - t_ori))
        topo,_=mne.viz.plot_topomap(evokeds_mean[i].data[good_chs_idx,t_ori_idx]*1e15, info, axes=ax, sensors=True,sphere=0.115,
                         contours=5,cmap='RdBu_r',vlim=vlims[i])


    axes[i,2].axis('off')
    cax = fig.add_axes([0.9, 0.6-i*0.5, 0.015, 0.35])
    cb = plt.colorbar(topo, cax=cax)
    cb.set_ticks(vlims[i])
    cb.set_label('Amplitude (fT)',labelpad=-20)


for i in range(len(experiments)):
    plt.delaxes(fig.axes[6])

plt.show()


fig.savefig(f'./fig/3-{paradigm}/evoked.png',dpi=900)
fig.savefig(f'./fig/3-{paradigm}/evoked.pdf')
fig.savefig(f'./fig/3-{paradigm}/evoked.svg')







# %%
subjects_dir = './MEGDataset/0-MRI'
trans = mne.transforms.Transform('head', 'mri')

stcs_all = dict()

for experiment in ['OPM','SQUID']:
    stcs_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        epochs = epochs_all[experiment][sub_idx]
        evoked = epochs.average()

        if not os.path.exists(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif'):
            src = mne.setup_source_space(sub_id, spacing='oct6', add_dist='patch',
                                subjects_dir=subjects_dir)
            src.save(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')
        else:
            src = mne.read_source_spaces(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')

        if not os.path.exists(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif'):
            conductivity = (0.3,)
            model = mne.make_bem_model(subject=sub_id, ico=4,
                                    conductivity=conductivity,
                                    subjects_dir=subjects_dir)
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif',bem)
        else:
            bem = mne.read_bem_solution(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif')

        if not os.path.exists(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif'):
            fwd = mne.make_forward_solution(evoked.info, trans=trans, src=src, bem=bem,
                                            meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                            verbose=True)
            fwd.save(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')
        else:
            fwd = mne.read_forward_solution(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')

        noise_cov = mne.compute_covariance(epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True)

        inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, 
                                                                noise_cov, loose=0.2, depth=0.8)

        method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
        snr = 3.0
        lambda2 = 1.0 / snr**2
        stc, residual = mne.minimum_norm.apply_inverse(
            evoked,
            inverse_operator,
            lambda2,
            method=method,
            pick_ori=None,
            return_residual=True,
            verbose=True,
        )

        stcs_experiment.append(stc)
    stcs_all[experiment] = stcs_experiment


# %%
if not os.path.exists(f'./result/3-{paradigm}/stcs_all.pkl'):
    with open(f'./result/3-{paradigm}/stcs_all.pkl','wb') as f:
        pickle.dump(stcs_all,f)
else:
    raise FileExistsError(f'./result/3-{paradigm}/stcs_all.pkl')




# %%
# 计算源空间信噪比
stcs_snr_all = dict()
for experiment in ['OPM','SQUID']:
    stc_snr_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        stc = stcs_all[experiment][sub_idx]
        epochs = epochs_all[experiment][sub_idx]
        noise_cov = mne.compute_covariance(epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True)
        fwd = mne.read_forward_solution(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')
        stc_snr = stc.estimate_snr(epochs.info, fwd, noise_cov)
        stc_snr_experiment.append(stc_snr)
    stcs_snr_all[experiment] = stc_snr_experiment

# %%
if not os.path.exists(f'./result/3-{paradigm}/stcs_snr_all.pkl'):
    with open(f'./result/3-{paradigm}/stcs_snr_all.pkl','wb') as f:
        pickle.dump(stcs_snr_all,f)
else:
    raise FileExistsError(f'./result/3-{paradigm}/stcs_snr_all.pkl')









# %%
if os.path.exists(f'./result/3-{paradigm}/stcs_all.pkl'):
    with open(f'./result/3-{paradigm}/stcs_all.pkl','rb') as f:
        stcs_all = pickle.load(f)
else:
    raise FileNotFoundError(f'./result/3-{paradigm}/stcs_all.pkl')


if os.path.exists(f'./result/3-{paradigm}/stcs_snr_all.pkl'):
    with open(f'./result/3-{paradigm}/stcs_snr_all.pkl','rb') as f:
        stcs_snr_all = pickle.load(f)
else:
    raise FileNotFoundError(f'./result/3-{paradigm}/stcs_snr_all.pkl')


subjects_dir = './MEGDataset/0-MRI'





# %%
# 投影到平均脑模版
stcs_fsaverage = dict()
stcs_snr_fsaverage = dict()
for experiment in ['OPM','SQUID']:
    stc_fsaverage_experiment = []
    stc_snr_fsaverage_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        # if sub_idx != 0:
            stc = stcs_all[experiment][sub_idx]
            stc_snr = stcs_snr_all[experiment][sub_idx]

            morph = mne.compute_source_morph(
                stc,
                subject_from=sub_id,
                subject_to="S4",
                # src_to=src_to,
                subjects_dir=subjects_dir)
            
            morph_snr = mne.compute_source_morph(
                stc_snr,
                subject_from=sub_id,
                subject_to="S4",
                # src_to=src_to,
                subjects_dir=subjects_dir)

            stc_fsaverage = morph.apply(stc)
            stc_snr_fsaverage = morph_snr.apply(stc_snr)

            stc_fsaverage_experiment.append(stc_fsaverage)
            stc_snr_fsaverage_experiment.append(stc_snr_fsaverage)

    stcs_fsaverage[experiment] = stc_fsaverage_experiment
    stcs_snr_fsaverage[experiment] = stc_snr_fsaverage_experiment








# %%
# 计算平均源空间
stcs_mean = []
stcs_snr_mean = []
for experiment in ['OPM','SQUID']:
    stcs_data_experiment = []
    stcs_snr_data_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        stc = stcs_fsaverage[experiment][sub_idx]
        stc_data = stc.data
        data_min = stc_data.min()     # (1, 1201)
        data_max = stc_data.max()
        data_norm = (stc_data - data_min) / (data_max - data_min)
        stcs_data_experiment.append(data_norm)

        stc_snr = stcs_snr_fsaverage[experiment][sub_idx]
        stc_snr_data = stc_snr.data
        stcs_snr_data_experiment.append(stc_snr_data)

    stcs_data_experiment_mean = np.mean(stcs_data_experiment,axis=0)
    data_min = stcs_data_experiment_mean.min()     # (1, 1201)
    data_max = stcs_data_experiment_mean.max()
    data_norm = (stcs_data_experiment_mean - data_min) / (data_max - data_min)
    stc_mean = stc.copy()
    stc_mean._data = data_norm
    stcs_mean.append(stc_mean)

    
    stcs_snr_data_experiment_mean = np.mean(stcs_snr_data_experiment,axis=0)
    stc_snr_mean = stc_snr.copy()
    stc_snr_mean._data = stcs_snr_data_experiment_mean
    stcs_snr_mean.append(stc_snr_mean)














# %%
import matplotlib.colors as mcolors
colors = [
    (0.7, 0.7, 0.7),  # 灰色
    (1.0, 0.0, 0.0),  # 红
    (1.0, 0.5, 0.0),  # 橙
    (1.0, 1.0, 0.0),  # 黄
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
clim = dict(kind="value", lims=[0.6,0.6,1])
fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))
brain = stcs_mean[0].plot(
    hemi="lh",
    subject="S4",
    subjects_dir=subjects_dir,
    colormap=cmap,
    initial_time=0.147,
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
mne.viz.set_3d_view(brain, azimuth=0, elevation=-60,focalpoint='auto', distance=500)
screenshot_opm=fig.plotter.screenshot(scale=10)



# %%
colors = [
    (0.7, 0.7, 0.7),  # 灰色
    (1.0, 0.0, 0.0),  # 红
    (1.0, 0.5, 0.0),  # 橙
    (1.0, 1.0, 0.0),  # 黄
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
clim = dict(kind="value", lims=[0.6,0.6,1])
fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))
brain = stcs_mean[1].plot(
    hemi="lh",
    subject="S4",
    subjects_dir=subjects_dir,
    colormap=cmap,
    initial_time=0.147,
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
mne.viz.set_3d_view(brain, azimuth=0, elevation=-60,focalpoint='auto', distance=500)
screenshot_squid=fig.plotter.screenshot(scale=10)









# %%
cropped_screenshot_all = []
for screenshot in [screenshot_opm, screenshot_squid]:
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    cropped_screenshot_all.append(cropped_screenshot)



# %%
import matplotlib.colors as mcolors
import matplotlib.cm as cm
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 2),constrained_layout=False,
                         gridspec_kw={'width_ratios': [1,1,0.3], 'height_ratios': [1]})
plt.subplots_adjust(top=0.93,bottom=0,left=0.005,right=0.995,wspace=0.2,hspace=0.3)
colors = ['#E9657F','#007ACC','#2ECC40','#F1C40F','#E74C3C','#00FFFF']

colors = [
    (1.0, 0.0, 0.0),  # 红
    (1.0, 0.5, 0.0),  # 橙
    (1.0, 1.0, 0.0),  # 黄
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

ax = axes[0]
im = ax.imshow(cropped_screenshot_all[0])
ax.axis('off')
ax.set_title('OPM', fontsize=12, fontweight='normal')



ax = axes[1]
ax.imshow(cropped_screenshot_all[1])
ax.axis('off')
ax.set_title('SQUID', fontsize=12, fontweight='normal')


ax = axes[2]

# ax = axes[1,1]
ax.axis('off')
cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
norm = mcolors.Normalize(vmin=0.6, vmax=1.0)
im = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = plt.colorbar(im, cax=cax, orientation='vertical')
cb.set_ticks([0, 0.6, 1])
cb.set_label('Normalized source power',labelpad=2)




plt.show()

fig.savefig(f'./fig/3-{paradigm}/source.png',dpi=900)









# %%
from scipy.stats import gaussian_kde, norm
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2.8, 2.5),constrained_layout=True,
                         gridspec_kw={'width_ratios': [1,0.1], 'height_ratios': [0.15,1]})
plt.subplots_adjust(top=0.93,bottom=0,left=0.005,right=0.995,wspace=0.2,hspace=0.05)

colors = ['#E9657F','#007ACC','#2ECC40','#F1C40F','#E74C3C','#00FFFF']

axes[0,0].axis('off')

axes[0,1].axis('off')

axes[1,1].axis('off')



t_ori = 0.14
t_ori_idx = np.argmin(np.abs(stcs_mean[0].times - t_ori))



ax = axes[1,0]
ax.scatter(stcs_snr_mean[1].data[:,t_ori_idx],stcs_snr_mean[0].data[:,t_ori_idx],s=4,color='black',alpha=0.1)
ax.plot([120,170],[120,170],color='r',linewidth=1,linestyle='--')

ax.vlines(np.mean(stcs_snr_mean[1].data[:,t_ori_idx]),120,170,color=colors[1],linewidth=1,linestyle='--')
ax.hlines(np.mean(stcs_snr_mean[0].data[:,t_ori_idx]),120,170,color=colors[0],linewidth=1,linestyle='--')

ax.set_xlim(120,170)
ax.set_ylim(120,170)
ax.set_xticks([120,130,140,150,160,170])
ax.set_yticks([120,130,140,150,160,170])
ax.set_xlabel('SQUID SNR')
ax.set_ylabel('OPM SNR')
ax.set_aspect('equal')


ax = fig.add_axes([0.225, 0.85, 0.56, 0.13])
data = stcs_snr_mean[1].data[:,t_ori_idx]

counts, bins, _ = ax.hist(data, bins=80, alpha=0.3, density=True,
                            color=colors[1],rwidth=1, edgecolor=None, linewidth=0.5)

x = np.linspace(data.min(), data.max(), 500)
mu, std = norm.fit(data)
pdf = norm.pdf(x, mu, std)
ax.plot(x, pdf, color=colors[1], lw=1, linestyle='-')
ax.set_xlim(120,170)
ax.set_yticks([])
ax.set_xticks([])

ax = fig.add_axes([0.805, 0.202, 0.13, 0.63])
data = stcs_snr_mean[0].data[:,t_ori_idx]
counts, bins, _ = ax.hist(data, bins=80, alpha=0.3, density=True,
                            color=colors[0],rwidth=1, edgecolor=None, linewidth=0.5,orientation='horizontal')

x = np.linspace(data.min(), data.max(), 500)
mu, std = norm.fit(data)
pdf = norm.pdf(x, mu, std)
ax.plot(pdf,x, color=colors[0], lw=1, linestyle='-')
ax.set_ylim(120,170)
ax.set_yticks([])
ax.set_xticks([])


# fig.text(0.428,0.93,f'mean: {np.mean(stcs_snr_mean[1].data[:,t_ori_idx]):.2f}',va='center',ha='center',color=colors[1])



plt.show()


fig.savefig(f'./fig/3-{paradigm}/source_snr.png',dpi=900)
fig.savefig(f'./fig/3-{paradigm}/source_snr.pdf')
fig.savefig(f'./fig/3-{paradigm}/source_snr.svg')
























# %%
subject = 'S4'
subjects_dir = './MEGDataset/0-MRI'

labels_name = ['rh.V2_exvivo.label','rh.V1_exvivo.label','lh.V2_exvivo.label','lh.V1_exvivo.label']
labels = [mne.read_label(f'{subjects_dir}/{subject}/label/{label_name}') for label_name in labels_name]
labels0 = labels[0]+labels[1]
labels1 = labels[2]+labels[3]

fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))
brain = mne.viz.Brain(subject=subject, subjects_dir=subjects_dir, hemi='both', surf='white',figure=fig,alpha=0.3)
brain.add_label(labels0, color='#E74C3C', alpha=0.7)
brain.add_label(labels1, color='#E74C3C', alpha=0.7)

fig.plotter.remove_all_lights()
mne.viz.set_3d_view(brain, azimuth=90, elevation=0,roll=-1,focalpoint='auto', distance=500)
screenshot=fig.plotter.screenshot(scale=10)

# %%
nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]


# %%
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3),constrained_layout=True,
                         gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})

ax.imshow(cropped_screenshot,cmap='gray')
ax.axis('off')
plt.show()

# fig.savefig(f'./fig/3-{paradigm}/source_label.png',dpi=900)


# %%
stcs_data_pick_all = dict()
for experiment in ['OPM','SQUID']:
    stcs_data_pick_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        stc = stcs_all[experiment][sub_idx]
        fwd = mne.read_forward_solution(f'./result/3-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')
        src = fwd['src']

        labels = mne.read_labels_from_annot(
            subject=sub_id,
            parc='aparc.a2009s',
            hemi='both',
            subjects_dir=subjects_dir
        )

        labels_name = ['rh.V2_exvivo.label','rh.V1_exvivo.label','lh.V2_exvivo.label','lh.V1_exvivo.label']
        labels = [mne.read_label(f'{subjects_dir}/{subject}/label/{label_name}') for label_name in labels_name]

        visual = labels[0]+labels[2]+labels[1]+labels[3]

        pick_data = stc.extract_label_time_course(visual,src,mode=None)[0]

        stcs_data_pick_experiment.append(pick_data)

    stcs_data_pick_all[experiment] = stcs_data_pick_experiment


# %%
stcs_data_pick_all_mean_opm = np.array([np.mean(stcs_data_pick_all['OPM'][i],axis=0) for i in range(6)])
times_opm = stcs_all['OPM'][0].times

stcs_data_pick_all_mean_squid = np.array([np.mean(stcs_data_pick_all['SQUID'][i],axis=0) for i in range(6)])
times_squid = stcs_all['SQUID'][0].times

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 13
plt.rcParams['font.weight'] = 'normal'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 2.5),constrained_layout=True,
                         gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})
colors = ['#007ACC','#E9657F','#2ECC40','#F1C40F','#E74C3C','#00FFFF']



ax.plot(times_opm+0.03,np.mean(stcs_data_pick_all_mean_opm,axis=0),color=colors[1],label='OPM')
ax.fill_between(times_opm+0.03,
                np.min(stcs_data_pick_all_mean_opm,axis=0),
                np.max(stcs_data_pick_all_mean_opm,axis=0),
                color=colors[1],alpha=0.2
                )


ax.plot(times_squid,np.mean(stcs_data_pick_all_mean_squid,axis=0),color=colors[0],label='SQUID')
ax.fill_between(times_squid,
                np.min(stcs_data_pick_all_mean_squid,axis=0),
                np.max(stcs_data_pick_all_mean_squid,axis=0),
                color=colors[0],alpha=0.2
                )

ax.vlines(0,0,18,color='black',linestyles='--',linewidth=1)

ax.set_ylim([0,15])
ax.set_yticks([0,3,6,9,12,15])
ax.set_xlim([-0.1,0.8])

ax.set_xlabel('Time (s)')
ax.set_ylabel('Source amplitude (a.u.)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

lg = ax.legend(loc='upper right',frameon=False,ncol=2,handlelength=1.2)


plt.show()

fig.savefig(f'./fig/3-{paradigm}/source_amplitude.png',dpi=900)
fig.savefig(f'./fig/3-{paradigm}/source_amplitude.pdf')
fig.savefig(f'./fig/3-{paradigm}/source_amplitude.svg')


# %%
