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
paradigm = '2back'

# %%
epochs_all = dict()
bads_all = []
for experiment in ['OPM','SQUID']:
    epochs_experiment = []
    bads_experiment = []

    for sub_id in sub_ids:
        data_path = f'./MEGDataset/2-Preprocessed data/{sub_id}/6-{paradigm}/{experiment}-MEG/{sub_id}_{paradigm}_{experiment.lower()}_preprocessed-raw.fif'
        if not os.path.exists(f'./result/6-{paradigm}/epochs/{experiment}_{sub_id}-epo.fif'):
            if experiment == 'SQUID':
                raw = mne.io.read_raw_fif(data_path, preload=True).filter(4,8)

                events = mne.find_events(raw, stim_channel='UPPT001',min_duration = 10/raw.info['sfreq'])

                jj = 0
                events_new_1 = []
                events_new_2 = []
                for e_i in range(len(events)):
                    if e_i == 0:
                        continue
                    elif e_i in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]:
                        events_new_1.append([events[e_i][0], events[e_i][1], 1])
                        jj += 1
                    elif e_i in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]:
                        events_new_2.append([events[e_i][0], events[e_i][1], 2])
                        jj += 1

                events_new_1 = np.array(events_new_1)
                events_new_2 = np.array(events_new_2)

            elif experiment == 'OPM':
                raw = mne.io.read_raw_fif(data_path, preload=True).filter(4,8)
                events = mne.find_events(raw, stim_channel='Trigger')[:41]

                jj = 0
                events_new_1 = []
                events_new_2 = []
                for e_i in range(len(events)):
                    if e_i == 0:
                        continue
                    elif e_i in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]:
                        events_new_1.append([events[e_i][0], events[e_i][1], 1])
                        jj += 1
                    elif e_i in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]:
                        events_new_2.append([events[e_i][0], events[e_i][1], 2])
                        jj += 1

                events_new_1 = np.array(events_new_1)
                events_new_2 = np.array(events_new_2)

            epochs = mne.Epochs(raw, events_new_1, tmin=0, tmax=60, preload=True, baseline=None, reject_by_annotation=True).resample(100)

            del raw

            epochs.save(f'./result/6-{paradigm}/epochs/{experiment}_{sub_id}-epo.fif')

        else:
            print(f'{experiment}_{sub_id}-epo.fif already exists')


# %%
def pseudoT(C,Ca,Cc,leadfield,mu):
    # Regularized inverse of C
    Cr = C + mu * np.max(np.linalg.svd(C, compute_uv=False)) * np.eye(C.shape[0])
    Cr_inv = np.linalg.inv(Cr)

    n_sources = leadfield.shape[2]
    pseudoT = np.zeros(n_sources)

    ws = []
    for i in range(n_sources):
        this_L = leadfield[:,:,i] # shape: (n_sensors, 3)
        # Weight matrix
        W_v = np.linalg.inv(this_L.T @ Cr_inv @ this_L) @ (this_L.T @ Cr_inv)
        # 把 Cr_inv 投影到该体素的三维方向子空间，得到 3×3 的矩阵。这个矩阵反映了在给定传感器协方差（或其逆）加权下，不同偶极子方向在传感器上的“功率”或能量分布。
        iPower_v = this_L.T @ Cr_inv @ this_L # shape: (3,3)
        v, d, _ = np.linalg.svd(iPower_v)
        
        # 具有最大波束形成器投影的方向被视为体素的源的方向
        id_min = np.argmin(d)
        lopt = this_L @ v[:, id_min] # shape: (n_sensors,)

        denom = lopt.T @ Cr_inv @ lopt
        w = (lopt.T @ Cr_inv / denom).T # shape: (n_sensors, 1)

        ws.append(w)

        numerator = w.T @ Ca @ w - w.T @ Cc @ w
        denominator = 0.5 * (w.T @ Ca @ w + w.T @ Cc @ w)
        pseudoT[i] = numerator / denominator

    return pseudoT,ws



# %%
import mne.transforms
subjects_dir = './MEGDataset/0-MRI'
trans = mne.transforms.Transform('head', 'mri')

stcs_all = dict()
for experiment in ['OPM','SQUID']:
    stcs_all_experiment = []
    for sub_idx,sub_id in enumerate(sub_ids):
        if not os.path.exists(f'./result/6-{paradigm}/stc/{experiment}_{sub_id}_pseudoT-stc.pkl'):

            epochs = mne.read_epochs(f'./result/6-{paradigm}/epochs/{experiment}_{sub_id}-epo.fif')

            if not os.path.exists(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif'):
                surface = subjects_dir + '/' + sub_id + '/' + "bem" + '/' + "inner_skull.surf"
                mri = subjects_dir + '/' + sub_id + '/' + "mri" + '/' + "T1.mgz"
                src = mne.setup_volume_source_space(
                    sub_id, pos=5,subjects_dir=subjects_dir,add_interpolator=True,mri=mri)
                src.save(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')
            else:
                src = mne.read_source_spaces(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')

            if not os.path.exists(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif'):
                conductivity = (0.3,)
                model = mne.make_bem_model(subject=sub_id, ico=4,conductivity=conductivity,subjects_dir=subjects_dir)
                bem = mne.make_bem_solution(model)
                mne.write_bem_solution(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif',bem)
            else:
                bem = mne.read_bem_solution(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif')

            if not os.path.exists(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif'):
                fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem,
                                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                                verbose=True)
                fwd.save(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')
            else:
                fwd = mne.read_forward_solution(f'./result/6-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')

            leadfield = fwd["sol"]["data"]

            chs_names = epochs.ch_names[:-1]
            bads = epochs.info['bads']
            goods = [chs_name for chs_name in chs_names if chs_name not in bads]
            goods_idx = [chs_names.index(chs_name) for chs_name in goods]
            leadfield = leadfield[goods_idx]

            n_channels, three_n_voxels = leadfield.shape
            n_voxels = three_n_voxels // 3

            # 先 reshape 再 transpose
            leadfield_reshaped = leadfield.reshape(n_channels, n_voxels, 3).transpose(0, 2, 1)

            cov = mne.compute_covariance(epochs, tmin=0, tmax=60)
            cov_on = mne.compute_covariance(epochs, tmin=0, tmax=40)
            cov_off = mne.compute_covariance(epochs, tmin=40, tmax=60)

            C = cov['data']
            Ca = cov_on['data']
            Cc = cov_off['data']
            T,ws = pseudoT(C,Ca,Cc,leadfield_reshaped,0.05)
            stc = mne.VolSourceEstimate(T[:,np.newaxis],[fwd['src'][0]['vertno']],0,0.01,sub_id)
            with open(f'./result/6-{paradigm}/stc/{experiment}_{sub_id}_pseudoT-stc.pkl', 'wb') as file:
                pickle.dump(stc, file)

            with open(f'./result/6-{paradigm}/stc/{experiment}_{sub_id}_ws.pkl', 'wb') as file:
                pickle.dump(ws, file)

        else:
            print(f'{experiment}_{sub_id}_pseudoT-stc.pkl already exists')



# %%
subjects_dir = './MEGDataset/0-MRI'
trans = mne.transforms.Transform('head', 'mri')

if os.path.exists(f'./result/6-{paradigm}/volume_source_space-src.fif'):
    src_to = mne.read_source_spaces(f'./result/6-{paradigm}/volume_source_space-src.fif')
else:
    src_to = mne.setup_volume_source_space(
        subject='S4',     # 或 'MNI152'
        subjects_dir=subjects_dir,
        pos=5.0,                  # 必须和 morph spacing 一致
        mri='T1.mgz',
        add_interpolator=True,
        verbose=True
    )
    src_to.save(f'./result/6-{paradigm}/volume_source_space-src.fif')

if os.path.exists(f'./result/6-{paradigm}/stcs_fsaverage.pkl'):
    stcs_fsaverage = pickle.load(open(f'./result/6-{paradigm}/stcs_fsaverage.pkl', 'rb'))
else:
    stcs_fsaverage = dict()
    stcs_fsaverage['OPM'] = []
    stcs_fsaverage['SQUID'] = []

for experiment in ['OPM','SQUID']:
    stc_fsaverage_experiment = stcs_fsaverage[experiment]

    nub_sub = len(sub_ids)
    nub_sub_exists = len(stc_fsaverage_experiment)
    sub_need_compute = list(sub_ids)[nub_sub_exists:]

    if len(sub_need_compute) == 0:
        print(f'{experiment}: {nub_sub_exists} / {nub_sub} subjects have been computed.')
        continue
    else:
        print(f'{experiment}: {nub_sub_exists} / {nub_sub} subjects have been computed. {len(sub_need_compute)} subjects need to be computed.')

    for sub_idx, sub_id in enumerate(sub_need_compute):

        stc_file = f'./result/6-{paradigm}/stc/{experiment}_{sub_id}_pseudoT-stc.pkl'
        with open(stc_file, 'rb') as file:
            stc = pickle.load(file)
        fwd_path = f'./result/6-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif'
        fwd = mne.read_forward_solution(fwd_path)
        src = fwd['src']
        morph = mne.compute_source_morph(
            src=src,
            subject_from=sub_id,
            subject_to='S4',  # 或 'MNI152'
            subjects_dir=subjects_dir,
            spacing=5,               # MNI 体素间距（mm）
            src_to=src_to,
            verbose=True
        )

        stc_fsaverage = morph.apply(stc)

        stc_fsaverage_experiment.append(stc_fsaverage)

    stcs_fsaverage[experiment] = stc_fsaverage_experiment


# %%
stcs_fsaverage_path = f'./result/6-{paradigm}/stcs_fsaverage.pkl'
if not os.path.exists(stcs_fsaverage_path):
    with open(stcs_fsaverage_path, 'wb') as file:
        pickle.dump(stcs_fsaverage, file)
else:
    raise FileExistsError(f'{stcs_fsaverage_path} already exists')



# %%
stcs_fsaverage_path = f'./result/6-{paradigm}/stcs_fsaverage.pkl'
if os.path.exists(stcs_fsaverage_path):
    with open(stcs_fsaverage_path, 'rb') as file:
        stcs_fsaverage = pickle.load(file)
else:
    raise FileNotFoundError(f'{stcs_fsaverage_path} not found')

src_to = mne.read_source_spaces(f'./result/6-{paradigm}/volume_source_space-src.fif')


# %%
# 计算平均源空间
stcs_mean_all = []
for experiment in ['OPM','SQUID']:
    stcs_mean_experiment = []

    for sub_idx, sub_id in enumerate(sub_ids):
        stc = stcs_fsaverage[experiment][sub_idx]
        stc_data = stc.data
        # 归一化
        data_min = stc_data.min()
        data_max = stc_data.max()
        data_norm = (stc_data - data_min) / (data_max - data_min)
        stcs_mean_experiment.append(stc_data)

    stcs_data_event = np.array(stcs_mean_experiment) # subjects x sources x times
    # 跨受试者平均
    stcs_data_event_avg = np.mean(stcs_data_event, axis=0)  # sources × times
    # 归一化
    data_min = stcs_data_event_avg.min()
    data_max = stcs_data_event_avg.max()
    data_norm = (stcs_data_event_avg - data_min) / (data_max - data_min)

    stc_mean = stc.copy()
    stc_mean._data = stcs_data_event_avg
    
    stcs_mean_all.append(stc_mean)







# %%
from matplotlib.colors import LinearSegmentedColormap
from mne.viz._3d import _load_subject_mri
from nilearn.plotting import plot_stat_map,plot_img
from nilearn.image import index_img
from nilearn import image, plotting


colors = ["#ff0b00", "#fff801"]
custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))
plt.subplots_adjust(top=0.97,bottom=0.02,left=0,right=0.995,wspace=0.1,hspace=0.1)
thresholds = [0.12,0.08]
vmaxs = [0.15,0.11]
for i,ax in enumerate(axes):
    stc_mean = stcs_mean_all[i]
    img = stc_mean.as_volume(src_to, mri_resolution=False)

    bg_img = "T1.mgz"
    subject = 'S4'
    bg_img = _load_subject_mri(bg_img, stc_mean, subject, subjects_dir, "bg_img")

    display = plot_stat_map(
        img,
        bg_img=bg_img,
        # display_mode='ortho', # 'ortho' or 'x', 'y', 'z'
        # cut_coords=[10,50,0],
        display_mode='x',
        cut_coords= [10],
        threshold=thresholds[i],
        vmin=thresholds[i],
        vmax=vmaxs[i],
        cmap=custom_cmap,
        dim=0,
        axes=axes[i],
        draw_cross=False,
        annotate=False,
        colorbar=True,
        black_bg=True,
    )
    cbar = display._cbar
    cbar.set_label('Pseudo-T statictic', fontsize=10,color='white',labelpad=-20)
    cbar.set_ticks([thresholds[i], vmaxs[i]])
    cbar.ax.tick_params(labelsize=12)

fig.text(0.005,0.9,'OPM', fontsize=12,color='white')
fig.text(0.525,0.9,'SQUID', fontsize=12,color='white')

plotting.show()


fig.savefig(f'./fig/6-{paradigm}/stc_pseudoT.png', dpi=900)


# %%
# 获取伪T统计量峰值点对应的源时间序列
source_data_all = dict()
for experiment in ['OPM','SQUID']:
    source_data_all[experiment] = []
    for sub_idx, sub_id in enumerate(sub_ids):
        epochs_path = f'./result/6-{paradigm}/epochs/{experiment}_{sub_id}-epo.fif'
        epochs = mne.read_epochs(epochs_path)

        ws_path= f'./result/6-{paradigm}/stc/{experiment}_{sub_id}_ws.pkl'
        with open(ws_path, 'rb') as file:
            ws = pickle.load(file)

        stc_path= f'./result/6-{paradigm}/stc/{experiment}_{sub_id}_pseudoT-stc.pkl'
        with open(stc_path, 'rb') as file:
            stc = pickle.load(file)

        data = stc.data[:, 0]
        src_peak_idx = np.argmax(data)

        w = ws[src_peak_idx]

        chs_names = epochs.ch_names[:-1]
        bads = epochs.info['bads']
        goods = [chs_name for chs_name in chs_names if chs_name not in bads]
        goods_idx = [chs_names.index(chs_name) for chs_name in goods]
        w = w[goods_idx]

        epochs_data = epochs.get_data(picks='data')
        num_trials,num_chs,num_times = epochs_data.shape

        source_data = []
        for e in range(num_trials):
            epochs_data_trial = epochs_data[e,:,:]
            source_data.append(w @ epochs_data_trial / np.sqrt(w.T @ w))
        
        source_data = np.array(source_data) # trials x times
        source_data_all[experiment].append(source_data)

# %%
from scipy.signal import hilbert

sfreq = 100
times = np.arange(0, 60, 1.0 / sfreq)
baseline_win = (40, 60)
task_win = (10, 40)

envelope_subject_all = dict()
SNR_all = dict()
for experiment in ['OPM', 'SQUID']:
    envelope_subject_all[experiment] = []
    SNR_all[experiment] = []

    for source_data in source_data_all[experiment]:
        # Hilbert envelope（trial-wise）
        analytic = hilbert(source_data, axis=1)
        envelope = np.abs(analytic)  # (n_trials, n_times)
        # trial average
        envelope_trial_avg = envelope.mean(axis=0)  # (n_times,)

        # baseline
        baseline_mask = (times >= baseline_win[0]) & (times <= baseline_win[1])
        baseline_amp = envelope_trial_avg[baseline_mask].mean()


        # task
        task_mask = (times >= task_win[0]) & (times <= task_win[1])
        task_amp = envelope_trial_avg[task_mask].mean()

        SNR = (task_amp - baseline_amp) / np.std(envelope_trial_avg[baseline_mask])
        SNR_all[experiment].append(SNR)

        # relative change
        envelope_rel = (envelope_trial_avg - baseline_amp) / baseline_amp
        envelope_subject_all[experiment].append(envelope_rel)
        

# 对包络跨受试者平均
group_envelope = dict()
for experiment in envelope_subject_all:
    group_envelope[experiment] = dict()
    group_envelope[experiment]['mean'] = np.mean(envelope_subject_all[experiment],axis=0)
    group_envelope[experiment]['se'] = np.std(envelope_subject_all[experiment],axis=0) / np.sqrt(len(envelope_subject_all[experiment]))


# %%
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2),constrained_layout=True)
colors = ['#E9657F','#007ACC','#F1C40F','#2ECC40','#E74C3C','#00FFFF']
ax.plot(times, group_envelope['OPM']['mean'],color=colors[0],linewidth=0.8,label='OPM')
ax.fill_between(times, group_envelope['OPM']['mean']-group_envelope['OPM']['se'], 
                group_envelope['OPM']['mean']+group_envelope['OPM']['se'],
                color=colors[0],
                alpha=0.1)

ax.plot(times, group_envelope['SQUID']['mean'],color=colors[1],linewidth=0.8,label='SQUID')
ax.fill_between(times, group_envelope['SQUID']['mean']-group_envelope['SQUID']['se'],
                group_envelope['SQUID']['mean']+group_envelope['SQUID']['se'],
                color=colors[1],
                alpha=0.1)

ax.vlines(40, ymin=-0.3, ymax=0.4, color='k', linestyle='--', linewidth=0.8)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Relative theta amplitude')
ax.set_xlim([0, 60])
ax.set_xticks([0,10,20,30,40,50,60])
ax.set_ylim(-0.3,0.4)

lg = ax.legend(loc=(0.02,0.02), frameon=False, fontsize=12,
               handlelength=1,
               handletextpad=0.4,
               handleheight=0.5,
               labelspacing=0.5,
               columnspacing=1.2,
               ncol=3)
for h in lg.get_lines():
    h.set_linewidth(1.2)

plt.show()

# print(pearsonr(group_envelope['OPM']['mean'], group_envelope['SQUID']['mean']))

# fig.savefig(f'./fig/6-{paradigm}/hilbert_envelope.svg')







# %%
# 0-40 s 与 40-60 s 源时间序列的功率谱密度
# 每个 trial 单独算 PSD -> 跨 trial 平均得到单个受试者的 PSD 曲线 -> 再跨受试者平均
# 注意：这里跨 trial 平均是在功率谱上做的（保留 induced 成分）；
#       若想用"先跨 trial 平均的波形"求 PSD，可把下面 seg 换成 source_data_trial_avg_all
from scipy.signal import welch

psd_sfreq = 100.0            # epochs 已重采样到 100 Hz
nperseg = 1000               # 10 s 窗 -> 0.1 Hz 频率分辨率
noverlap = 500               # 50% 重叠
psd_windows = {'0-40s': (0, 40), '40-60s': (40, 60)}

psd_subject = dict()         # [experiment][window] = 每个受试者的 PSD 曲线列表
psd_freqs = None
for experiment in ['OPM', 'SQUID']:
    psd_subject[experiment] = {name: [] for name in psd_windows}
    for source_data in source_data_all[experiment]:
        # source_data: (n_trials, n_times)，时间 0-60 s，采样率 100 Hz
        for name, (t0, t1) in psd_windows.items():
            seg = source_data[:, int(t0 * psd_sfreq):int(t1 * psd_sfreq)]
            freqs, psd = welch(seg, fs=psd_sfreq, nperseg=nperseg,
                               noverlap=noverlap, axis=-1)      # (n_trials, n_freqs)
            psd_freqs = freqs
            # 跨 trial 平均（功率平均），再转成幅度谱：T -> fT
            psd_subject[experiment][name].append(np.sqrt(psd.mean(axis=0) * 1e30))

# 跨受试者平均 ± SEM
psd_group = dict()
for experiment in ['OPM', 'SQUID']:
    psd_group[experiment] = dict()
    for name in psd_windows:
        arr = np.array(psd_subject[experiment][name])           # (n_subjects, n_freqs)
        psd_group[experiment][name] = dict(
            mean=arr.mean(axis=0),
            sem=arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]),
            n_sub=arr.shape[0],
        )
        print(f'{experiment} {name}: {arr.shape[0]} subjects, {arr.shape[1]} freqs')

with open(f'./result/6-{paradigm}/source_psd.pkl', 'wb') as f:
    pickle.dump({'subject': psd_subject, 'group': psd_group, 'freqs': psd_freqs,
                 'windows': psd_windows, 'nperseg': nperseg, 'noverlap': noverlap}, f)
print(f'[保存] ./result/6-{paradigm}/source_psd.pkl')


# %%
# 画图：OPM、SQUID 各一张（0-40 s vs 40-60 s，mean ± SEM）
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
colors = {'0-40s': '#E9657F', '40-60s': '#007ACC'}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 4), constrained_layout=True)
os.makedirs(f'./fig/6-{paradigm}', exist_ok=True)
labels = ['ON','OFF']
for i, experiment in enumerate(['OPM', 'SQUID']):
    ax = axes[i]
    for i, name in enumerate(psd_windows):
        mean = psd_group[experiment][name]['mean']
        sem = psd_group[experiment][name]['sem']
        ax.plot(psd_freqs, mean, color=colors[name], linewidth=1.0, label=f'{labels[i]} mean')
        ax.fill_between(psd_freqs, mean - sem, mean + sem, color=colors[name], alpha=0.15,label=f'{labels[i]} se')

    ax.set_xlim(4, 8)          # 源时间序列是 4-8 Hz 带通后的数据
    ax.set_xticks([4,5,6,7,8])
    ax.set_ylim(20,90)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'PSD (fT/$\sqrt{\mathrm{Hz}}$)')
    # ax.set_title(experiment)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc=(0.05,0.72), frameon=False,handlelength=1,columnspacing=2)
    
plt.show()

fig.savefig(f'./fig/6-{paradigm}/source_psd.png', dpi=900)
fig.savefig(f'./fig/6-{paradigm}/source_psd.svg')
