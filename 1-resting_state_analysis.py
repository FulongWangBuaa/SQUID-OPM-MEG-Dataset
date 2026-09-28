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
paradigm = 'resting_state'


# %%
raws_all = dict()
for experiment in ['OPM', 'SQUID']:
    raws_experiment = []
    for sub_id in sub_ids:
        data_path = f'./MEGDataset/2-Preprocessed data/{sub_id}/1-{paradigm}/{experiment}-MEG/{sub_id}_{paradigm}_{experiment.lower()}_preprocessed-raw.fif'

        raw = mne.io.read_raw_fif(data_path, preload=True)
        # 保留预处理阶段标记的坏道（如 OPM 所有被试的 P7），
        # 后续 PSD 计算/平均前会按 info['bads'] 自动剔除

        if experiment == 'SQUID':
            events = mne.find_events(raw, stim_channel='UPPT001')
        elif experiment == 'OPM' or experiment == 'OPMHZ':
            events = mne.find_events(raw, stim_channel='Trigger')

        if events.shape[0] != 4:
            raise ValueError(f'Wrong number of events for {sub_id}')

        sfreq = raw.info['sfreq']

        raw_open = raw.copy().crop(events[0,0]/sfreq, events[1,0]/sfreq).resample(100)
        raw_close = raw.copy().crop(events[2,0]/sfreq, events[3,0]/sfreq).resample(100)

        raws_experiment.append([raw_open, raw_close])

    raws_all[experiment] = raws_experiment








# %%
# ============================================================
# 计算并保存每个受试者睁眼/闭眼的传感器级功率谱密度(PSD)
# 保存结构: result/1-resting_state/psd/<实验>/<受试者>/<睁眼/闭眼>_psd.npz
#   psd_power : Welch 功率谱密度 (T^2/Hz, MNE 原始单位)
#   psd_amp   : 幅度谱密度 (fT/sqrt(Hz))，等价于原来的 sqrt(psd*1e30)
#   freqs     : 频率轴 (Hz)，所有被试/条件一致
#   ch_names  : 参与计算的 MEG 通道名
#   ch_types  : 与 ch_names 对应的通道类型（本数据均为 mag）
# 之后做跨受试者平均时只需读取这些 npz，无需重新计算 PSD。
# ============================================================
experiments = ['OPM', 'SQUID']
conditions = ['eyes_open', 'eyes_closed']   # 对应 raws_all[实验][被试] 的第 0/1 个 raw

psd_dir = f'./result/1-{paradigm}/psd'

# Welch 参数：所有被试统一，保证频率轴完全一致
# （数据已重采样到 100 Hz，n_fft=1024 时频率分辨率约 0.1 Hz，50% overlap）
psd_kwargs = dict(method='welch', fmin=0, fmax=50, n_fft=1024, n_overlap=512)

psd_all = dict()   # psd_all[experiment][sub_id][condition] = 文件路径
for experiment in experiments:
    psd_all[experiment] = dict()
    for sub_idx, sub_id in enumerate(sub_ids):
        psd_all[experiment][sub_id] = dict()
        for cond_idx, condition in enumerate(conditions):
            raw = raws_all[experiment][sub_idx][cond_idx]

            # 只保留非坏道 MEG 通道（OPM 数据会剔除预处理标记的 P7）
            meg_picks = mne.pick_types(raw.info, meg=True, exclude='bads')
            spectrum = raw.compute_psd(picks=meg_picks, **psd_kwargs)
            psd_power, freqs = spectrum.get_data(return_freqs=True)  # (n_good_meg, n_freqs)
            psd_amp = np.sqrt(psd_power) * 1e15                      # T/sqrt(Hz) -> fT/sqrt(Hz)
            # psd_amp = 10*np.log10(psd_amp)                         # dB(fT/sqrt(Hz))

            save_dir = os.path.join(psd_dir, experiment, sub_id)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{condition}_psd.npz')
            np.savez(save_path,
                     psd_power=psd_power,
                     psd_amp=psd_amp,
                     freqs=freqs,
                     ch_names=np.asarray(spectrum.ch_names),
                     ch_types=np.asarray(spectrum.get_channel_types()),
                     sfreq=raw.info['sfreq'])
            psd_all[experiment][sub_id][condition] = save_path

            print(f'[PSD] {experiment} | {sub_id} | {condition} | '
                  f'{psd_amp.shape[0]} ch x {psd_amp.shape[1]} freqs -> {save_path}')


# %%
# ============================================================
# 跨受试者平均：读取上面保存的逐被试 PSD（无需重算），
# 按 实验 × 睁眼闭眼 求跨被试平均/标准差。
# 同一实验内各受试者的 MEG 通道名和顺序完全一致，可直接对齐；
# 代码里仍做一次顺序校验，避免静默错位。
# 结果: psd_group[experiment][condition] =
#       {'mean': (n_meg, n_freq) 跨被试平均幅度谱,
#        'std' : (n_meg, n_freq) 跨被试标准差,
#        'n_sub': 被试数, 'freqs': 频率轴, 'ch_names': 通道名}
# ============================================================
psd_group = dict()
for experiment in experiments:
    psd_group[experiment] = dict()
    for condition in conditions:
        psd_stack = []
        ch_names_ref = None
        freqs_ref = None
        for sub_id in sub_ids:
            fname = os.path.join(psd_dir, experiment, sub_id, f'{condition}_psd.npz')
            with np.load(fname) as d:
                psd_amp = d['psd_amp']
                ch_names = d['ch_names'].tolist()
                freqs = d['freqs']

            if ch_names_ref is None:
                ch_names_ref = ch_names
                freqs_ref = freqs
            elif ch_names != ch_names_ref:
                # 若某被试通道顺序不同，先按基准被试的顺序重排
                order = [ch_names.index(ch) for ch in ch_names_ref]
                psd_amp = psd_amp[order]
                ch_names = [ch_names[i] for i in order]
                if ch_names != ch_names_ref:
                    raise ValueError(f'{experiment} {sub_id} 的 MEG 通道与基准被试不一致')
            if not np.array_equal(freqs, freqs_ref):
                raise ValueError(f'{experiment} {sub_id} 的频率轴与基准被试不一致，'
                                 f'请检查 PSD 参数(n_fft 等)是否统一')

            psd_stack.append(psd_amp)

        psd_stack = np.stack(psd_stack, axis=0)      # (n_sub, n_meg, n_freq)
        psd_group[experiment][condition] = dict(
            mean=psd_stack.mean(axis=0),
            std=psd_stack.std(axis=0, ddof=1),
            n_sub=psd_stack.shape[0],
            freqs=freqs_ref,
            ch_names=ch_names_ref,
        )
        print(f'[平均] {experiment} | {condition} | '
              f'mean {psd_group[experiment][condition]["mean"].shape} '
              f'(n_sub={psd_stack.shape[0]})')

# 可选：把跨被试平均结果保存下来，供画图/统计直接读取
group_path = os.path.join(psd_dir, 'psd_group_mean.pkl')
with open(group_path, 'wb') as f:
    pickle.dump(psd_group, f)
print(f'[保存] 跨被试平均结果 -> {group_path}')



# %%
# 绘制功率谱密度曲线
#
# 把 4 种情况画在一张图上：OPM/SQUID × 睁眼/闭眼
# 每个被试先对全头 MEG 通道取平均，再做跨被试平均，
# 阴影为跨被试 SEM（n_sub = 6）。
import matplotlib.pyplot as plt

experiments = ['OPM', 'SQUID']
conditions = ['eyes_open', 'eyes_closed']   # 对应 raws_all[实验][被试] 的第 0/1 个 raw

psd_dir = f'./result/1-{paradigm}/psd'

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'normal'

fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5), constrained_layout=True)

condition_label = {'eyes_open': 'EO', 'eyes_closed': 'EC'}
line_style = {'eyes_open': '-', 'eyes_closed': '--'}
line_color = {'OPM': 'tab:red', 'SQUID': 'tab:blue'}

for experiment in experiments:
    for condition in conditions:
        sub_mean = []
        for sub_id in sub_ids:
            fname = os.path.join(psd_dir, experiment, sub_id, f'{condition}_psd.npz')
            with np.load(fname) as d:
                freqs = d['freqs']
                psd_amp = d['psd_amp']            # (n_meg, n_freq), fT/sqrt(Hz)
            sub_mean.append(psd_amp.mean(axis=0)) # 每个被试先对全头通道平均
        sub_mean = np.stack(sub_mean, axis=0)     # (n_sub, n_freq)

        mean = sub_mean.mean(axis=0)
        sem = sub_mean.std(axis=0, ddof=1) / np.sqrt(sub_mean.shape[0])

        sel = freqs >= 1                          # 从 1 Hz 开始，避开 0 Hz 直流附近伪迹
        label = f'{experiment} {condition_label[condition]}'

        ax.plot(freqs[sel], mean[sel], color=line_color[experiment],
                ls=line_style[condition], lw=1.5, label=label)
        ax.fill_between(freqs[sel], mean[sel] - sem[sel], mean[sel] + sem[sel],
                        color=line_color[experiment], alpha=0.18)

# ax.grid(False)
ax.grid(
    which='major',
    linestyle='--',
    linewidth=0.6,
    alpha=0.5
)

# 次网格线
ax.grid(
    which='minor',
    linestyle='--',
    linewidth=0.4,
    alpha=0.5
)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['right'].set_linewidth(1.2)
ax.spines['top'].set_linewidth(1.2)
ax.tick_params(axis='both', width=1.2, length=4)


ax.set_xlim(1, 40)
ax.set_ylim(5,150)
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'PSD amplitude (fT/$\sqrt{\mathrm{Hz}}$)')
ax.legend(frameon=False, loc=(0.01,0),handlelength=1.3, handletextpad=0.4,ncol=2)


os.makedirs(f'./fig/1-{paradigm}', exist_ok=True)
# fig.savefig(f'./fig/1-{paradigm}/psd_4conditions.png', dpi=900)
# fig.savefig(f'./fig/1-{paradigm}/psd_4conditions.svg')
# fig.savefig(f'./fig/1-{paradigm}/psd_4conditions.pdf')
plt.show()







# %%
# 时频分析，并以睁眼数据作为基线展示相对变化
from my_code.tfr.wfl_tfr_multitaper import wfl_tfr_multitaper

# 多锥度时频参数（所有被试统一；raws_all 已重采样到 100 Hz）
tfr_params = dict(
    frequency_range=[1, 40],
    time_bandwidth=5,        # 频率平滑带宽约 2*TW/窗长 = 2 Hz
    num_tapers=9,
    window_params=[5, 1],    # 5 s 窗、1 s 步进
    detrend_opt='linear',
    multiprocess=True,
    n_jobs=None,
    weighting='unity',
)

# OPM 各被试静息 200 s，SQUID S4-S6 只有 150 s；
# 统一截取前 150 s，保证跨被试时频图的时间轴一致
tfr_duration = 150  # s

# 计算每个被试/条件的通道平均时频功率 (n_freq, n_time)
tfr_sub_mean = dict()   # [experiment][condition][sub_id]
tfr_times_ref = None
tfr_freqs_ref = None
for experiment in experiments:
    tfr_sub_mean[experiment] = dict()
    for cond_idx, condition in enumerate(conditions):
        tfr_sub_mean[experiment][condition] = dict()
        for sub_idx, sub_id in enumerate(sub_ids):
            raw = raws_all[experiment][sub_idx][cond_idx].copy()
            raw.crop(0, min(tfr_duration, raw.times[-1]))

            tfr_data, tfr_times, tfr_freqs, ch_names = wfl_tfr_multitaper(raw, tfr_params)
            if tfr_times_ref is None:
                tfr_times_ref = tfr_times
                tfr_freqs_ref = tfr_freqs
            else:
                if not np.array_equal(tfr_times, tfr_times_ref):
                    raise ValueError(f'{experiment} {sub_id} 的 TFR 时间轴不一致，'
                                     f'请检查截取时长是否统一')

            # 全头通道平均，只保留每个被试一条 (n_freq, n_time) 时频曲线
            tfr_sub_mean[experiment][condition][sub_id] = tfr_data.mean(axis=0)
            print(f'[TFR] {experiment} | {sub_id} | {condition} | '
                  f'{tfr_data.shape[0]} ch -> channel-mean '
                  f'{tfr_sub_mean[experiment][condition][sub_id].shape}')

tfr_times = tfr_times_ref
tfr_freqs = tfr_freqs_ref

# 每个被试：先对睁眼 TFR 按时间取平均，得到每个频率的基线
# baseline(f) = mean_t(睁眼 TFR)，睁眼/闭眼都除以这个基线
tfr_open_base = dict()   # [experiment][sub_id] -> (n_freq,)
for experiment in experiments:
    tfr_open_base[experiment] = dict()
    for sub_id in sub_ids:
        tfr_open_base[experiment][sub_id] = \
            tfr_sub_mean[experiment]['eyes_open'][sub_id].mean(axis=1)

# 逐被试计算相对变化 (TFR - baseline)/baseline * 100（百分比）
tfr_rel_sub = dict()   # [experiment][condition][sub_id] -> (n_freq, n_time)
for experiment in experiments:
    tfr_rel_sub[experiment] = dict()
    for condition in conditions:
        tfr_rel_sub[experiment][condition] = dict()
        for sub_id in sub_ids:
            baseline = tfr_open_base[experiment][sub_id][:, None]   # (n_freq, 1)
            tfr = tfr_sub_mean[experiment][condition][sub_id]
            tfr_rel_sub[experiment][condition][sub_id] = \
                (tfr - baseline) / baseline * 100

# 跨被试平均：tfr_rel_change[experiment][condition] -> (n_freq, n_time)
tfr_rel_change = dict()
for experiment in experiments:
    tfr_rel_change[experiment] = dict()
    for condition in conditions:
        rel_subs = np.stack(list(tfr_rel_sub[experiment][condition].values()))
        tfr_rel_change[experiment][condition] = rel_subs.mean(axis=0)

# 保存分组结果，之后画图/统计可直接读取
os.makedirs(f'./result/1-{paradigm}', exist_ok=True)
with open(f'./result/1-{paradigm}/tfr_group.pkl', 'wb') as f:
    pickle.dump({'sub_mean': tfr_sub_mean, 'open_baseline': tfr_open_base,
                 'rel_sub': tfr_rel_sub, 'rel_change': tfr_rel_change,
                 'times': tfr_times, 'freqs': tfr_freqs}, f)
print(f'[保存] TFR 结果 -> ./result/1-{paradigm}/tfr_group.pkl')


# %%
# 只画相对变化图：2 行(OPM/SQUID) × 2 列(睁眼/闭眼)
# 基准均为“睁眼 TFR 的时间平均（逐频率）”
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 11
fig, axes = plt.subplots(2, 2, figsize=(6, 5), constrained_layout=True)

# 四张图共用同一色标范围，方便互相比较
rel_all = np.concatenate([tfr_rel_change[exp][cond].ravel()
                          for exp in experiments for cond in conditions])

for row, experiment in enumerate(experiments):
    for col, condition in enumerate(conditions):
        ax = axes[row, col]
        rel = tfr_rel_change[experiment][condition]
        im = ax.pcolormesh(tfr_times, tfr_freqs, rel, cmap='RdBu_r',
                           vmin=-150, vmax=150, shading='auto')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_xticks([tfr_times[0], 30,60,90,120,tfr_times[-1]])
        ax.set_xticklabels([0,30,60,90,120,150])
        ax.set_yticks([tfr_freqs[0],10,20,30,40])
        ax.set_yticklabels([0,10,20,30,40])

        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['top'].set_linewidth(1.2)
        ax.tick_params(axis='both', width=1.2, length=4)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
cbar.set_label('Relative change vs eyes-open (%)')

os.makedirs(f'./fig/1-{paradigm}', exist_ok=True)
# fig.savefig(f'./fig/1-{paradigm}/tfr_relative_change.png', dpi=900)
# fig.savefig(f'./fig/1-{paradigm}/tfr_relative_change.pdf')
plt.show()



















# %%
subjects_dir = './MEGDataset/0-MRI'
trans = mne.transforms.Transform('head', 'mri')

stcs_all = dict()

for experiment in ['OPM','SQUID']:
    stcs_experiment = []
    for sub_idx, sub_id in enumerate(sub_ids):
        
        raw_open = raws_all[experiment][sub_idx][0]
        raw_close = raws_all[experiment][sub_idx][1]

        if not os.path.exists(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif'):
            src = mne.setup_source_space(sub_id, spacing='oct6', add_dist='patch',
                                subjects_dir=subjects_dir)
            src.save(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')
        else:
            src = mne.read_source_spaces(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-src.fif')

        if not os.path.exists(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif'):
            conductivity = (0.3,)
            model = mne.make_bem_model(subject=sub_id, ico=4,
                                    conductivity=conductivity,
                                    subjects_dir=subjects_dir)
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif',bem)
        else:
            bem = mne.read_bem_solution(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_ico4-bem.fif')

        if not os.path.exists(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif'):
            fwd = mne.make_forward_solution(raw_close.info, trans=trans, src=src, bem=bem,
                                            meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                            verbose=True)
            fwd.save(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')
        else:
            fwd = mne.read_forward_solution(f'./result/1-{paradigm}/source/{experiment}/{sub_id}_oct6_surface-fwd.fif')

        noise_cov = mne.compute_raw_covariance(raw_open)
        signal_cov = mne.compute_raw_covariance(raw_close)

        inverse_operator = mne.minimum_norm.make_inverse_operator(raw_open.info, forward=fwd, noise_cov=signal_cov, verbose=True)

        freq_bands = dict(theta=(4,8),alpha=(8, 12), beta=(13, 30),gama=(30, 40))
        snr = 3.0
        lambda2 = 1.0 / snr**2
        stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
                raw_close,
                inverse_operator,
                lambda2=lambda2,
                n_fft=1024,
                dB=False,
                return_sensor=True,
                verbose=True,
            )
        topo_norm = sensor_psd.data.sum(axis=1, keepdims=True)
        stc_norm = stc_psd.sum()  # same operation on MNE object, sum across freqs
        # Normalize each source point by the total power across freqs
        topos = dict()
        stcs = dict()

        for band, limits in freq_bands.items():
            data = sensor_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
            topos[band] = mne.EvokedArray(100 * data / topo_norm, sensor_psd.info)
            stcs[band] = stc_psd.copy().crop(*limits).sum() / stc_norm.data

        stcs_experiment.append(stcs)
    stcs_all[experiment] = stcs_experiment


# %%
if not os.path.exists(f'./result/1-{paradigm}/stcs_all.pkl'):
    with open(f'./result/1-{paradigm}/stcs_all.pkl','wb') as f:
        pickle.dump(stcs_all,f)
else:
    raise FileExistsError(f'./result/1-{paradigm}/stcs_all.pkl')

# %%
if os.path.exists(f'./result/1-{paradigm}/stcs_all.pkl'):
    with open(f'./result/1-{paradigm}/stcs_all.pkl','rb') as f:
        stcs_all = pickle.load(f)
else:
    raise FileNotFoundError(f'./result/1-{paradigm}/stcs_all.pkl')

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
        for event_idx, event_key in enumerate(['theta','alpha','beta','gama']):
            stc = stcs_all[experiment][sub_idx][event_key]

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
# 计算平均源空间
stcs_mean = []
for experiment in ['OPM','SQUID']:
    stcs_data_experiment = dict()
    for event_idx, event_key in enumerate(['theta','alpha','beta','gama']):
        stcs_data_rthythm = []
        for sub_idx, sub_id in enumerate(sub_ids):
            # 受试者源归一化，防止单个被试影响组平均结果
            stc = stcs_fsaverage[experiment][sub_idx][event_idx]
            stc_data = stc.data
            data_min = stc_data.min()
            data_max = stc_data.max()
            data_norm = (stc_data - data_min) / (data_max - data_min)
            stcs_data_rthythm.append(data_norm)

        stcs_data_rthythm_mean = np.mean(stcs_data_rthythm,axis=0) # 跨受试者平均
        data_min = stcs_data_rthythm_mean.min()
        data_max = stcs_data_rthythm_mean.max()
        data_norm = (stcs_data_rthythm_mean - data_min) / (data_max - data_min)
        stc_mean = stc.copy()
        stc_mean._data = data_norm
        stcs_data_experiment[event_key] = stc_mean
    stcs_mean.append(stcs_data_experiment)



# %%
subject = 'S4'

experiment = 'OPM'

clims = [[dict(kind="value", lims=(0,0,0.7)),
         dict(kind="value", lims=(0,0,0.5)),
         dict(kind="value", lims=(0,0,0.7)),
         dict(kind="value", lims=(0,0,0.7))],
        [dict(kind="value", lims=(0,0,0.5)),
         dict(kind="value", lims=(0,0,0.3)),
         dict(kind="value", lims=(0,0,1)),
         dict(kind="value", lims=(0,0,1))]]

screenshots = dict()
for i, experiment in enumerate(['OPM','SQUID']):
    screenshots[experiment] = []
    for j, rthythm in enumerate(['theta','alpha','beta','gama']):
        experiment_idx = ['OPM','SQUID'].index(experiment)
        # fig = mne.viz.create_3d_figure((600,400), bgcolor=(255, 255, 255))
        brain = stcs_mean[experiment_idx][rthythm].plot(
            hemi="split",
            subject=subject,
            subjects_dir=subjects_dir,
            surface='white',
            colormap="RdBu_r",
            time_viewer=False,
            show_traces=False,
            colorbar=False,
            size=(600,250),
            # clim=dict(kind="percent", lims=(0, 90, 99)),
            clim=clims[i][j],
            background='white',
            smoothing_steps=10,
            # figure=fig
        )
        fig = brain._renderer
        fig.plotter.remove_all_lights()
        # mne.viz.set_3d_view(brain, azimuth=-10, elevation=0,focalpoint='auto', distance=400)
        screenshot=fig.plotter.screenshot(scale=10)
        screenshots[experiment].append(screenshot)
        brain.close()

# %%
# 中间也裁剪
# cropped_screenshots = dict()
# for experiment in ['OPM','SQUID']:
#     cropped_screenshots[experiment] = []
#     for idx, rthythm in enumerate(['theta','alpha','beta','gama']):
#         screenshot = screenshots[experiment][idx]
#         nonwhite_pix = (screenshot != 255).any(-1)
#         nonwhite_row = nonwhite_pix.any(1)
#         nonwhite_col = nonwhite_pix.any(0)
#         cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
#         cropped_screenshots[experiment].append(cropped_screenshot)


# 只裁剪边缘白色
cropped_screenshots = dict()
for experiment in ['OPM', 'SQUID']:
    cropped_screenshots[experiment] = []
    for idx, rthythm in enumerate(['theta', 'alpha', 'beta', 'gama']):
        screenshot = screenshots[experiment][idx]
        white_pix = (screenshot >= 250).all(axis=-1)
        h, w = white_pix.shape
        top = 0
        while top < h and white_pix[top, :].all():
            top += 1
        bottom = h
        while bottom > top and white_pix[bottom - 1, :].all():
            bottom -= 1
        left = 0
        while left < w and white_pix[:, left].all():
            left += 1
        right = w
        while right > left and white_pix[:, right - 1].all():
            right -= 1
        cropped_screenshot = screenshot[top:bottom, left:right]
        cropped_screenshots[experiment].append(cropped_screenshot)




# %%
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 2.5),constrained_layout=True,
                         gridspec_kw={'width_ratios': [1], 'height_ratios': [1,1]})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.3, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )

jet = plt.get_cmap('jet')
jet_no_blue = truncate_colormap(jet, minval=0.5, maxval=1.0)

ax = axes[0]
screenshot = cropped_screenshots['OPM'][1]
ax.imshow(screenshot, cmap='RdBu_r')
ax.axis('off')

ax = axes[1]
screenshot = cropped_screenshots['SQUID'][1]
im = ax.imshow(screenshot, cmap='RdBu_r',vmin=0, vmax=0.4)
ax.axis('off')


cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.1, pad=0.02, orientation='vertical')
cbar.set_label('Normalized power')

plt.show()


fig.savefig(f'./fig/1-{paradigm}/source.png',dpi=900)


# %%
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 4),constrained_layout=True,
                         gridspec_kw={'width_ratios': [1,1,0.3], 'height_ratios': [1,1,1,1]})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.3, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )

jet = plt.get_cmap('jet')
jet_no_blue = truncate_colormap(jet, minval=0.5, maxval=1.0)

for experiment_idx, experiment in enumerate(['OPM','SQUID']):
    for rthythm_idx, rthythm in enumerate(['theta','alpha','beta','gama']):
        screenshot = cropped_screenshots[experiment][rthythm_idx]
        ax = axes[rthythm_idx, experiment_idx]
        ax.imshow(screenshot, cmap=cm.gray)
        ax.axis('off')

for i in range(4):
    ax = axes[i, 2]
    ax.axis('off')

cax = fig.add_axes([0.915, 0.3, 0.02, 0.4])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=jet_no_blue, norm=norm, orientation='vertical')
cbar.set_label('Normalized Power')
cbar.set_ticks([0,0.5,1])

plt.show()

# fig.savefig(f'./fig/{paradigm}/source.png',dpi=900)
