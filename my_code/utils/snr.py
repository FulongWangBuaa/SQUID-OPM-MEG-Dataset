import numpy as np
import mne

def SNRLB(inst,times=None,pre_time=None,post_time=None, percent_ci=0.9, num_bootstraps=9999, return_all_snr=False):
    """
    Computes confidence intervals for ERP signal-to-noise ratio (SNR) using bootstrap resampling.

    Parameters:
    - segment_dat: np.ndarray, 1D array of concatenated EEG segments
    - segment_points: tuple, (pre_stim, post_stim) in sample points
    - percent_ci: float, desired confidence interval (default 0.9)
    - num_bootstraps: int, number of bootstrap iterations (default 9999)

    Returns:
    - snr_lb: lower bound of SNR (dB)
    - snr_ub: upper bound of SNR (dB)
    - snr_mean: mean SNR (dB)
    """
    
    from tqdm import tqdm
    # Set alpha
    ci_alpha = (1.0 - percent_ci) / 2.0

    if isinstance(inst, mne.Epochs):
        data = inst.get_data(picks='data')*1e15
        times = inst.times
    elif isinstance(inst, np.ndarray):
        if times is None:
            raise ValueError('times must be provided if inst is an array')
        data = inst*1e15

    # SNR window
    zero_idx = np.argmin(np.abs(times - 0))
    pre_idx = np.argmin(np.abs(times - pre_time))
    post_idx = np.argmin(np.abs(times - post_time))


    data_segment = data[:, :, pre_idx:post_idx]
    n_trials,n_chs,n_samples = data_segment.shape

    # 虽然s_value=n_trials,但是实际上每次bootstrap只取了s_value个trial,有的trial可能被重复取到
    s_value = n_trials
 
    snrs_mean = []
    snrs_lb = []
    snrs_ub = []
    snrs_dist_sorted = []
    for ch_idx in tqdm(range(n_chs)):
        # Bootstrap resampling
        rect_boot_erps = np.zeros((num_bootstraps, n_samples))
        data_segment_todo = data_segment[:, ch_idx, :]
        for i in range(num_bootstraps):
            samples = data_segment_todo[np.random.randint(0, n_trials, s_value), :]
            erp = np.mean(samples, axis=0)
            erp -= np.mean(erp[:zero_idx])  # baseline correction
            rect_boot_erps[i, :] = np.sqrt(erp ** 2)  # rectification

        # SNR calculation
        snr_dist = 20 * np.log10(
            np.mean(rect_boot_erps[:, zero_idx:], axis=1) /
            np.mean(rect_boot_erps[:, :zero_idx], axis=1)
        )

        snr_dist_sorted = np.sort(snr_dist)
        snr_lb = snr_dist_sorted[int(round(ci_alpha * num_bootstraps))]
        snr_ub = snr_dist_sorted[int(round((1.0 - ci_alpha) * num_bootstraps))]
        snr_mean = np.mean(snr_dist_sorted)

        snrs_mean.append(snr_mean)
        snrs_lb.append(snr_lb)
        snrs_ub.append(snr_ub)
        snrs_dist_sorted.append(snr_dist_sorted)
    if return_all_snr:
        return snrs_lb, snrs_ub, snrs_mean, snrs_dist_sorted
    else:
        return snrs_lb, snrs_ub, snrs_mean