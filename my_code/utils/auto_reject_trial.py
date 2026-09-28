import numpy as np
import mne
def get_bad_trials(inst, thresh_val):
    """
    Detect bad trials using variance threshold.

    Parameters
    ----------
    inst : mne.Epochs or np.ndarray
        if np.ndarray, data shape = (n_epochs, n_channels, n_samples)
    thresh_val : float
        Threshold multiplier (e.g., 3, 3*std) 

    Returns
    -------
    bad_trials : np.ndarray
        1D array of bad trial indices
    """
    if isinstance(inst, mne.Epochs):
        data = inst.get_data(picks='data')
    elif isinstance(inst, np.ndarray):
        data = inst.copy()
    n_epochs, n_channels, n_samples = data.shape

    # 每个 trial 在时间维度上取标准差
    std_data_mat = np.std(data, axis=2)  # shape (n_epochs, n_channels)
    # 对每个通道去中心化
    std_mean = np.mean(std_data_mat, axis=0, keepdims=True)
    std_data_mat_cent = std_data_mat - std_mean
    # 每个通道 across trials 的标准差
    std_trials = np.std(std_data_mat_cent, axis=0, keepdims=True)
    # 阈值矩阵
    threestd_trials_mat = thresh_val * std_trials
    # 超过阈值的部分
    logic_mat = std_data_mat_cent > threestd_trials_mat
    # 统计每个 trial 超过阈值的通道数量
    bad_trials_vec = np.sum(logic_mat, axis=1)
    # 若某个 trial 超过1个通道异常，则认为是 bad
    bad_trials = np.where(bad_trials_vec > 1)[0]

    return bad_trials




