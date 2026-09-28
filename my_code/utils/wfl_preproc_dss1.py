import numpy as np
import scipy
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import mne
from mne.io.pick import _picks_to_idx

def nt_pcarot(cov,nkeep=None,threshold=None):
    S, V = np.linalg.eigh(cov)
    V = V.real
    S = S.real
    idx = np.argsort(S)[::-1] # reverse sort ev order
    eigenvalues = S[idx]
    topcs = V[:, idx]

    if threshold is not None:
        ii = np.where(eigenvalues/eigenvalues[0]>threshold)[0]
        topcs = topcs[:,ii]
        eigenvalues = eigenvalues[ii]
    if nkeep is not None:
        nkeep=min(nkeep,topcs.shape[1])
        topcs = topcs[:,0:nkeep]
        eigenvalues = eigenvalues[0:nkeep]

    return topcs,eigenvalues


def nt_dss0(c0,c1,keep1=None,keep2=None):
    if keep2 is None:
        keep2 = 1e-9
    if c1 is None:
        raise ValueError('needs at least two arguments')
    
    if c0.shape != c1.shape:
        raise ValueError("C0 and C1 should have the same size")
    if c0.shape[0] != c0.shape[1]:
        raise ValueError("C0 should be square")
    
    if np.isnan(c0).any():
        raise ValueError('NaN in c0')
    if np.isnan(c1).any():
        raise ValueError('NaN in c1')
    if np.isinf(c0).any():
        raise ValueError('INF in c0')
    if np.isinf(c1).any():
        raise ValueError('INF in c1')
    
    topcs1,evs1=nt_pcarot(c0,keep1,keep2)
    evs1=abs(evs1)

    if keep1 is not None:
        topcs1 = topcs1[:, :keep1]
        evs1 = evs1[:keep1]
    if keep2 is not None:
        idx = np.where(evs1 / max(evs1) > keep2)[0]
        topcs1 = topcs1[:, idx]
        evs1 = evs1[idx]

    # PCA and whitening matrix from the unbiased covariance
    N = np.diag(1 / np.sqrt(evs1))

    c2 = N.T @ topcs1.T @ c1 @ topcs1 @ N

    # matrix to convert PCA-whitened data to DSS
    topcs2,evs2=nt_pcarot(c2,keep1,keep2)

    # DSS matrix (raw data to normalized DSS)
    todss = topcs1 @ N @ topcs2
    N2 = np.diag(todss.T @ c0 @ todss)
    todss = todss @ np.diag(1 / np.sqrt(N2))  # adjust so that components are normalized

    # power per DSS component
    pwr0 = np.sqrt(np.sum((c0.T @ todss) ** 2, axis=0))  # unbiased
    pwr1 = np.sqrt(np.sum((c1.T @ todss) ** 2, axis=0))  # biased

    return todss,pwr0,pwr1

def nt_cov(x,w=None):
    m,n,o = x.shape # samples * channels * trials
    if w is None:
        c = np.zeros((n,n))
        for k in range(o):
            xx = x[:,:,k]
            c = c + np.dot(xx.T, xx)
        tw = xx.shape[0]*o
    else:
        c = np.zeros((n,n))
        for k in range(o):
            xx = x[:,:,k] * w
            c = c + np.dot(xx.T, xx)
        tw=np.sum(w)
    return c,tw

def on_press(event):
    if event.button == 1:
        return event.xdata, event.ydata

click_x = None
click_y = None
def onclick(event):
    global click_x, click_y
    if event.button == 1:  # 点击鼠标左键
        click_x = event.xdata
        click_y = event.ydata
        # print(f"鼠标点击位置：({click_x}, {click_y})")

def get_click_position(y,title,linestyle='-',marker='o'):
    global click_x, click_y
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'bold'
    fig, ax = plt.subplots()
    ax.plot(y,linestyle=linestyle,marker=marker)
    ax.set_title(title)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    while click_x is None or click_y is None:
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid)
    x = click_x
    click_x = None
    click_y = None
    plt.close(fig)
    return int(np.floor(x)+1)


def nt_dss1(epochs,nremove='interactive',
            w=None,keep1=None,keep2=None,
            picks=None,return_todss = False):
    if isinstance(epochs,mne.Epochs):
        picks = _picks_to_idx(epochs.info, picks=picks, exclude=())
        picks_good, picks_bad = list(), list()  # these are indices into picks
        for ii, pi in enumerate(picks):
            if epochs.ch_names[pi] in epochs.info["bads"]:
                picks_bad.append(ii)
            else:
                picks_good.append(ii)
        picks_good = np.array(picks_good, int)
        picks_bad = np.array(picks_bad, int)

        x = epochs._data[:,picks_good,:]
    elif isinstance(epochs,(mne.io.Raw,mne.io.RawArray)):
       
        picks = _picks_to_idx(epochs.info, picks=picks, exclude=())
        picks_good, picks_bad = list(), list()  # these are indices into picks
        for ii, pi in enumerate(picks):
            if epochs.ch_names[pi] in epochs.info["bads"]:
                picks_bad.append(ii)
            else:
                picks_good.append(ii)
        picks_good = np.array(picks_good, int)
        picks_bad = np.array(picks_bad, int)

    if keep2 is None:
        keep2 = 1e-12
    
    if x.ndim != 3:
        raise ValueError("x should be a 3D array")

    # x: trials * channels * samples
    x = np.transpose(x, (2, 1, 0)) # samples * channels * trials
    m,n,o = x.shape

    if w is None:
        c0,nc0 = nt_cov(x)
        c0=c0/nc0

        c1,nc1 = nt_cov(np.mean(x,axis=2,keepdims=True))
        c1=c1/nc1

    todss,pwr0,pwr1=nt_dss0(c0,c1,keep1,keep2)
    # nremove ='interactive'
    if nremove is None:
        ttext = 'enter the spatial dimension for the inside field: '
        nremove = int(input(ttext))
    elif isinstance(nremove, str) and nremove == 'interactive':
        title = 'nremove'
        nremove = get_click_position(pwr1/pwr0,title)
    print('Using %d components' % nremove)

    fromdss = np.linalg.pinv(todss)
    y = np.zeros((m,n,o))
    for i in range(o):
        xx = x[:,:,i]
        y[:,:,i] = xx @ todss[:,0:nremove] @ fromdss[0:nremove,:]
    y = np.transpose(y, (2, 1, 0))
    epochs_dss1 = epochs.copy()
    epochs_dss1._data[:,picks_good,:] = y

    if return_todss:
        return epochs_dss1,todss
    else:
        return epochs_dss1

        
    