import mne
import numpy as np
import matplotlib.pyplot as plt
# from mne.io.pick import channel_indices_by_type,_DATA_CH_TYPES_SPLIT
import copy
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from mne.viz.backends.renderer import _get_renderer
from mne.viz._3d import _plot_head_surface
from mne.transforms import apply_trans,_get_trans,_get_transforms_to_coord_frame,_frame_to_str
from mne._fiff.pick import pick_types
from mne.defaults import DEFAULTS
from mne._fiff.constants import FIFF
from mne.surface import get_meg_helmet_surf,_get_head_surface
from mne.transforms import transform_surface_to
from mne.bem import ConductorModel, _bem_find_surface, _ensure_bem_surfaces

def wfl_plot_alignment(info=None, trans=None, subject=None, subjects_dir=None, 
                       surfaces=['head-dense'], coord_frame='mri', meg=None, 
                       eeg='original', fwd=None, dig=False, ecog=True, 
                       src=None, mri_fiducials=False, bem=None, seeg=False, 
                       fnirs=False, show_axes=False, dbs=False, fig=None, 
                       interaction='terrain', sensor_colors=None,
                       helmet_alpha=0.5,helmet_color=None,
                       head_surface=[],head_alpha=0.3,head_color=None,
                       ch_pos=True,ch_ori=False,ch_names=False,
                       pos_color='r',ori_color='r',ch_color='w',
                       pos_mode = 'cylinder',ori_mode = 'arrow',
                       ori_scale=0.03,pos_scale=0.003,ch_scale=0.005,
                       volume_src=False, volume_color=(1.0, 1.0, 0.0),volume_scale=0.003,
                       verbose=None):
    
    if "helmet" in meg:
        meg1 = copy.deepcopy(meg)
        meg1.remove("helmet")
    else:
        meg1 = copy.deepcopy(meg)
    if trans is None:
        trans = mne.transforms.Transform('head', 'mri')
    fig = mne.viz.plot_alignment(
        info=info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces=surfaces,
        coord_frame=coord_frame,
        meg=meg1,
        eeg=eeg,
        fwd=fwd,
        dig=dig,
        ecog=ecog,
        src=src,
        mri_fiducials=mri_fiducials,
        bem=bem,
        seeg=seeg,
        fnirs=fnirs,
        show_axes=show_axes,
        dbs=dbs,
        fig=fig,
        interaction=interaction,
        sensor_colors=sensor_colors,
        verbose=verbose,
    )
    
    renderer = _get_renderer(
            fig,
            name=f"Sensor alignment: {subject}",
            bgcolor=(0.5, 0.5, 0.5),
            size=(800, 800),
        )
    head_mri_t = _get_trans(trans, "head", "mri")[0]
    to_cf_t = _get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)

    # Head surface:
    head_keys = ("auto", "head", "outer_skin", "head-dense", "seghead")
    head = [s for s in head_surface if s in head_keys]
    if len(head) > 1:
        raise ValueError("Can only supply one head-like surface name, " f"got {head}")
    head = head[0] if head else False
    if head is not False:
        head_surface.pop(head_surface.index(head))
    elif "projected" in eeg:
        raise ValueError(
            "A head surface is required to project EEG, "
            '"head", "outer_skin", "head-dense" or "seghead" '
            'must be in surfaces or surfaces must be "auto"'
        )
    bem = _ensure_bem_surfaces(bem, extra_allow=(ConductorModel, None))
    assert isinstance(bem, ConductorModel) or bem is None
    _, _, head_surf = _plot_head_surface(
        renderer,
        head,
        subject,
        subjects_dir,
        bem,
        coord_frame,
        to_cf_t,
        alpha=head_alpha,
        color=head_color,
    )

    if "helmet" in meg and pick_types(info, meg=True).size > 0:
        _, _, src_surf = _plot_helmet(
            renderer,
            info,
            to_cf_t,
            head_mri_t,
            coord_frame,
            color=helmet_color,
            alpha=helmet_alpha,
        )

    if volume_src:
        for ss in src:
            src_rr = ss["rr"][ss["inuse"].astype(bool)]
            src_nn = ss["nn"][ss["inuse"].astype(bool)]

            # update coordinate frame
            src_trans = to_cf_t[_frame_to_str[src[0]["coord_frame"]]]
            src_rr = apply_trans(src_trans, src_rr)
            src_nn = apply_trans(src_trans, src_nn, move=False)

            if len(src_rr) > 0:
                renderer.quiver3d(
                    x=src_rr[:, 0],
                    y=src_rr[:, 1],
                    z=src_rr[:, 2],
                    u=src_nn[:, 0],
                    v=src_nn[:, 1],
                    w=src_nn[:, 2],
                    color=volume_color,
                    mode=pos_mode,
                    scale=volume_scale,
                    opacity=0.75,
                    glyph_height=0.25,
                    glyph_center=(0.0, 0.0, 0.0),
                    glyph_resolution=20,
                    backface_culling=True,
                )
    # 获取pos
    # ch_indices = channel_indices_by_type(info)
    # picks = list()
    # allowed_types = _DATA_CH_TYPES_SPLIT
    # for this_type in allowed_types:
    #     picks += ch_indices[this_type]

    if coord_frame == "auto":
        coord_frame = "head" if trans is None else "mri"
    
    chs = info["chs"]
    pos = []
    for ci, ch in enumerate(chs):
        pos1 = ch["loc"][:3]
        if ch["coord_frame"] != mne.io.constants.FIFF.FIFFV_COORD_UNKNOWN:
            pos.append(apply_trans(to_cf_t['meg'], pos1))
    pos = np.array(pos)
    
    
    if ch_ori:
        # 绘制探头方向 
        ori = []
        for ch in chs:
            if ch["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH:
                ori.append(ch["loc"][9:12])
        ori = np.array(ori)

        for x,y,z,u,v,w in zip(pos[:,0], pos[:,1], pos[:,2],ori[:,0],ori[:,1],ori[:,2]):
            # mode: 'arrow', 'cone', 'cylinder', 'oct', 'sphere'
            renderer.quiver3d(
                    x=x,
                    y=y,
                    z=z,
                    u=u,
                    v=v,
                    w=w,
                    mode=ori_mode,
                    scale=ori_scale,
                    color=ori_color,
                    scale_mode="scalar",
                    resolution=20,
                )
    if ch_pos:
        # 绘制探头位置
        for x,y,z in zip(pos[:,0], pos[:,1], pos[:,2]):
            renderer.sphere(center=np.array([x,y,z]), scale=pos_scale, color=pos_color)
    # renderer.sphere(center=np.array([0,0,0]), scale=0.01, color='r')
    if ch_names is True:
        for x,y,z,text in zip(pos[:,0], pos[:,1], pos[:,2],info['ch_names']):
            renderer.text3d(x,y,z,text,color=ch_color,scale=ch_scale)
    return fig


def _plot_helmet(
    renderer,
    info,
    to_cf_t,
    head_mri_t,
    coord_frame,
    color,
    *,
    alpha=0.25,
    scale=1.0,
):
    if color is None:
        color = DEFAULTS["coreg"]["helmet_color"]
    src_surf = get_meg_helmet_surf(info, head_mri_t)
    assert src_surf["coord_frame"] == FIFF.FIFFV_COORD_MRI
    if to_cf_t is not None:
        src_surf = transform_surface_to(
            src_surf, coord_frame, [to_cf_t["mri"], to_cf_t["head"]], copy=True
        )
    actor, dst_surf = renderer.surface(
        surface=src_surf, color=color, opacity=alpha, backface_culling=False
    )
    return actor, dst_surf, src_surf