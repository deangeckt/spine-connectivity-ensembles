"""Figure 6 panel A - the oracle raster.

This is the one part of figure 6 that reads the calcium H5 directly: it re-runs
detection on one scan rather than reading the run pickles. It detects on the RAW
oracle frames, while the pickles behind panels B-E were produced with
``ORACLE_TRIAL_RESIDUAL = True``, so the raster is an illustration of what a detected
ensemble looks like and its ``k`` indices are not interchangeable with theirs.
"""

import numpy as np
import matplotlib.pyplot as plt

# Deconvolved spikes, not raw calcium - higher SNR, and what the detector is fitted on.
TRACE_KEY = 'spike_trace'

# Detector settings. These must match scripts/ensemble_run.py's: a divergence silently
# changes K, and K changes which ensemble is ranked where.
LDS_N_SURROGATES         = 1000
LDS_K_FROM               = 'auto'
LDS_MEMBERSHIP_THRESHOLD = 2
LDS_MAX_MEMBER_FRAC      = 0.8
RANDOM_SEED              = 0

# Ensembles can have more members than a panel carries legibly. Trim to this many, drawn
# at random from the membership. Display only - the analysis uses the full membership.
MAX_MEMBERS      = 4
MEMBER_PICK_SEED = 0

# Colormap ceiling, as a percentile over the normalised values. A plain max would pin
# vmax to each neuron's single largest pixel and wash every other clip out to pale grey.
ORACLE_PCT_CLIP = 99.0

# Grayscale, as in the published panels: white low, black high.
CMAP = 'binary'


def scan_pool_root_ids(neurons_df, session, scan_idx, clf_type='E'):
    """root_ids of column neurons that have a unit in this scan.

    Reads only the `session` / `scan_idx` scalars out of the H5 — no traces — and
    only visits the ~1.2k nucleus groups that belong to column neurons, so this
    is seconds rather than the minutes a full `load_ex_functional_data` costs.
    """
    import h5py
    from activity_utils import build_nucleus_to_root_id
    from connectome_types import CALCIUM_H5_PATH

    df = neurons_df[neurons_df.clf_type == clf_type] if clf_type else neurons_df
    nuc_to_root = build_nucleus_to_root_id(df, df.root_id)

    out = []
    with h5py.File(CALCIUM_H5_PATH, 'r') as f:
        grp = f['neurons']
        for nuc, rid in nuc_to_root.items():
            g = grp.get(str(nuc))
            if g is None:
                continue
            for unit_name in g.keys():
                u = g[unit_name]
                if int(u['session'][()]) == session and int(u['scan_idx'][()]) == scan_idx:
                    out.append(int(rid))
                    break
    return sorted(out)


def units_from_func(func, root_ids, session, scan_idx):
    """`[(root_id, payload)]` for one scan, in `root_ids` order.

    Pulled out of the dict the detector already loaded rather than re-reading the
    H5: the rasters need the same traces the fit saw.
    """
    out = []
    for rid in root_ids:
        units = func.get(str(int(rid)), {})
        payload = next((p for (s, sc, _u), p in units.items()
                        if s == session and sc == scan_idx), None)
        if payload is not None:
            out.append((int(rid), payload))
    return out


def neuron_labels(root_ids, neurons_df, sep=' '):
    """Cell type + the last 6 digits of each root_id, for the stdout listing."""
    ct = neurons_df.set_index('root_id')['cell_type'].to_dict()
    return [f'{ct.get(int(r), "?")}{sep}…{str(int(r))[-6:]}' for r in root_ids]


def detect_scan_ensembles(neurons_df, session, scan_idx):
    """Run the detector on one scan.

    Returns `(ens_result, ens_root_ids, Z, pool_units)`:
      ens_result   — the detect_ensembles_dispatch dict (V, M, S, …)
      ens_root_ids — row order of V/M, i.e. the neurons the fit saw
      Z            — the (N, T) matrix the fit ran on (stimulus-window trimmed)
      pool_units   — `[(root_id, payload)]` with the *untrimmed* traces, which is
                     what the rasters are cut from
    """
    from ensembles import build_activity_matrix, detect_ensembles_dispatch
    from activity_utils import load_ex_functional_data_by_root_id
    from connectome_types import CALCIUM_H5_PATH, STIMULI_H5_PATH

    pool_rids = scan_pool_root_ids(neurons_df, session, scan_idx)
    func = load_ex_functional_data_by_root_id(
        CALCIUM_H5_PATH, neurons_df, pool_rids, align='interp',
        datasets=(TRACE_KEY,), progress=False)

    # stimulus_type='oracle' keeps only the maximally-repeated clip frames
    Z, ens_root_ids, _fps = build_activity_matrix(
        func, session, scan_idx, use_spikes=True,
        stimulus_type='oracle', h5_stim_path=STIMULI_H5_PATH)

    kwargs = dict(n_surrogates=LDS_N_SURROGATES, k_from=LDS_K_FROM,
                  membership_threshold_std=LDS_MEMBERSHIP_THRESHOLD,
                  max_member_frac=LDS_MAX_MEMBER_FRAC,
                  random_state=RANDOM_SEED)
    print(f'  detecting ensembles on ({Z.shape[0]}, {Z.shape[1]}) ({kwargs})…')
    ens_result = detect_ensembles_dispatch(
        Z, list(ens_root_ids), method='lds', trace_type='spike', **kwargs)

    return ens_result, list(ens_root_ids), Z, units_from_func(
        func, ens_root_ids, session, scan_idx)


def ensemble_members(ens_result, ens_root_ids, k):
    """root_ids of ensemble `k`, in the fit's row order."""
    mask = np.asarray(ens_result['M'])[:, k].astype(bool)
    return [int(r) for r, m in zip(ens_root_ids, mask) if m]


def rank_ensembles(ens_result, Z):
    """Component indices `k` ordered best-first by activation energy, Σ_t S_k(t)².

    The detector returns components unordered, so they have to be ranked before one can
    be named. `k` — the detector's own index — is the criterion-independent name; quote
    that when referring to one.
    """
    S = np.asarray(ens_result['S'], dtype=np.float64)
    return list(np.argsort(-(S ** 2).sum(axis=0)))


def load_oracle_windows(session, scan_idx):
    """The scan's oracle clip windows, or `[]` if the scan has none."""
    from activity_utils import load_scan_trials_df, get_oracle_condition_hashes
    from connectome_types import STIMULI_H5_PATH

    trials = load_scan_trials_df(STIMULI_H5_PATH, session, scan_idx)
    return get_oracle_condition_hashes(trials)


def build_oracle_raster(payload, oracle_windows):
    """One unit's oracle raster: `(n_repeats, n_clips, n_frames)`.

    `oracle_windows` comes from `activity_utils.get_oracle_condition_hashes` —
    `[(condition_hash, [(start_idx, end_idx), ...]), ...]`, sorted by hash so the clip
    ordering is identical across units in the same scan.

    Repeat counts and clip spans can differ by a frame or two, so everything is
    truncated to the common minimum.

    Values are min-maxed per neuron over all clips and repeats at once: deconvolved
    amplitudes are in arbitrary per-neuron units, so without this a single bright ROI
    blacks out the rest of the figure. Amplitudes stay comparable across clips within a
    row, but not between rows.
    """
    if payload.get('align') is None:
        raise ValueError(
            'build_oracle_raster needs ms_delay-aligned traces: trial windows are '
            'frame indices into the scan (field-1) clock, so an unaligned trace is '
            "sliced at the wrong frames. Reload with align='interp'.")
    if not oracle_windows:
        raise ValueError('oracle_windows is empty — no oracle clips in this scan')

    trace = np.asarray(payload[TRACE_KEY], dtype=np.float32)
    n_repeats = min(len(w) for _ch, w in oracle_windows)
    n_frames = min(e - s for _ch, w in oracle_windows for s, e in w[:n_repeats])
    if n_frames <= 0:
        raise ValueError('oracle clips have non-positive span')

    out = np.empty((n_repeats, len(oracle_windows), n_frames), dtype=np.float32)
    for j, (_ch, windows) in enumerate(oracle_windows):
        for i, (s, _e) in enumerate(windows[:n_repeats]):
            out[i, j] = trace[s:s + n_frames]

    out = out - out.min()
    peak = out.max()
    if peak > 0:
        out = out / peak
    return out


def build_oracle_rasters(units, oracle_windows):
    """`build_oracle_raster` over several units → `(n_neurons, n_repeats, n_clips, n_frames)`."""
    return np.stack([build_oracle_raster(p, oracle_windows) for _rid, p in units])


def _style_panel(ax):
    """Published-panel style: a plain black box, no ticks, no tick labels.

    Frame indices are not information a reader needs — the clip is named in the column
    title and the time base comes from the scale bar.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color('black')


def _container_size_inches(fig):
    """Width/height in inches of `fig`, whether it's a Figure or a SubFigure.

    A SubFigure has no get_figwidth/get_figheight — its allocated area is a fraction of
    the parent decided by the subfigures() ratios, and its rendered extent is only known
    after a draw, so this forces one.
    """
    if hasattr(fig, 'get_figwidth'):
        return fig.get_figwidth(), fig.get_figheight()
    parent = fig.get_figure()
    parent.canvas.draw()
    bbox = fig.get_window_extent()
    return bbox.width / parent.dpi, bbox.height / parent.dpi


def plot_oracle_rasters(rasters, fig, fps=None, clips_group_title=None, row_labels=None,
                        show_time_label=True, left_in=1.0, right_in=0.95, bottom_in=1.05,
                        fontsize=None):
    """Grid of oracle rasters: one panel row per neuron, one column per clip.

    `rasters` is `(n_neurons, n_repeats, n_clips, n_frames)`. Every panel shares one
    colormap scale, which is meaningful because `build_oracle_raster` normalised them —
    unlike a per-row stretch, which would draw a weak neuron as dark as a strong one.

    `fig`          — the Figure or SubFigure to draw the grid into (figure6.ipynb passes
                     one row of its composite figure)
    `clips_group_title` — one label centered above the whole column of clip numbers
                     (e.g. 'Oracle clips'); the columns themselves are numbered
    `row_labels`   — one string per neuron (e.g. cell type), drawn to the right of that
                     row's last panel. None leaves rows unlabelled — which cell sits
                     where then only goes to stdout.
    `show_time_label` — draw the 'Time →' caption under the scale bar. Turn off when the
                     panel is small enough that the bar reads fine on its own.
    `left_in`, `right_in`, `bottom_in` — fixed margins in inches for the axis furniture
                     (repeats bar, time bar, row labels). A narrow embed needs these
                     shrunk or the margins dominate the panel.
    """
    # Single knob overrides every hardcoded text size in the panel; pass None
    # to keep the original 10/11/13 mix.
    _fs_col   = fontsize if fontsize is not None else 11   # per-column clip number
    _fs_grp   = fontsize if fontsize is not None else 11   # clips_group_title
    _fs_row   = fontsize if fontsize is not None else 10   # row labels
    _fs_bar   = fontsize if fontsize is not None else 10   # scale-bar '1' / n_repeats
    _fs_rep   = fontsize if fontsize is not None else 11   # 'Repeats' caption
    _fs_sec   = fontsize if fontsize is not None else 11   # 's' + 'Time →'

    rasters = np.asarray(rasters)
    if rasters.ndim != 4:
        raise ValueError(f'expected (n_neurons, n_repeats, n_clips, n_frames), '
                         f'got shape {rasters.shape}')
    n_neurons, n_repeats, n_clips, n_frames = rasters.shape

    axes = fig.subplots(n_neurons, n_clips, squeeze=False)

    # Margins in inches, not fractions: the furniture around the grid (scale
    # bars, their labels, row labels) needs a fixed physical strip, and a
    # fractional margin shrinks it to nothing as soon as the figure is small.
    fig_w, fig_h = _container_size_inches(fig)
    top_in = 0.55 + (0.3 if clips_group_title else 0.0)
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        bottom=bottom_in / fig_h,
                        top=1 - top_in / fig_h,
                        wspace=0.14, hspace=0.14)

    vmax = float(np.percentile(rasters, ORACLE_PCT_CLIP)) or float(rasters.max()) or 1.0
    dur = n_frames / fps if fps else None

    for i in range(n_neurons):
        for j in range(n_clips):
            ax = axes[i][j]
            ax.imshow(rasters[i, :, j, :], cmap=CMAP, vmin=0.0, vmax=vmax,
                      aspect='auto', interpolation='nearest')
            _style_panel(ax)
            if i == 0:
                ax.set_title(str(j + 1), fontsize=_fs_col)

    # Offsets are given in inches and converted here, so the margin furniture
    # keeps its physical size when n_neurons or n_clips changes the figure.
    def _dx(inches):
        return inches / fig_w

    def _dy(inches):
        return inches / fig_h

    if clips_group_title:
        # One label centered over the whole grid, above the per-column numbers,
        # rather than repeating 'Oracle clip' on every column.
        _lx = axes[0][0].get_position().x0
        _rx = axes[0][-1].get_position().x1
        _ty = axes[0][0].get_position().y1
        fig.text(0.5 * (_lx + _rx), _ty + _dy(0.34), clips_group_title,
                 va='bottom', ha='center', fontsize=_fs_grp)

    if row_labels is not None:
        for i in range(n_neurons):
            p = axes[i][-1].get_position()
            fig.text(p.x1 + _dx(0.08), 0.5 * (p.y0 + p.y1), row_labels[i],
                     va='center', ha='left', fontsize=_fs_row)

    bl = axes[-1][0].get_position()      # bottom-left panel anchors the axis furniture

    # Repeats scale bar: a vertical line to the LEFT of the bottom-left panel,
    # exactly one panel's height, so it reads directly against the horizontal
    # time bar in the same corner rather than needing its own separate axis.
    x_bar = bl.x0 - _dx(0.30)
    fig.add_artist(plt.Line2D([x_bar, x_bar], [bl.y0, bl.y1], color='black', lw=1.5))
    fig.text(x_bar - _dx(0.06), bl.y1, '1', va='top', ha='right', fontsize=_fs_bar)
    fig.text(x_bar - _dx(0.06), bl.y0, str(n_repeats), va='bottom', ha='right', fontsize=_fs_bar)
    fig.text(x_bar - _dx(0.40), 0.5 * (bl.y0 + bl.y1), 'Repeats', rotation=90,
             va='center', ha='center', fontsize=_fs_rep)

    # Time scale bar spanning the panel, in place of a numbered x axis.
    if dur:
        y_bar = bl.y0 - _dy(0.22)
        fig.add_artist(plt.Line2D([bl.x0, bl.x0 + bl.width], [y_bar, y_bar],
                                  color='black', lw=1.5))
        fig.text(bl.x0 + 0.5 * bl.width, y_bar - _dy(0.07),
                 f'{dur:.3g} s', va='top', ha='center', fontsize=_fs_sec)
        if show_time_label:
            fig.text(0.5 * (bl.x0 + axes[-1][-1].get_position().x1),
                     y_bar - _dy(0.42), 'Time →', va='top', ha='center', fontsize=_fs_sec)

    return fig, axes
