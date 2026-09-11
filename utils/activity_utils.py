"""Calcium / functional data loading for figures 6, 7, S14 and S15.

Trimmed from the repo-root ``activity_utils.py``; only the loading path the four
figures need is kept. ``get_oracle_condition_hashes`` (from
``analysis/spine_project/activity/activity_utils.py``) and ``select_scan_matrix`` /
``slice_window`` (from ``.../activity/video/video_utils.py``) are inlined at the
bottom, so this is the single activity module in ``utils/``.

The calcium H5 is ``coreg_manual_v4_calcium_v2.h5``, keyed by **nucleus_id**. It is
NOT part of the Zenodo snapshot - see the README for how to build it. Only figure 6
panel A, figure 7 panel E, and figure S15's recorded-neuron list read it; every other
panel comes from the pickles in ``data/activity/ensembles/``.

Two things the loader handles once, so consumers never have to (see ``activity.md``):

1. **Keying.** v1 was keyed by ``pt_root_id`` at materialization v1507. v2 is keyed by
   ``nucleus_id``, stable across materializations. ``load_ex_functional_data_by_root_id``
   re-keys to the current v1718 root_id, which is what every join here wants.

2. **Acquisition timing.** A two-photon "frame" is not an instant: the beam rasters 4
   depths in sequence, so two neurons sharing frame index ``k`` can have been sampled up
   to one full frame (~159 ms at 6.3 fps) apart. ``ms_delay`` is that per-unit offset.
   The true sample time of unit ``u`` at frame ``k`` is
   ``frame_times[k] + ms_delay_u / 1000`` - *not* ``k / fps``, which is additionally
   wrong by up to 2.5 s over a 40k-frame scan.

     'interp' (default) - resample each unit's traces onto the scan's shared
                          ``frame_times`` grid, so column ``t`` of every unit really is
                          the same instant. This is what the figures use.
     None               - leave traces untouched, expose the unit's true time axis as
                          ``payload['time']``.
     'shift'            - whole-frame roll, a cruder alternative to 'interp'.

   Either way ``payload['frame_times']`` is the real per-scan clock and
   ``payload['time']`` the correct time axis. Never compute ``np.arange(nframes) / fps``.
"""

from typing import Dict, Tuple, Iterable

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── H5 v2 schema ─────────────────────────────────────────────────────────────
# Genuinely per-unit, so they carry the unit's own `ms_delay` offset:
UNIT_TRACES = ('calcium_trace', 'spike_trace')
# Per-*scan* quantities that the extractor wrote a copy of under every unit.
# nda.ManualPupil and nda.Treadmill are both documented as "synced to the field 1
# scan times", i.e. already on `frame_times` — shifting them would be wrong.
SCAN_TRACES = ('pupil_radius', 'treadmill')
# Scalars copied through as-is.
UNIT_META = ('fps', 'nframes', 'nfields', 'oracle_score', 'ms_delay', 'field',
             'mask_id', 'mask_type', 'um_x', 'um_y', 'um_z', 'px_x', 'px_y',
             'field_x', 'field_y', 'field_z', 'pt_root_id_v1507')

_ALIGN_MODES = (None, 'interp', 'shift')


# ── the two files that are not in the Zenodo snapshot ────────────────────────
_H5_BLURB = {
    'coreg_manual_v4_calcium_v2.h5': 'calcium + deconvolved spike traces, ~19 GB',
    'microns_per_scan_stimuli.h5':   'per-scan trial / oracle-clip table, ~68 MB',
}


def require_activity_h5(*paths, needed_by=None):
    """Fail early, with build instructions, if an activity H5 is missing.

    With no arguments, checks both files. Call it at the top of any notebook that reads
    the MICrONS functional recordings, so a missing download surfaces as a readable
    message instead of an h5py OSError forty cells later.
    """
    import os

    if not paths:
        from connectome_types import CALCIUM_H5_PATH, STIMULI_H5_PATH
        paths = (CALCIUM_H5_PATH, STIMULI_H5_PATH)

    missing = [p for p in paths if not os.path.exists(p)]
    if not missing:
        return

    what = f' needed by {needed_by}' if needed_by else ''
    lines = ['', f'Missing MICrONS activity data{what}:', '']
    for p in missing:
        lines.append(f'    {p}')
        blurb = _H5_BLURB.get(os.path.basename(p))
        if blurb:
            lines.append(f'        ({blurb})')
    lines += [
        '',
        'These files are NOT part of the Zenodo snapshot - they are extracted from the',
        'MICrONS microns_phase3_nda database, which is only reachable from inside its',
        'own container. To build them:',
        '',
        '    1. bring up the database container',
        '       https://github.com/cajal/microns-nda-access#database-container',
        '    2. copy scripts/extract_calcium_data_via_docker.ipynb into it, together',
        '       with data/raw_tables/coregistration_manual_v4.csv. That CSV is not in',
        '       the Zenodo snapshot either - get it with',
        '       `python download_data_cave.py --steps raw` (needs a CAVE account).',
        '    3. run that notebook top to bottom',
        '    4. copy the two .h5 files it writes back into data/activity/',
        '',
        "The notebook's first cell has the step-by-step; see also \"The activity",
        'half\" in the README.',
        '',
    ]
    raise FileNotFoundError('\n'.join(lines))


def _scan_group_path(session, scan_idx) -> str:
    return f'scans/s{int(session):02d}/scan{int(scan_idx):02d}'


def load_scan_frame_times(h5_path: str) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """`{(session, scan_idx): {'frame_times': (nframes,) float64, 'ndepths': int}}`.

    `frame_times` are `nda.ScanTimes.frame_times` — the real acquisition time of
    each frame, synced to the first pixel of field 1. The arrays are marked
    read-only and handed out by reference (one 40k-float64 array per scan is
    320 kB; copying it into all ~19k unit payloads would be ~6 GB).
    """
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, 'r') as f:
        if 'scans' not in f:
            raise KeyError(
                f"{h5_path!r} has no '/scans' group — this looks like the v1 calcium H5. "
                f"Use coreg_manual_v4_calcium_v2.h5 (see scripts/"
                f"extract_calcium_data_via_docker.ipynb).")
        for s_name, s_grp in f['scans'].items():
            for sc_name, sc_grp in s_grp.items():
                if 'frame_times' not in sc_grp:
                    continue
                ft = np.asarray(sc_grp['frame_times'][()], dtype=np.float64)
                ft.flags.writeable = False
                out[(int(s_name[1:]), int(sc_name.replace('scan', '')))] = {
                    'frame_times': ft,
                    'ndepths': int(sc_grp['ndepths'][()]) if 'ndepths' in sc_grp else None,
                }
    return out


def align_trace(trace: np.ndarray, frame_times: np.ndarray, ms_delay: float,
                fps: float, method: str = 'interp') -> np.ndarray:
    """Resample one unit's trace from its own clock onto the scan's `frame_times`.

    The unit's sample `k` was taken at `frame_times[k] + ms_delay/1000`, so the
    value belonging at grid point `frame_times[k]` is read off the unit's own
    (later) time axis. Since ms_delay >= 0 this shifts activity *forward* in index
    by `ms_delay * fps` frames — the raw index systematically under-reports when
    the event happened, by more for the deeper fields.

    Edges are clamped to the first/last sample, so length is preserved.
    """
    d = float(ms_delay) / 1000.0
    if d == 0.0 or method is None:
        return trace
    if method == 'interp':
        return np.interp(frame_times, frame_times + d, trace)
    if method == 'shift':
        n = int(round(d * float(fps)))
        if n <= 0:
            return trace
        n = min(n, len(trace))
        return np.concatenate([np.full(n, trace[0], dtype=trace.dtype), trace[:-n]])
    raise ValueError(f"align method must be one of {_ALIGN_MODES}, got {method!r}")


def load_ex_functional_data(h5_path: str, nucleus_ids: Iterable, align: str | None = 'interp',
                            datasets: Iterable[str] | None = None, progress: bool = True,
                            ) -> Dict[str, Dict[Tuple[int, int, int], Dict[str, np.ndarray]]]:
    """Load functional data from the v2 calcium H5, keyed by **nucleus_id**.

    Parameters
    ----------
    h5_path     : path to `coreg_manual_v4_calcium_v2.h5`
    nucleus_ids : nucleus_ids to load (int or str). These are ~7-digit ids, *not*
                  18-digit root_ids — passing root_ids is rejected loudly rather
                  than silently returning an empty dict, which is how the v1
                  keying bug hid for so long. To load by root_id use
                  `load_ex_functional_data_by_root_id`.
    align       : 'interp' | 'shift' | None — ms_delay handling, see module docstring
    datasets    : subset of UNIT_TRACES + SCAN_TRACES to load (default: all). Use
                  e.g. `('spike_trace',)` to halve the memory when the calcium
                  trace isn't needed.

    Returns
    -------
    `{str(nucleus_id): {(session, scan_idx, unit_id): payload}}` where payload has
    the traces, every scalar in UNIT_META that the file carries, plus:
        frame_times — (nframes,) real per-scan clock, shared read-only reference
        time        — the time axis *for this payload's traces*: `frame_times`
                      when aligned, `frame_times + ms_delay/1000` when align=None
        align       — the mode used, so consumers can assert on it
    """
    if align not in _ALIGN_MODES:
        raise ValueError(f"align must be one of {_ALIGN_MODES}, got {align!r}")

    want = set(UNIT_TRACES) | set(SCAN_TRACES) if datasets is None else set(datasets)
    ids = [str(int(n)) for n in nucleus_ids]
    # root_ids are ~8.6e17, nucleus_ids ~1e5-1e6. A root_id here means a caller
    # that hasn't been migrated off the v1 keying.
    if ids and min(len(i) for i in ids) > 12:
        raise ValueError(
            "load_ex_functional_data is keyed by nucleus_id, but the ids passed look "
            "like root_ids (18 digits). Use load_ex_functional_data_by_root_id(h5_path, "
            "neurons_df, root_ids) instead — the v1507 bridge no longer exists.")

    scans = load_scan_frame_times(h5_path)
    ex_func: Dict[str, Dict[Tuple[int, int, int], Dict[str, np.ndarray]]] = {}
    n_missing_scan = n_len_mismatch = 0

    with h5py.File(h5_path, 'r') as f:
        if 'neurons' not in f:
            raise KeyError("'neurons' group not found in H5 file")
        if f.attrs.get('key') != 'nucleus_id':
            raise KeyError(
                f"{h5_path!r} is not keyed by nucleus_id (root attr 'key'="
                f"{f.attrs.get('key')!r}). This is the v1 file; use "
                f"coreg_manual_v4_calcium_v2.h5.")

        neurons_grp = f['neurons']
        iterator: Iterable[str] = tqdm(ids, desc='Loading functional data') if progress else ids

        for nid in iterator:
            if nid not in neurons_grp:
                continue
            units_grp = neurons_grp[nid]
            unit_dict: Dict[Tuple[int, int, int], Dict[str, np.ndarray]] = {}

            for unit_name in units_grp.keys():
                u = units_grp[unit_name]
                session  = int(u['session'][()])
                scan_idx = int(u['scan_idx'][()])
                unit_id  = int(u['unit_id'][()])

                payload: Dict[str, np.ndarray] = {'nucleus_id': int(nid)}
                for name in UNIT_META:
                    if name not in u:
                        continue
                    v = u[name][()]
                    if isinstance(v, bytes):
                        v = v.decode('utf-8', 'replace')
                    payload[name] = v if isinstance(v, str) else v.item()

                fps      = float(payload['fps'])
                ms_delay = float(payload.get('ms_delay', 0.0))

                scan_meta = scans.get((session, scan_idx))
                if scan_meta is None:
                    n_missing_scan += 1
                    continue
                frame_times = scan_meta['frame_times']

                # Read the traces first so the clock can be matched to their length.
                # frame_times should be exactly nframes long, but one ragged scan
                # must not abort a 19k-unit load, so reconcile rather than raise:
                # a short trace is a truncated recording, and the first n frame_times
                # still apply; a trace *longer* than the clock cannot be placed in
                # time at all, so that unit is dropped and counted.
                raw, n_samples = {}, None
                for name in want:
                    if name not in u:
                        continue
                    raw[name] = u[name][()].astype(float)
                    if name in UNIT_TRACES:
                        n_samples = (len(raw[name]) if n_samples is None
                                     else min(n_samples, len(raw[name])))

                clock = frame_times
                if n_samples is not None and n_samples != len(frame_times):
                    if n_samples > len(frame_times):
                        n_len_mismatch += 1
                        continue
                    clock = frame_times[:n_samples]   # view; stays read-only

                for name, tr in raw.items():
                    tr = tr[:len(clock)]
                    if name in UNIT_TRACES and align is not None:
                        tr = align_trace(tr, clock, ms_delay, fps, method=align)
                    payload[name] = tr

                payload['frame_times'] = clock
                payload['ndepths'] = scan_meta['ndepths']
                payload['align'] = align
                payload['time'] = (clock if align is not None
                                   else clock + ms_delay / 1000.0)
                unit_dict[(session, scan_idx, unit_id)] = payload

            if unit_dict:
                ex_func[nid] = unit_dict

    if n_missing_scan:
        print(f'[load_ex_functional_data] WARNING: dropped {n_missing_scan} units whose '
              f'(session, scan_idx) has no frame_times in /scans')
    if n_len_mismatch:
        print(f'[load_ex_functional_data] WARNING: dropped {n_len_mismatch} units whose '
              f'trace is longer than the scan frame_times — they cannot be placed in '
              f'time. A large count here means /scans is wrong, not the traces.')
    return ex_func


def build_nucleus_to_root_id(neurons_df: pd.DataFrame, root_ids: Iterable | None = None) -> dict:
    """`{nucleus_id (int): root_id (int)}` for the given root_ids (default: all).

    Stays in python-int space throughout: 18-digit root_ids need 19 significant
    digits and silently lose precision the moment pandas promotes them to float64
    (which `Series.map` does as soon as there is one miss).
    """
    df = neurons_df[['root_id', 'nucleus_id']].dropna()
    if root_ids is not None:
        df = df[df['root_id'].isin({int(r) for r in root_ids})]
    return {int(n): int(r) for r, n in zip(df['root_id'], df['nucleus_id'])}


def load_ex_functional_data_by_root_id(h5_path: str, neurons_df: pd.DataFrame,
                                       root_ids: Iterable | None = None,
                                       align: str | None = 'interp', **kwargs
                                       ) -> Dict[str, Dict[Tuple[int, int, int], Dict[str, np.ndarray]]]:
    """`load_ex_functional_data` re-keyed to current (v1718) root_ids as `str`.

    This is the drop-in for every consumer in the repo — they all join to
    `neurons_df` on root_id downstream. It replaces the whole v1507 bridge block:

        ex_func_data = load_ex_functional_data_by_root_id(
            h5_path, neurons_df, ex_df.root_id)

    `nucleus_id → root_id` is many-to-one in principle (the coregistration table
    has one segment covering 6 nuclei — a merge error). If that ever bites the
    micro-column population the collision is reported rather than silently
    overwriting, because the two nuclei are genuinely different cells.
    """
    nuc_to_root = build_nucleus_to_root_id(neurons_df, root_ids)
    by_nucleus = load_ex_functional_data(h5_path, nuc_to_root.keys(), align=align, **kwargs)

    out: Dict[str, Dict[Tuple[int, int, int], Dict[str, np.ndarray]]] = {}
    collisions = []
    for nid_str, units in by_nucleus.items():
        rid = str(nuc_to_root[int(nid_str)])
        if rid in out:
            collisions.append(rid)
            out[rid].update(units)
        else:
            out[rid] = dict(units)
    if collisions:
        print(f'[load_ex_functional_data_by_root_id] WARNING: {len(set(collisions))} root_id(s) '
              f'map to >1 nucleus_id (merge-error segments); their units were pooled: '
              f'{sorted(set(collisions))[:5]}')
    print(f'Functional data: {len(out)} neurons '
          f'({sum(len(u) for u in out.values())} units, align={align!r})')
    return out
def load_scan_trials_df(h5_stim_path: str, session, scan_idx):
    with h5py.File(h5_stim_path, 'r') as f:
        g = f['scans'][f"s{int(session):02d}"][f"scan{int(scan_idx):02d}"]
        tg = g['trials']
        return pd.DataFrame({
            'trial_idx':      tg['trial_idx'][()].astype(np.int32),
            'type':           tg['type'][()].astype(str),
            'start_idx':      tg['start_idx'][()].astype(np.int32),
            'end_idx':        tg['end_idx'][()].astype(np.int32),
            'start_time':     tg['start_time'][()].astype(np.float64),
            'end_time':       tg['end_time'][()].astype(np.float64),
            'condition_hash': tg['condition_hash'][()].astype(str),
        })

# ── inlined from analysis/spine_project/activity/activity_utils.py ───────────

def get_oracle_condition_hashes(trials_df, stim_type='stimulus.clip'):
    """Find the oracle (maximally-repeated) condition_hashes in one scan's trial table.

    Returns a list of (condition_hash, [(start_idx, end_idx), ...]) sorted by
    condition_hash so any two units' response vectors share an identical clip ordering.
    Oracle clips are those whose repeat count equals the maximum across all clips of
    this stim_type (in the MICrONS CLIP set: the 6×10-repeat oracle set).
    """
    sub = trials_df[trials_df['type'].str.lower() == stim_type.lower()]
    if sub.empty:
        return []
    counts = sub.groupby('condition_hash').size()
    hashes = sorted(counts[counts == counts.max()].index.tolist())
    out = []
    for ch in hashes:
        grp = sub[sub.condition_hash == ch]
        windows = [(int(s), int(e)) for s, e in zip(grp['start_idx'], grp['end_idx'])]
        out.append((ch, windows))
    return out

# ── inlined from analysis/spine_project/activity/video/video_utils.py ────────

def select_scan_matrix(ex_func_data, session, scan_idx, trace_key='spike_trace'):
    """Stack one trace per neuron for a fixed (session, scan_idx) into an (N, T) matrix.

    Keeps only units with matching fps and nframes to the first surviving unit.
    Returns (root_ids: int64 (N,), traces: float32 (N, T), fps: float).

    The video paints all N neurons at a single cursor position, so column `t` has
    to be one instant across neurons — which requires ms_delay-aligned traces
    (`activity_utils` `align='interp'`, the default). Raw traces are up to one
    frame apart between imaging depths and would make the animation show the
    deeper fields lighting up late.
    """
    per_neuron = []
    for nid_str, units in ex_func_data.items():
        for (s, sc, _uid), payload in units.items():
            if s != session or sc != scan_idx:
                continue
            if trace_key not in payload:
                raise KeyError(f"trace_key {trace_key!r} missing on unit "
                               f"({s}, {sc}) of neuron {nid_str}")
            if payload.get('align') is None:
                raise ValueError(
                    f"select_scan_matrix requires ms_delay-aligned traces: neuron "
                    f"{nid_str} was loaded with align=None. Reload with align='interp'.")
            per_neuron.append((int(nid_str), payload))
            break  # one unit per neuron per (session, scan_idx)

    if not per_neuron:
        raise ValueError(f"No units found for session={session}, scan_idx={scan_idx}")

    fps0 = per_neuron[0][1]['fps']
    nframes0 = per_neuron[0][1]['nframes']
    kept = [(nid, p) for (nid, p) in per_neuron
            if p['fps'] == fps0 and p['nframes'] == nframes0]
    if len(kept) < 2:
        raise ValueError(f"< 2 units with matching fps/nframes in session={session}, "
                         f"scan_idx={scan_idx} (got {len(kept)})")

    root_ids = np.asarray([nid for (nid, _) in kept], dtype=np.int64)
    traces = np.vstack([np.asarray(p[trace_key], dtype=np.float32) for (_, p) in kept])
    return root_ids, traces, float(fps0)


def slice_window(traces, fps, start_sec=None, end_sec=None, speed=1):
    """Crop traces to [start_sec, end_sec) and downsample by `speed`.

    Speed is a frame stride: 4× means take every 4th frame. Non-integer speeds
    round to nearest int; speeds < 1 are clamped to 1. Returns:
      traces_win: (N, T') sub-array (view when possible)
      t0_frame:   absolute frame index of the first kept frame (before stride)
      stride:     integer stride actually used
    """
    stride = max(1, int(round(speed)))
    T = traces.shape[1]
    t0 = 0 if start_sec is None else int(round(start_sec * fps))
    t1 = T if end_sec   is None else int(round(end_sec   * fps))
    t0 = max(0, min(T - 1, t0))   # clamp to valid frame; ensures at least 1 frame
    t1 = max(t0 + 1, min(T, t1))
    return traces[:, t0:t1:stride], t0, stride
