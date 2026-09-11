"""Detect the ensembles and run the matched-control bootstraps behind figures 6, S14 and S15.

    python scripts/ensemble_run.py

Run it once. It writes one pickle per detector into ``data/activity/ensembles/``, plus a
``_numbers.md`` sidecar holding every count the paper quotes. Figures 6, S14 and S15 read
those pickles; no figure notebook runs the pipeline itself.

**This needs the two MICrONS activity H5 files** under ``data/activity/``. They are not in
the Zenodo snapshot either — build them first with
``scripts/extract_calcium_data_via_docker.ipynb``. See the README.

Budget 1-3 hours and 1-3 GB of RAM. Rejection sampling for the matched control groups
dominates; the population-event detector costs about 3x the ICA one.

Everything here follows the paper's "Functional ensemble detection and analysis" Methods
section; ``utils/ensembles.py`` holds the implementation. The settings below are that
section's parameters, in the same order it describes them.

Reproducibility
---------------
Deterministic given ``RANDOM_SEED``: the per-ensemble generators are seeded with
``zlib.crc32``. The matched nulls for connection probability and shared input depend on
which control groups the rejection sampler finds, so they are estimated, not fixed —
``MAX_ATTEMPTS`` and ``N_BOOTSTRAP`` below are what make those estimates tight enough to
be stable between runs. Observed values do not depend on any of this.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

# The progress lines below use arrows and box-drawing characters; a Windows console on a
# non-UTF-8 code page raises UnicodeEncodeError on them mid-run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from activity_utils import load_ex_functional_data_by_root_id
from connectome_types import (CALCIUM_H5_PATH, STIMULI_H5_PATH, SPINE_TABLE,
                              CONNECTOME_SYN_TABLE_PATH, ENSEMBLE_RESULTS_DIR)
from utils import load_neurons_table, load_synapses_position_transformed
from neuron_custom_features import calc_spines_features
from ensembles import (
    build_activity_matrix, detect_ensembles_dispatch, build_ensemble_members,
    build_ex_pool_arrays, build_pool_pos_arrays, build_pool_indegree_array,
    build_pool_feature_array, build_control_eligible_idx, build_member_cell_type_df,
    build_shared_input_strength_series, sample_matched_controls,
    run_spine_targeting_bootstrap, run_connection_probability_bootstrap,
    run_shared_input_bootstrap, run_shared_input_strength_bootstrap,
    ENSEMBLE_RUN_STEM, ensemble_results_path,
    shared_input_by_size, collect_run_numbers, format_run_numbers_md,
)


# ══════════════════════════════════════════════════════════════════════════════
# Settings — one pickle per detector
# ══════════════════════════════════════════════════════════════════════════════
# 'lds' is the ICA-based detector of the main text (figure 6); 'ecker' the
# population-event clustering used as an independent check (figure S14). Figure S15
# plots both.
METHODS = ['lds', 'ecker']

# Detection runs on the repeated "oracle" movie clips only, and on the trial residual:
# each clip's average over its ~10 repeats is subtracted from every repeat, so ensembles
# reflect trial-to-trial co-fluctuation rather than shared visual tuning.
ORACLE_TRIAL_RESIDUAL = True

# A scan needs at least this many recorded, structurally matched excitatory neurons.
MIN_NEURONS_PER_SCAN = 5

# A component whose members exceed this fraction of the scan's recorded population is a
# global fluctuation, not a discrete assembly, and is discarded.
MAX_MEMBER_FRAC = 0.8

RANDOM_SEED = 0

# ── ICA detector (Lopes-dos Santos et al., 2013) ──────────────────────────────
LDS_N_SURROGATES         = 1000    # circular shifts behind the eigenvalue threshold
LDS_MEMBERSHIP_THRESHOLD = 2       # member if its weight exceeds 2 SD of the component

# ── Population-event detector (Carrillo-Reid 2015; Herzog 2021; Ecker 2024) ───
HRZ_K_RANGE           = (2, 20)   # cluster counts scanned, Davies-Bouldin picks one
HRZ_N_SHUFFLES_BIN    = 100       # surrogates behind the network-event threshold
HRZ_N_SHUFFLES_MEMBER = 10000     # surrogates behind the membership test
HRZ_MEMBER_ALPHA      = 0.05      # Benjamini-Hochberg q across all neuron x cluster tests

# ── Matched control groups ────────────────────────────────────────────────────
# Per ensemble, up to N_CONTROLS random groups of the same size, drawn from excitatory
# neurons belonging to no ensemble in any scan, so the observed and null populations are
# fully disjoint. Beyond size, matched on:
#   compactness  the group's mean pairwise soma-to-soma distance, within the greater of
#                COMPACTNESS_TOL or COMPACTNESS_FLOOR_UM — every measure
#   in-degree    the group's summed excitatory and summed inhibitory in-degree, each
#                within INDEGREE_TOL — shared input and shared input strength only
# Internal connectivity (the number of connected member pairs) is matched by the
# measures that read internal wiring; `sample_matched_controls` handles it per call.
N_CONTROLS           = 1000
# Attempts the rejection sampler may spend per ensemble before giving up and keeping
# whatever it found. The tightest ensembles accept roughly one draw in 10^5-10^6, and a
# pool left short there is what makes a null swing between runs. The sampler tests
# candidates in numpy batches at ~10^6/s, so this budget costs minutes. Ensembles that
# fill early stop early; only the tight ones spend it all.
MAX_ATTEMPTS         = 200_000_000

# Replicates of the paired bootstrap that turns the drawn control groups into a null.
# Cheap — it only resamples groups already in hand — and 1000 is too few to pin a
# p-value near a significance threshold.
N_BOOTSTRAP          = 200_000
COMPACTNESS_TOL      = 0.35
COMPACTNESS_FLOOR_UM = 15.0
INDEGREE_TOL         = 0.35


def method_kwargs_for(method):
    return {
        'lds': dict(
            n_surrogates=LDS_N_SURROGATES,
            membership_threshold_std=LDS_MEMBERSHIP_THRESHOLD,
            max_member_frac=MAX_MEMBER_FRAC,
            random_state=RANDOM_SEED,
        ),
        'ecker': dict(
            k_range=HRZ_K_RANGE,
            n_shuffles_bin=HRZ_N_SHUFFLES_BIN,
            n_shuffles_member=HRZ_N_SHUFFLES_MEMBER,
            member_alpha=HRZ_MEMBER_ALPHA,
            max_member_frac=MAX_MEMBER_FRAC,
            random_state=RANDOM_SEED,
        ),
    }[method]


# ══════════════════════════════════════════════════════════════════════════════
# Everything both detectors share — loaded once
# ══════════════════════════════════════════════════════════════════════════════
def load_shared():
    """The column, its calcium traces, its synapses and the pool arrays built over them."""
    neurons_df = load_neurons_table(use_column_manual_ct=True)
    ex_df      = neurons_df[neurons_df.clf_type == 'E'].copy()
    ex_func_data = load_ex_functional_data_by_root_id(
        CALCIUM_H5_PATH, neurons_df, ex_df.root_id)

    spine_df = pd.read_csv(SPINE_TABLE)
    syn_df   = load_synapses_position_transformed(base_syn_table_path=CONNECTOME_SYN_TABLE_PATH)
    df, syn_with_tags = calc_spines_features(neurons_df, syn_df)

    # The baseline the figures draw as a black dashed line: the E→E spine-targeting rate
    # over the whole micro-column.
    ex_to_ex_syn_tags = syn_with_tags[(syn_with_tags.pre_clf_type == 'E') &
                                      (syn_with_tags.post_clf_type == 'E')]
    n_ee_spine = int((ex_to_ex_syn_tags['tag'] == 'spine').sum())
    baseline_ee_spine_frac = n_ee_spine / len(ex_to_ex_syn_tags)
    print(f'Baseline E→E spine fraction: {n_ee_spine:,} / {len(ex_to_ex_syn_tags):,} '
          f'= {baseline_ee_spine_frac:.3f}')

    # One index space for everything: all excitatory neurons of the column. Every
    # pool-indexed array — soma positions, in-degree, per-neuron features — spans all of
    # them. Which of them a control group may be *drawn* from is a separate question,
    # settled per run by build_control_eligible_idx once the ensembles are known. Dropping
    # members from the index space instead would make their own compactness and in-degree
    # targets uncomputable.
    full_column_root_ids = ex_df['root_id'].values
    recorded_root_ids    = full_column_root_ids[
        np.isin(full_column_root_ids, list({int(r) for r in ex_func_data.keys()}))]
    print(f'Full column: {len(full_column_root_ids)} excitatory neurons  |  '
          f'recorded in ≥1 scan: {len(recorded_root_ids)}')

    pool = build_ex_pool_arrays(spine_df, full_column_root_ids)
    pos_array = build_pool_pos_arrays(pool)
    deg_array = build_pool_indegree_array(pool, syn_df, full_column_root_ids)

    # Shared input strength, per neuron: the mean within-column out-degree of all its
    # presynaptic partners — the same measure figure 4 plots.
    input_strength = build_pool_feature_array(
        pool, build_shared_input_strength_series(df), label='shared input strength')

    return dict(
        neurons_df=neurons_df, ex_func_data=ex_func_data, spine_df=spine_df, syn_df=syn_df,
        syn_with_tags=syn_with_tags, df=df,
        baseline_ee_spine_frac=baseline_ee_spine_frac,
        full_column_root_ids=full_column_root_ids, recorded_root_ids=recorded_root_ids,
        pool=pool, pos_array=pos_array, deg_array=deg_array,
        input_strength=input_strength,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Detection, scan by scan — the expensive step
# ══════════════════════════════════════════════════════════════════════════════
def detect_all_scans(method, shared):
    method_kwargs = method_kwargs_for(method)
    print(f'\n=== method={method!r}  kwargs={method_kwargs} ===')

    neurons_df   = shared['neurons_df']
    ex_func_data = shared['ex_func_data']
    spine_df     = shared['spine_df']

    scan_keys = sorted({(sess, sc) for scans in ex_func_data.values()
                        for (sess, sc, _uid) in scans})
    print(f'{len(scan_keys)} (session, scan_idx) pairs with recorded E neurons')

    all_ensemble_rows = []
    ensemble_members_by_scank = {}
    skip_counts = {}
    # One row per scan the detector was offered, kept whether or not it yielded
    # ensembles: the paper reports "N scans, M neurons recorded", and only this loop
    # ever sees the per-scan activity matrix.
    scan_rows = []

    def _scalar_meta(meta):
        """The scalar entries of a detector's population_meta, prefixed, for scan_info_df.

        K, the eigenvalue thresholds and the significant-bin fraction are Methods-section
        numbers that otherwise live only in this script's stdout. Filtered to scalars so
        the arrays in there — eigenvalue spectra, Davies-Bouldin curves — do not bloat
        every pickle; read those from the log if they are needed.
        """
        return {f'meta_{k}': v for k, v in (meta or {}).items()
                if isinstance(v, (bool, int, float, str, np.integer, np.floating))}

    def _skip(reason, sess, sc, n_neurons=np.nan, n_frames=np.nan, meta=None):
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
        scan_rows.append({'session': sess, 'scan_idx': sc, 'used': False,
                          'skip_reason': reason, 'n_neurons': n_neurons,
                          'n_frames': n_frames, 'n_ensembles': 0, **_scalar_meta(meta)})

    for (sess, sc) in scan_keys:
        try:
            Z, rec_ids, _fps = build_activity_matrix(
                ex_func_data, sess, sc, STIMULI_H5_PATH,
                trial_residual=ORACLE_TRIAL_RESIDUAL)
        except ValueError:
            _skip('no_stimulus_frames', sess, sc); continue
        except RuntimeError:
            _skip('no_traces', sess, sc); continue
        if Z.shape[0] < MIN_NEURONS_PER_SCAN:
            _skip('too_few_neurons', sess, sc, Z.shape[0], Z.shape[1]); continue

        res = detect_ensembles_dispatch(
            Z, rec_ids, method=method, trace_type='spike', **method_kwargs)
        if res['population_meta']['K'] == 0:
            _skip('no_ensembles', sess, sc, Z.shape[0], Z.shape[1],
                  meta=res['population_meta']); continue

        print(f'── s{sess} sc{sc}  N={Z.shape[0]} ──')
        ensemble_members, k_list, K = build_ensemble_members(res, rec_ids, neurons_df)
        rank_map = {int(kk): int(rr) + 1
                    for kk, rr in zip(res['ensemble_meta_df']['k'], res['ensemble_meta_df']['rank'])}

        for k in k_list:
            member_ids = list(ensemble_members[k]['root_id'])
            member_set = set(member_ids)
            scan_k     = f'{sess}_{sc}_{k}'
            ensemble_members_by_scank[scan_k] = member_ids
            sub = spine_df[spine_df['pre_pt_root_id'].isin(member_set) &
                           spine_df['post_pt_root_id'].isin(member_set)]
            n_total = len(sub)
            n_edges = sub.drop_duplicates(['pre_pt_root_id', 'post_pt_root_id']).shape[0]
            n_spine = int((sub['tag'] == 'spine').sum())
            n_multi = n_total - n_edges
            if n_total > 0 and n_multi > 0:
                print(f'    → multi-contact: {n_multi} extra synapses from {n_edges} pairs (scan_k={scan_k})')
            all_ensemble_rows.append({
                'session': sess, 'scan_idx': sc, 'k': k,
                'scan_k': scan_k,
                'rank': rank_map[k],
                'n_members': len(member_ids),
                'n_synapses': n_total, 'n_edges': n_edges, 'n_spine': n_spine,
            })
        scan_rows.append({'session': sess, 'scan_idx': sc, 'used': True,
                          'skip_reason': '', 'n_neurons': Z.shape[0],
                          'n_frames': Z.shape[1], 'n_ensembles': len(k_list),
                          **_scalar_meta(res['population_meta'])})

    all_ensembles_df = pd.DataFrame(all_ensemble_rows)
    scan_info_df     = pd.DataFrame(scan_rows)
    n_scans_used = (all_ensembles_df[['session', 'scan_idx']].drop_duplicates().shape[0]
                    if len(all_ensembles_df) else 0)
    print(f'\nUsed {n_scans_used} scans  |  skipped {sum(skip_counts.values())}:  '
          + '  '.join(f'{r}={n}' for r, n in sorted(skip_counts.items())))
    print(f'Total ensembles pooled: {len(all_ensembles_df)}')
    return all_ensembles_df, ensemble_members_by_scank, scan_info_df, skip_counts


# ══════════════════════════════════════════════════════════════════════════════
# The control pool, and every measure except shared input
# ══════════════════════════════════════════════════════════════════════════════
def run_wiring_measures(shared, all_ensembles_df, ensemble_members_by_scank):
    syn_df    = shared['syn_df']
    pool      = shared['pool']
    pos_array = shared['pos_array']
    recorded_root_ids    = shared['recorded_root_ids']
    full_column_root_ids = shared['full_column_root_ids']

    # Which neurons a control group may contain — settled only once detection is done.
    # Every ensemble member, in every scan, is subtracted from the pool every null draws
    # from, so no neuron can be ensemble signal in one comparison and null in another.
    eligible_idx, eligible_root_ids = build_control_eligible_idx(
        pool, ensemble_members_by_scank)

    seed_fn = lambda row: (RANDOM_SEED * 100_003 + int(row['session']) * 1_000_000
                           + int(row['scan_idx']) * 1_000 + int(row['k'])) % (2 ** 32)

    # Controls matched on size, internal connected pairs and compactness. Spine targeting
    # tests over these; shared input reuses them when it is not additionally in-degree
    # matched.
    matched_controls_df, matched_groups = sample_matched_controls(
        all_ensembles_df, label_col='scan_k', seed_fn=seed_fn, pool=pool,
        n_controls=N_CONTROLS, max_attempts=MAX_ATTEMPTS,
        label_to_members=ensemble_members_by_scank,
        pos_array=pos_array,
        dist_tol=COMPACTNESS_TOL, abs_floor_um=COMPACTNESS_FLOOR_UM,
        eligible_idx=eligible_idx, return_groups=True,
    )
    print(f'matched_controls_df shape: {matched_controls_df.shape}')

    # Figure 6C / S14C — spine targeting of the synapses between members.
    spine_targeting = run_spine_targeting_bootstrap(
        all_ensembles_df, matched_controls_df, label_col='scan_k',
        n_null=N_BOOTSTRAP, seed=RANDOM_SEED)

    # Figure 6B / S14B — connection probability among members. `all_ex_root_ids` sets the
    # E→E baseline, so it tracks the eligible pool.
    connection_probability = run_connection_probability_bootstrap(
        all_ensembles_df, label_col='scan_k',
        syn_df=syn_df, all_ex_root_ids=eligible_root_ids,
        pool=pool, n_controls=N_CONTROLS, seed=RANDOM_SEED,
        max_attempts=MAX_ATTEMPTS, n_boot=N_BOOTSTRAP,
        label_to_members=ensemble_members_by_scank,
        pos_array=pos_array, dist_tol=COMPACTNESS_TOL, abs_floor_um=COMPACTNESS_FLOOR_UM,
        eligible_idx=eligible_idx)

    # Figure S15B — shared input strength, a per-neuron feature of the members. Its
    # controls are matched on size, compactness and summed E and I in-degree, without the
    # edge matching that only applies to measures of internal wiring.
    #
    # In-degree is matched because the measure is confounded by it: members carry a median
    # E+I in-degree around 61 against a pool median near 43, and in-degree correlates with
    # the feature itself — a neuron with more input is a larger, better-reconstructed cell.
    _, indegree_matched_groups = sample_matched_controls(
        all_ensembles_df, label_col='scan_k', seed_fn=seed_fn, pool=pool,
        n_controls=N_CONTROLS, max_attempts=MAX_ATTEMPTS, match_edges=False,
        label_to_members=ensemble_members_by_scank,
        pos_array=pos_array,
        dist_tol=COMPACTNESS_TOL, abs_floor_um=COMPACTNESS_FLOOR_UM,
        degree_array=shared['deg_array'], degree_tol=INDEGREE_TOL,
        eligible_idx=eligible_idx, return_groups=True,
    )

    shared_input_strength = run_shared_input_strength_bootstrap(
        ensemble_members_by_scank, pool, indegree_matched_groups, shared['input_strength'],
        n_null=N_BOOTSTRAP, seed=RANDOM_SEED,
        name='shared input strength',
        value_label='shared input strength (mean pre-synaptic out-degree)')

    return {
        'all_ensembles_df':          all_ensembles_df,
        'ensemble_members_by_scank': ensemble_members_by_scank,
        'matched_controls_df':       matched_controls_df,
        'spine_targeting':           spine_targeting,
        'connection_probability':    connection_probability,
        'shared_input_strength':     shared_input_strength,
        # descriptive, not a test: what the detector grouped, by cell type
        'member_cell_type_df':       build_member_cell_type_df(
                                         ensemble_members_by_scank, shared['neurons_df']),
        'baseline_ee_spine_frac':    shared['baseline_ee_spine_frac'],
        'control_pool':              'disjoint',
        'n_eligible':                len(eligible_idx),
        # The paper quotes plain counts — N scans, M neurons, a pool of P — that are not
        # recoverable from the measure dicts alone, so the run records them alongside.
        'n_column_ex':               len(full_column_root_ids),
        'n_recorded':                len(recorded_root_ids),
        'stimulus_type':             'oracle',
        'oracle_trial_residual':     ORACLE_TRIAL_RESIDUAL,
        'n_controls':                N_CONTROLS,
        # carried out so shared input can reuse them without resampling
        '_eligible_idx':             eligible_idx,
        '_matched_groups':           matched_groups,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Shared presynaptic input — figures 6D, 6E, S14D, S14E
# ══════════════════════════════════════════════════════════════════════════════
def run_shared_input(shared, all_ensembles_df, ensemble_members_by_scank, base):
    # `all_ex_root_ids` stays the full column here: it decides which post-synaptic neurons
    # have a presynaptic set at all. Narrowing it to the eligible pool would erase the
    # observed ensembles' own shared-input counts, since members are not in that pool.
    return run_shared_input_bootstrap(
        ensemble_members_by_scank, shared['full_column_root_ids'], shared['syn_df'],
        ensembles_df=all_ensembles_df,
        pool=shared['pool'],
        label_col='scan_k', n_controls=N_CONTROLS, seed=RANDOM_SEED,
        max_attempts=MAX_ATTEMPTS, n_boot=N_BOOTSTRAP,
        pos_array=shared['pos_array'],
        dist_tol=COMPACTNESS_TOL, abs_floor_um=COMPACTNESS_FLOOR_UM,
        degree_array=shared['deg_array'], degree_tol=INDEGREE_TOL,
        eligible_idx=base['_eligible_idx'],
        syn_with_tags=shared['syn_with_tags'],
    )


def save_results(method, results):
    """Write the pickle and its `_numbers.md` sidecar.

    The sidecar is the paper-facing output: every count a sentence might quote,
    regenerated on every run.
    """
    os.makedirs(ENSEMBLE_RESULTS_DIR, exist_ok=True)
    stem = ENSEMBLE_RUN_STEM.format(method=method)
    results_path = ensemble_results_path(method)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved results      → {results_path}')

    numbers_path = os.path.join(ENSEMBLE_RESULTS_DIR, stem + '_numbers.md')
    md = format_run_numbers_md(
        collect_run_numbers(results, label=method),
        d_size=shared_input_by_size(results),
        heading=f'{method} — {stem}')
    with open(numbers_path, 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    print(f'Saved run numbers  → {numbers_path}')


def main():
    print(f'stimulus=oracle  trial_residual={ORACLE_TRIAL_RESIDUAL}')
    print(f'methods: {METHODS} = {len(METHODS)} runs')

    shared = load_shared()
    for done, method in enumerate(METHODS, start=1):
        print(f'\n{"─" * 78}\n[{done}/{len(METHODS)}] {method}\n{"─" * 78}')
        all_ensembles_df, members, scan_info_df, skip_counts = detect_all_scans(method, shared)
        if not len(all_ensembles_df):
            print(f'!! {method}: no ensembles detected in any scan — skipping')
            continue
        base = run_wiring_measures(shared, all_ensembles_df, members)
        shared_input = run_shared_input(shared, all_ensembles_df, members, base)

        results = {k: v for k, v in base.items() if not k.startswith('_')}
        results['shared_input'] = shared_input
        results['method']       = method
        results['scan_info_df'] = scan_info_df
        results['skip_counts']  = skip_counts
        save_results(method, results)


if __name__ == '__main__':
    main()
