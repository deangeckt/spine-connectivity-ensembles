"""Ensemble detection and the matched-control bootstraps behind figures 6, S14 and S15.

Everything the paper's "Functional ensemble detection and analysis" Methods section
describes, in one module:

  detection        two independent algorithms over each scan's trial-residual activity —
                   `detect_ensembles_lds`   ICA-based (Lopes-dos Santos et al., 2013),
                                            the main-text detector behind figure 6
                   `detect_ensembles_ecker` population-event clustering (Carrillo-Reid
                                            et al., 2015; Herzog et al., 2021; Ecker
                                            et al., 2024), the independent check behind
                                            figure S14
  matched controls `sample_matched_controls` draws, per ensemble, up to 1,000 random
                   groups of the same size from a pool of excitatory neurons belonging to
                   no ensemble in any scan, matched on soma compactness and — depending on
                   the measure — internal connectivity and summed E/I in-degree
  measures         one paired-bootstrap per structural measure of the paper:
                   `run_connection_probability_bootstrap`   figure 6B / S14B
                   `run_spine_targeting_bootstrap`          figure 6C / S14C
                   `run_shared_input_bootstrap`             figure 6D, 6E / S14D, S14E
                   `run_shared_input_strength_bootstrap`    figure S15B
  reporting        `shared_input_by_size` re-aggregates shared input per ensemble size
                   (figure 6D right)

`scripts/ensemble_run.py` drives all of it and writes one pickle per detector into
`data/activity/ensembles/`. The figure notebooks only read those pickles; none of them
runs detection.

Reproducibility
---------------
Every run from here on is deterministic given the run's random seed: the per-ensemble
generators are seeded with `zlib.crc32`, where the run behind the paper used
`hash(str(label))`, which CPython randomises per process unless `PYTHONHASHSEED` is set
(that run's value was never recorded). A fresh run therefore does not reproduce the
published pickles bit for bit.

What it reproduces exactly: detection, the ensembles and their members, the synapses
among them, spine targeting and shared input strength. What it does not: the matched
nulls for connection probability and shared input, which depend on *which* control groups
the rejection sampler happens to find. Observed values are identical either way — only
the nulls move.

The sampler's accept rate falls to ~1e-5 on the tightest ensembles, so `MAX_ATTEMPTS`
and `N_BOOTSTRAP` in `scripts/ensemble_run.py` are set large enough that those nulls are
stable between runs.
"""

import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import davies_bouldin_score

from connectome_types import ENSEMBLE_RESULTS_DIR

# ═════════════════════════════════════════════════════════════════════════════
# Where the run results live
# ═════════════════════════════════════════════════════════════════════════════

# -- the run pickles -----------------------------------------------------------
# Not part of the Zenodo snapshot: you produce them once, locally, with
# scripts/ensemble_run.py, which first needs the two activity H5 files. No figure
# notebook runs the pipeline itself - they only read its output.
ENSEMBLE_RUN_STEM = '{method}_oracle-resid_disjoint_distance_deg_allmem'


def ensemble_results_path(method):
    """Path to one detector's run pickle ('lds' or 'ecker')."""
    import os
    return os.path.join(ENSEMBLE_RESULTS_DIR,
                        ENSEMBLE_RUN_STEM.format(method=method) + '.pkl')


def require_ensemble_results(*methods):
    """Paths to the run pickles, failing early with instructions if any is missing.

    Returns a single path for one method, a tuple for several.
    """
    import os

    paths = [ensemble_results_path(m) for m in methods]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        lines = ['', 'Missing ensemble run results:', '']
        lines += [f'    {p}' for p in missing]
        lines += [
            '',
            'These are not downloaded - you produce them once, locally:',
            '',
            '    python scripts/ensemble_run.py',
            '',
            'It detects ensembles in every scan and runs the matched-control bootstraps',
            'for both detectors, writing one .pkl per detector into',
            'data/activity/ensembles/. Budget 1-2 hours and 1-3 GB of RAM.',
            '',
            'It needs the two activity H5 files first - run',
            'activity_utils.require_activity_h5() to check, and see the README for how',
            'to build them.',
            '',
        ]
        raise FileNotFoundError('\n'.join(lines))
    return paths[0] if len(paths) == 1 else tuple(paths)


def detect_ensembles_lds(
    traces: np.ndarray,
    root_ids: list,
    trace_type: str = 'spike',
    membership_threshold_std: float = 2.0,
    n_surrogates: int = 1000,
    max_member_frac: float = 1.0,
    random_state: int = 0,
) -> dict:
    """
    Eigenvalue threshold → FastICA ensemble detection (Lopes-dos-Santos 2013).

    Reference: Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013) "Detecting cell
    assemblies in large neuronal populations." J Neurosci Methods 220:149-166.

    Pipeline
    --------
    1.  Z-score each neuron over time → Zn (N, T).
    2.  Correlation matrix C = Zn·Znᵀ/T, eigendecomposed.
    3.  Count the eigenvalues that exceed the surrogate noise ceiling λ_cs → K.
    4.  Project onto the top-K eigenvectors and run FastICA inside that subspace.
        PCA alone would force the components orthogonal, which is a mathematical
        constraint with no biological basis (real assemblies overlap); ICA finds
        K statistically *independent* directions in the same subspace, un-mixing
        overlapping assemblies. The PCA step is only there to denoise first.
    5.  Membership: v_k[i] > `membership_threshold_std`·std(v_k), one-sided, so
        only genuine co-activators join (not anti-correlated neurons).
    6.  Activation strength R_k(t) = (z·v_k)² − Σᵢ v_k[i]²z_i². The subtracted
        term zeroes the projector diagonal, so one neuron firing alone does not
        register as an assembly activation — only true co-activation does.

    Corrections vs. the published pipeline
    --------------------------------------
    Taking K from the circular-shift ceiling λ_cs rather than the Marchenko-Pastur
    edge λ_MP is the one deviation, and it is a correction to a real failure mode
    rather than a preference:

    The published ceiling is the Marchenko-Pastur upper edge
    λ_MP = (1 + √(N/T))², which is the largest eigenvalue attainable when the N
    neurons are mutually independent *and* each is temporally i.i.d. Deconvolved
    calcium violates the second assumption badly (residual indicator kernel,
    159 ms frames, slow shared drift), and temporal autocorrelation inflates the
    top eigenvalues. With T ≫ N the MP edge is also extremely tight — at N=56,
    T=31951 it is λ_MP ≈ 1.0855 — so nearly anything clears it and K is
    over-estimated.

    The circular-shift surrogate ceiling λ_cs fixes this: shifting each neuron by
    an independent random lag preserves that neuron's own firing rate *and*
    autocorrelation while destroying cross-neuron alignment, so it is the honest
    null for this data. λ_cs is the 95th percentile of the surrogate maximum
    eigenvalue.

    Crucially, λ_cs is now applied **before** the ICA, not after. Previously the
    surrogates ran at the end and λ_cs was reported but never used: the ICA was
    always fit with K_mp components, and callers were expected to drop the excess
    post-hoc. That does not work, for three reasons:
      - ICA is a joint fit, so K_cs components selected out of a K_mp-component
        fit are not the components a K_cs-component fit would have produced;
      - if K_mp ≫ K_cs the ICA is fit inside a subspace whose extra dimensions
        are pure autocorrelation noise, and ICA mixes that noise across *all*
        components — including the ones that would have been kept;
      - after ICA there is no longer a 1:1 map from component to eigenvalue, so
        "the components that survived λ_cs" is not even well defined.
    Deciding K first and fitting once in the clean subspace avoids all three.

    Note the surrogate shifts here are per-neuron independent (unlike the
    sequence-shift null in `detect_ensembles_ecker`): the statistic being nulled
    is an eigenvalue of the *joint* correlation matrix, so a shift shared across
    neurons would leave the cross-neuron structure intact and null nothing.

    Parameters
    ----------
    traces                   : (N, T) ndarray
    root_ids                 : length-N sequence of neuron root_ids
    trace_type               : 'spike' | 'calcium' — stored in population_meta only
    membership_threshold_std : θ; neuron i joins ensemble k when v_k[i] > θ·std(v_k)
    n_surrogates             : circular-shift surrogates for λ_cs. 1000 is ample —
                               λ_cs is a 95th percentile, not a tail quantile, and
                               a surrogate costs ~3 ms at (N=56, T=32k).
    max_member_frac          : post-detection size cutoff, same backstop as in
                               `detect_ensembles_ecker`. 1.0 (default) = off. When
                               set to a fraction f in (0, 1), any component whose
                               one-sided membership (M[:, k].sum()) exceeds f·N is
                               dropped as a non-specific / near-global co-activation
                               mode and does NOT count towards K. Dropped-component
                               count is reported in population_meta['n_size_dropped'].
    random_state             : seed for FastICA and circular-shift surrogates

    Returns
    -------
    dict with keys
        ensembles_df      — long-format DataFrame [k, root_id, weight, is_member]
        ensemble_meta_df  — one row per ensemble: k, n_members, weight_mean/std/max
        population_meta   — K, K_mp, K_cs, lambda_max_mp, lambda_cs,
                            lambda_gap, N, T, trace_type
        V                 — (N, K) L2-normalised weight matrix  (float32)
        M                 — (N, K) membership matrix (int8, one-sided)
        S                 — (T, K) per-frame activation strength (float32)

    `population_meta['K']` is the K actually used to fit — it equals `K_cs` — so
    downstream code needs no post-hoc trimming.
    """
    traces = np.asarray(traces, dtype=np.float32)
    N, T = traces.shape
    if len(root_ids) != N:
        raise ValueError(f"len(root_ids)={len(root_ids)} != N={N}")
    # z-score per neuron
    mu    = traces.mean(axis=1, keepdims=True)
    sigma = traces.std(axis=1,  keepdims=True) + 1e-8
    Zn    = (traces - mu) / sigma                              # (N, T)

    # correlation matrix + eigendecompose
    C = (Zn @ Zn.T) / T                                       # (N, N)
    eigenvalues, eigenvectors = np.linalg.eigh(C)             # ascending
    lambda_max_mp = (1.0 + np.sqrt(N / T)) ** 2
    K_mp          = int(np.sum(eigenvalues > lambda_max_mp))
    lambda_gap    = float(eigenvalues[-1] / lambda_max_mp)

    # ── circular-shift surrogate ceiling λ_cs — BEFORE choosing K (see docstring)
    rng   = np.random.default_rng(random_state)
    maxes = np.empty(n_surrogates)
    Zs    = np.empty_like(Zn)
    for s in range(n_surrogates):
        # per-row np.roll, not the vectorised `_shift_rows`: at this shape the
        # latter's (N, T) int64 index array costs more than N contiguous rolls
        # (~12 ms vs ~2.4 ms at N=56, T=32k), and this loop runs n_surrogates times
        for i, sh in enumerate(rng.integers(1, max(2, T), size=N)):
            Zs[i] = np.roll(Zn[i], int(sh))
        maxes[s] = np.linalg.eigvalsh((Zs @ Zs.T) / T).max()
    lambda_cs = float(np.percentile(maxes, 95))
    K = K_cs  = int(np.sum(eigenvalues > lambda_cs))

    print(f"[detect_ensembles_lds] N={N}  T={T}  q={T/N:.2f}  "
          f"λ_MP={lambda_max_mp:.4f} (K_mp={K_mp})  "
          f"λ_cs={lambda_cs:.4f} (K_cs={K_cs})  "
          f"→ K={K}  λ_gap={lambda_gap:.3f}")
    if K_mp > 0:
        print(f"[detect_ensembles_lds] {K_mp - K_cs}/{K_mp} MP-significant components "
              f"are explained by temporal autocorrelation alone "
              f"({100 * (K_mp - K_cs) / K_mp:.0f}% of K_mp)")

    V = np.empty((N, 0), dtype=np.float32)
    M = np.zeros((N, 0), dtype=np.int8)
    S = np.empty((T, 0), dtype=np.float32)
    n_size_dropped = 0

    if K > 0:
        top_vecs = eigenvectors[:, -K:]                       # (N, K)
        Z_proj   = (top_vecs.T @ Zn).T                       # (T, K)

        ica = FastICA(n_components=K, max_iter=1000,
                      random_state=random_state, whiten='unit-variance')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            ica.fit(Z_proj)

        V_raw = top_vecs @ ica.components_.T                  # (N, K)
        V = (V_raw / (np.linalg.norm(V_raw, axis=0, keepdims=True) + 1e-12)
             ).astype(np.float32)

        # sign-flip: largest-magnitude weight per component → positive
        for k in range(K):
            if V[np.argmax(np.abs(V[:, k])), k] < 0:
                V[:, k] *= -1.0

        # one-sided membership: true co-activators only
        M = np.zeros((N, K), dtype=np.int8)
        for k in range(K):
            thresh  = membership_threshold_std * float(np.std(V[:, k]))
            M[:, k] = (V[:, k] > thresh).astype(np.int8)

        # per-frame activation strength: R_k(t) = (z·v_k)² − Σ_i v_k[i]² z[i]²
        A     = Zn.T @ V                                      # (T, K)
        B_mat = (Zn.T ** 2) @ (V ** 2)                       # (T, K)
        S     = (A ** 2 - B_mat).astype(np.float32)          # (T, K)

        # post-detection size cutoff (same backstop as detect_ensembles_ecker): a
        # component recruiting most of the population is a non-specific / global
        # co-activation mode, not a targeted assembly, so it is dropped and does
        # not count towards K.
        if max_member_frac < 1.0:
            max_members = max_member_frac * N
            keep = M.sum(axis=0) <= max_members
            n_size_dropped = int((~keep).sum())
            if n_size_dropped:
                print(f"[detect_ensembles_lds] dropping {n_size_dropped} component(s) "
                      f"whose membership exceeds {max_member_frac:.0%} of N")
                V, M, S = V[:, keep], M[:, keep], S[:, keep]
                K = int(keep.sum())

    # ── build DataFrames ───────────────────────────────────────────────────────
    root_ids_arr = np.asarray(list(root_ids), dtype=np.int64)
    rows = []
    for k in range(K):
        for i, rid in enumerate(root_ids_arr):
            rows.append({'k': k, 'root_id': int(rid),
                         'weight': float(V[i, k]),
                         'is_member': bool(M[i, k])})
    ensembles_df = pd.DataFrame(rows, columns=['k', 'root_id', 'weight', 'is_member'])

    meta_rows = []
    for k in range(K):
        w = V[:, k]
        meta_rows.append({
            'k':           k,
            'n_members':   int(M[:, k].sum()),
            'weight_mean': float(w.mean()),
            'weight_std':  float(w.std()),
            'weight_max':  float(w.max()),
        })
    ensemble_meta_df = (
        pd.DataFrame(meta_rows) if meta_rows
        else pd.DataFrame(columns=['k', 'n_members', 'weight_mean', 'weight_std', 'weight_max'])
    )

    population_meta = {
        'K':             K,
        'K_mp':          K_mp,
        'K_cs':          K_cs,
        'n_surrogates':  int(n_surrogates),
        'lambda_max_mp': float(lambda_max_mp),
        'lambda_cs':     float(lambda_cs),
        'lambda_gap':    float(lambda_gap),
        'N':             N,
        'T':             T,
        'trace_type':    trace_type,
        'method':        'lds',
        'max_member_frac': float(max_member_frac),
        'n_size_dropped':  n_size_dropped,
    }

    return {
        'ensembles_df':     ensembles_df,
        'ensemble_meta_df': ensemble_meta_df,
        'population_meta':  population_meta,
        'V': V, 'M': M, 'S': S,
    }


def _shift_rows(X, shifts):
    """Circularly roll each row i of X (N, n_bins) by shifts[i], vectorised."""
    N, n_bins = X.shape
    idx = (np.arange(n_bins)[None, :] - shifts[:, None]) % n_bins
    return X[np.arange(N)[:, None], idx]


def _shift_cols(S, shifts):
    """Circularly roll each column k of S (n_bins, K) by shifts[k], vectorised."""
    n_bins, K = S.shape
    idx = (np.arange(n_bins)[:, None] - shifts[None, :]) % n_bins
    return S[idx, np.arange(K)[None, :]]


def _prenorm_rows(X):
    """Center and L2-normalise each row of X (N, n_bins) once, so that a later
    Pearson correlation against any centered+normalised column vector is a plain
    matmul. Rows with zero variance normalise to all-zero → correlation 0."""
    Xc = X - X.mean(axis=1, keepdims=True)
    return Xc / (np.sqrt((Xc ** 2).sum(axis=1, keepdims=True)) + 1e-12)


def _prenorm_cols(S):
    """Column-wise counterpart of `_prenorm_rows` for a (n_bins, K) matrix."""
    Sc = S - S.mean(axis=0, keepdims=True)
    return Sc / (np.sqrt((Sc ** 2).sum(axis=0, keepdims=True)) + 1e-12)


def _multi_corr(X, seqs):
    """Pearson correlation of every row of X (N, n_bins) against every column of
    seqs (n_bins, K), vectorised into one matmul → (N, K). Used so the expensive
    part of the membership-shuffle test (shifting X) only has to happen once per
    shuffle, not once per shuffle per candidate cluster."""
    Xc = X - X.mean(axis=1, keepdims=True)                      # (N, n_bins)
    Sc = seqs - seqs.mean(axis=0, keepdims=True)                 # (n_bins, K)
    num  = Xc @ Sc                                               # (N, K)
    xden = np.sqrt((Xc ** 2).sum(axis=1))                        # (N,)
    sden = np.sqrt((Sc ** 2).sum(axis=0))                        # (K,)
    den  = xden[:, None] * sden[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    return np.nan_to_num(out)


def detect_ensembles_ecker(
    traces: np.ndarray,
    root_ids: list,
    trace_type: str = 'spike',
    bin_frames: int = 1,
    k_range: tuple = (2, 20),
    n_shuffles_bin: int = 100,
    n_shuffles_member: int = 10000,
    bin_sig_percentile: float = 95.0,
    member_alpha: float = 0.05,
    max_member_frac: float = 1.0,
    random_state: int = 0,
) -> dict:
    """
    Significant-bin clustering → per-neuron correlation ensemble detection
    (Carrillo-Reid 2015 / Herzog 2021 / Ecker 2024, assemblyfire).

    Reference: Ecker A, Egas Santander D, Bolaños-Puchet S, Isbister JB, Reimann MW
    (2024) "Cortical cell assemblies and their underlying connectivity: An in silico
    study." PLOS Comp Biol. doi:10.1371/journal.pcbi.1011891
    Implemented in the `assemblyfire` package (github.com/BlueBrain/assemblyfire).

    Validation note (2025-07-13): the five pipeline steps below were verified
    step-by-step against the "Assembly detection" Methods section of the above paper.
    All steps match. The one ambiguity is the shuffle range used for the bin-
    significance null (step 2): the paper says "strict spike shifting" but does not
    specify the shift magnitude; this implementation uses full circular shifts (lag
    1..n_bins-1), consistent with assemblyfire.

    Data setup (this project): traces are deconvolved spike signals derived from
    calcium imaging at 6.3 fps (~159 ms/frame). Since each frame is already much
    coarser than the paper's 20 ms bins, `bin_frames=1` (default) is correct — no
    further temporal binning is needed.

    NOTE on fidelity: the PLOS Comp Biol Methods section gives a "minimal
    description" of the pipeline and defers full details to `assemblyfire`'s source.
    This function follows that minimal description step-for-step (see below), but
    a few implementation choices are ours where the papers don't specify low-level
    detail (noted inline). If you need bit-for-bit parity with the published
    results, diff this against github.com/BlueBrain/assemblyfire directly.

    Pipeline
    --------
    1.  Optionally bin `traces` into windows of `bin_frames` frames (sum within
        each window). Default `bin_frames=1` = no binning, i.e. operate at the
        native frame resolution of `traces` — the paper bins ms-resolution
        spikes into 20 ms windows. If your spike signal is at ms resolution
        (fps ≈ 1000), pass `bin_frames=20` to match the paper exactly. If it
        is already at a coarser frame rate, set `bin_frames` accordingly or
        leave at 1 if each frame already spans ~20 ms. To compute it from fps:
        `bin_frames=max(1, round(0.02 * fps))`.
    2.  Population activity per bin is compared to a threshold = mean activity
        + the 95th percentile of the std of `n_shuffles_bin` circularly-shifted
        controls (each neuron's own trace shifted by an independent random
        amount). Bins above threshold are "significant".
    3.  Cosine similarity between the (N,)-activation vectors of the significant
        bins → Ward-linkage hierarchical clustering (on 1-cosine-similarity as
        the distance; the papers say "hierarchically clustered" using Ward's
        linkage on the similarity matrix but don't spell out the distance
        conversion, so this is our reading of it). Cluster counts in `k_range`
        are scanned and the one with the lowest Davies-Bouldin index is kept —
        exactly as `Figure 6—figure supplement` panels of the eLife paper you
        attached (and its companion paper) describe.
    4.  Each surviving cluster's bins define a binary "activation sequence"
        (1 at that cluster's significant bins, 0 elsewhere, over *all* bins,
        significant or not). Every neuron's binned trace is correlated against
        this sequence, and tested against a circular-shift null (see
        "Corrections" for how significance is assessed).
    5.  A cluster is only kept as a functional assembly if the mean pairwise
        spike correlation of its surviving members exceeds the mean pairwise
        correlation of the whole recorded population — Herzog et al.'s final
        co-firing criterion. Clusters failing this are dropped (do not count
        towards K). Note this criterion is weak by construction: members were
        selected for correlating with the same sequence, so they will almost
        always correlate with each other above the population average. Treat it
        as a sanity check, not as evidence the assembly is real.

    Deviations from the literal published algorithm
    -----------------------------------------------
    Three problems surface when the algorithm as published is run on this data.

    (a) Per-neuron SD normalisation — step 2 thresholds `X.sum(axis=0)`, a raw sum
        across neurons. Deconvolved calcium amplitudes are in *arbitrary
        per-neuron units* (a bright ROI yields systematically larger values than
        a dim one), so a handful of loud neurons decided which bins counted as
        network events, and the same imbalance skewed the cosine similarities in
        step 3. Each neuron's trace is now divided by its own SD before both
        steps, which equalises contributions while preserving non-negativity and
        the "sum = population activity" interpretation. Centering is deliberately
        *not* applied: it would change the sparse non-negative structure the method
        assumes. Pearson correlations in steps 4-5 are scale-invariant, so this only
        affects *which* bins and clusters are found, never the membership
        correlations themselves.

    (b) Benjamini-Hochberg membership + p-values from exceedance counts —
        membership significance was a `np.percentile` of the null correlations,
        with the multiplicity correction computed by the *caller* as α/N. Two
        problems: the number of tests is N·K_candidate, not N (at N=56, K=6 that
        is 336 tests, so the correction was ~6× too lenient); and a 99.91th
        percentile estimated from 1000 shuffles is interpolated between the top
        one or two draws, i.e. almost pure noise. Membership is now an empirical
        p-value p = (1 + #{null ≥ observed}) / (n_shuffles + 1), with K_candidate
        known at test time.

        The correction is Benjamini-Hochberg rather than Bonferroni. Bonferroni
        controls the family-wise error rate — the probability of *any*
        false member anywhere — which is the wrong target when many true members
        are expected, and it scales badly: at K_candidate=20 the per-test
        threshold is α/1120 = 4.5e-5, so the empirical p-value floor 1/(n+1)
        cannot even reach it without >200k shuffles. BH controls the false
        discovery rate (the fraction of *called* members that are spurious),
        which is what a downstream group statistic actually cares about, and its
        rank-i threshold α·i/m stays well clear of the p-value floor. The function
        warns whenever the shuffle budget leaves the calls resolution-limited.

    (c) `k_range` default (2, 20) instead of (5, 20) — Davies-Bouldin only ranks
        partitions relative to each other; it never reports "there are no
        clusters here". A floor of 5 therefore *guarantees* at least 5 clusters
        regardless of the data. The previous run selected k=6, one step off the
        floor, which is exactly the signature of a scan that wanted to go lower
        and was fenced in. The full DB curve is now returned in
        population_meta['davies_bouldin_curve'] so the choice can be inspected
        rather than trusted.

    Also fixed: the DB index is now scored on the L2-normalised activation
    vectors, matching the cosine geometry the Ward tree was actually built in
    (it was previously scored on the unnormalised vectors, so the criterion
    selecting k disagreed with the criterion that built the tree). For unit
    vectors ‖x−y‖² = 2(1−cos), so Ward is properly Euclidean-consistent there.

    Performance note: the step-4 null now circularly shifts the K *activation
    sequences* rather than the N *neuron traces*. Shifting either side by a
    uniform random lag produces the identical marginal null for each
    (neuron, cluster) pair — the circular cross-correlation is the same — but
    shifting a (n_bins, K) matrix instead of an (N, n_bins) one, with the neuron
    traces centered and normalised once outside the loop, is ~5× faster
    (~3.3 ms vs ~17 ms per shuffle at N=56, T=32k). Within one shuffle all
    neurons then share a lag, which makes the null draws dependent *across*
    neurons; that is harmless here because each (i, k) threshold is computed from
    its own marginal, and Bonferroni does not assume independence across tests.

    Parameters
    ----------
    traces, root_ids, trace_type : same as `detect_ensembles_lds`
    bin_frames            : frames per bin (sum); 1 = no extra binning (see above)
    k_range                : (lo, hi) cluster counts scanned for the Davies-Bouldin
                             minimum. Default (2, 20); the paper used 5..20 — see
                             correction (c) for why the floor was lowered.
    n_shuffles_bin         : circular-shift surrogates for the bin-significance
                             threshold (paper used 100)
    n_shuffles_member      : circular-shift surrogates for the per-neuron
                             membership test. Default 10000; the paper used 1000,
                             which cannot resolve a Bonferroni-corrected α — see
                             correction (b). Costs ~3.3 ms each at N=56, T=32k.
    member_alpha           : α for the membership test — the Benjamini-Hochberg FDR
                             level q (default 0.05)
    bin_sig_percentile     : percentile of shuffled std defining the bin threshold
    max_member_frac        : post-detection size cutoff. 1.0 (default) = off. When
                             set to a fraction f in (0, 1), any assembly that
                             *passed* the co-firing criterion but whose membership
                             exceeds f·N is dropped as non-specific and does NOT
                             count towards K. This is the backstop for a genuine
                             near-global co-activation mode — one that survives the
                             co-firing criterion yet still recruits most of the
                             population. Such an assembly
                             carries no targeting *specificity* and would otherwise
                             dominate any pooled member-vs-member statistic, so it is
                             excluded rather than reported. Off by default so faithful
                             runs keep every co-firing survivor; only an explicit
                             fraction (e.g. 0.5) triggers it. Size-dropped clusters
                             are tallied separately from co-firing drops in the log
                             and in population_meta['n_size_dropped'].
    random_state           : seed for the shuffles (reproducibility, like the
                             `random_state=0` already used for FastICA above)

    Returns
    -------
    Same dict schema as `detect_ensembles_lds` — this is the whole point, so the
    rest of the notebook (dataframe-building, spine stats, plotting) doesn't change:
        ensembles_df, ensemble_meta_df, population_meta, V, M, S
    with the following method-specific reinterpretation of V/M/S:
        V[i, k] — correlation of neuron i's binned trace with assembly k's binary
                  activation sequence (plays the role LdS's ICA loading plays:
                  a continuous per-neuron "weight")
        M[i, k] — 1 if neuron i passed both the per-neuron correlation threshold
                  and the cluster survived the pairwise-correlation criterion
        S[t, k] — assembly k's binary activation sequence broadcast back out to
                  the original T frames (so downstream `(S**2).sum(axis=0)` is
                  still a valid "how long was this assembly active" ranking,
                  just duration-based here rather than energy-based)
    `population_meta['K_cs']`, `['lambda_max_mp']`, `['lambda_cs']`,
    `['lambda_gap']` are all `None` here — there's no MP/eigenvalue step in this
    method, so `USE_K_CS`-style downstream branches that check
    `if K_cs is not None` will simply no-op (K is already the final,
    significance-filtered count). Extra method-specific fields are also included
    (n_bins, bin_frames, n_sig_bins, sig_bin_frac, k_scanned, davies_bouldin_best,
    davies_bouldin_curve, n_candidate_clusters, member_p_threshold) — additive, so
    they won't break code that only reads the keys above.
    """
    traces = np.asarray(traces, dtype=np.float32)
    N, T = traces.shape
    if len(root_ids) != N:
        raise ValueError(f"len(root_ids)={len(root_ids)} != N={N}")
    rng = np.random.default_rng(random_state)
    bin_frames = max(1, int(bin_frames))

    def _empty_result(**meta_extra):
        V = np.empty((N, 0), dtype=np.float32)
        M = np.zeros((N, 0), dtype=np.int8)
        S = np.empty((T, 0), dtype=np.float32)
        ensembles_df = pd.DataFrame(columns=['k', 'root_id', 'weight', 'is_member'])
        ensemble_meta_df = pd.DataFrame(
            columns=['k', 'n_members', 'weight_mean', 'weight_std', 'weight_max'])
        population_meta = {
            'K': 0, 'K_cs': None, 'lambda_max_mp': None, 'lambda_cs': None,
            'lambda_gap': None, 'N': N, 'T': T, 'trace_type': trace_type,
            'method': 'ecker', 'bin_frames': bin_frames,
            # defaults for every method-specific key, so an early return still
            # yields the same schema as a full run and callers can read any key
            # without guarding (meta_extra overrides these)
            'n_bins': None, 'n_sig_bins': 0, 'sig_bin_frac': None,
            'k_scanned': list(k_range), 'davies_bouldin_best': None,
            'davies_bouldin_curve': {}, 'n_candidate_clusters': 0,
            'n_shuffles_member': int(n_shuffles_member),
            'member_alpha': float(member_alpha),
            'member_p_threshold': None,
            'whole_pop_mean_corr': None,
            'max_member_frac': float(max_member_frac), 'n_size_dropped': 0,
            **meta_extra,
        }
        return {'ensembles_df': ensembles_df, 'ensemble_meta_df': ensemble_meta_df,
                'population_meta': population_meta, 'V': V, 'M': M, 'S': S}

    # 1. optional binning ------------------------------------------------------
    if bin_frames > 1:
        n_bins = T // bin_frames
        X = traces[:, :n_bins * bin_frames].reshape(N, n_bins, bin_frames).sum(axis=2)
    else:
        n_bins = T
        X = traces.copy()

    # 1b. per-neuron normalisation (correction (a) — see docstring). Deconvolved
    # amplitudes are in arbitrary per-neuron units, so without this the loudest
    # ROIs dominate both the population sum in step 2 and the cosine similarities
    # in step 3. Dividing by each neuron's own SD equalises the contributions; it
    # is not centered, which would break the sparse non-negative structure.
    X = X / (X.std(axis=1, keepdims=True) + 1e-12)

    # 2. population-activity threshold via circular-shift shuffles ------------
    pop_activity = X.sum(axis=0)                                          # (n_bins,)
    shuffle_stds = np.empty(n_shuffles_bin)
    for s in range(n_shuffles_bin):
        shifts = rng.integers(1, max(2, n_bins), size=N)
        shuffle_stds[s] = _shift_rows(X, shifts).sum(axis=0).std()
    # published behaviour: mean + one representative null SD (~1 SD, permissive)
    bin_thresh   = float(pop_activity.mean() + np.percentile(shuffle_stds, bin_sig_percentile))
    _thresh_desc = f"mean+p{bin_sig_percentile:g}(SD_null)"
    sig_bins = np.where(pop_activity > bin_thresh)[0]
    n_sig = len(sig_bins)
    sig_frac = n_sig / max(1, n_bins)
    print(f"[detect_ensembles_ecker] N={N}  n_bins={n_bins} (bin_frames={bin_frames})  "
          f"{n_sig} bins pass significance threshold ({bin_thresh:.3f}, {_thresh_desc}) "
          f"= {sig_frac:.1%} of bins")
    if sig_frac > 0.10:
        print(f"[detect_ensembles_ecker] WARNING: {sig_frac:.1%} of bins are 'network "
              f"events'. The source papers see a few percent; a fraction this high "
              f"usually means the threshold is tracking a slow stimulus/population "
              f"envelope rather than discrete synchronous events, which lets one "
              f"cluster absorb most of the population.")

    lo, hi = k_range
    hi = min(hi, n_sig - 1) if n_sig > 1 else 0
    if n_sig < max(2, lo) or hi < lo:
        print(f"[detect_ensembles_ecker] too few significant bins ({n_sig}) "
              f"to scan k_range={k_range} — returning K=0")
        return _empty_result(n_bins=n_bins, n_sig_bins=n_sig, k_scanned=list(k_range),
                              davies_bouldin_best=None, n_candidate_clusters=0)

    # 3. cosine similarity + Ward-linkage clustering, scanning k for min DB ----
    activation_vectors = X[:, sig_bins].T                                 # (n_sig, N)
    norms = np.linalg.norm(activation_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vecs = activation_vectors / norms
    sim  = unit_vecs @ unit_vecs.T
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    Z_link = linkage(squareform(dist, checks=False), method='ward')

    # DB is scored on `unit_vecs`, not on the raw activation vectors: the Ward tree
    # was built in cosine geometry, where ‖x−y‖² = 2(1−cos) for unit vectors, so
    # scoring the unnormalised vectors would select k under a different metric than
    # the one that produced the candidate partitions.
    best_k, best_db, best_labels = None, np.inf, None
    db_curve = {}
    for k in range(lo, hi + 1):
        labels = fcluster(Z_link, k, criterion='maxclust')
        if len(np.unique(labels)) < 2:
            continue
        db = float(davies_bouldin_score(unit_vecs, labels))
        db_curve[int(k)] = db
        if db < best_db:
            best_db, best_k, best_labels = db, k, labels

    if best_labels is None:
        print(f"[detect_ensembles_ecker] no valid clustering found in k_range={k_range} "
              f"— returning K=0")
        return _empty_result(n_bins=n_bins, n_sig_bins=n_sig, k_scanned=list(k_range),
                              davies_bouldin_best=None, n_candidate_clusters=0)

    print(f"[detect_ensembles_ecker] scanned k={lo}..{hi}  best k={best_k}  "
          f"Davies-Bouldin={best_db:.3f}")
    if best_k in (lo, hi):
        print(f"[detect_ensembles_ecker] WARNING: best k={best_k} sits on the edge of "
              f"k_range={k_range} — the scan was fenced in, not converged. Davies-Bouldin "
              f"only ranks partitions relative to each other and never reports 'no "
              f"clusters', so widen k_range and inspect "
              f"population_meta['davies_bouldin_curve'].")

    # 4-5. per-cluster activation sequence, membership, co-firing criterion ---
    # NOTE on the null: circularly shifting neuron i's trace by lag L and shifting
    # cluster k's activation sequence by -L give the *same* circular cross-
    # correlation, so either side may be shifted. Shifting the (n_bins, K) sequence
    # matrix instead of the (N, n_bins) trace matrix lets the neuron traces be
    # centered and L2-normalised once outside the loop, after which each shuffle is
    # a single matmul — ~5x faster at N=56, T=32k. All neurons then share a lag
    # within a shuffle, which is harmless: every (i, k) test uses only its own
    # marginal null, and Bonferroni does not assume independence across tests.
    X_test = X

    with np.errstate(invalid='ignore'):
        pop_corr = np.corrcoef(X_test)
    iu = np.triu_indices(N, k=1)
    whole_pop_mean_corr = float(np.nanmean(pop_corr[iu])) if len(iu[0]) else 0.0

    cluster_ids = np.unique(best_labels)
    K_candidate = len(cluster_ids)

    seq_matrix = np.zeros((n_bins, K_candidate), dtype=np.float32)
    for j, cid in enumerate(cluster_ids):
        seq_matrix[sig_bins[best_labels == cid], j] = 1.0

    corr_all = _multi_corr(X_test, seq_matrix)                            # (N, K_candidate)

    # Membership test: an empirical p-value from exceedance counts — no percentile
    # interpolation, and the multiplicity correction uses the true number of tests,
    # N * K_candidate, which is only knowable here (the caller cannot know K_candidate
    # in advance).
    Xn_test = _prenorm_rows(X_test)                                       # (N, n_bins)
    n_tests = N * K_candidate
    exceed = np.zeros((N, K_candidate), dtype=np.int32)
    for s in range(n_shuffles_member):
        null_c = Xn_test @ _prenorm_cols(_shift_cols(seq_matrix,
                                                      rng.integers(1, max(2, n_bins),
                                                                   size=K_candidate)))
        exceed += (null_c >= corr_all)
    member_p = (1.0 + exceed) / (n_shuffles_member + 1.0)              # (N, K_candidate)
    p_floor = 1.0 / (n_shuffles_member + 1)

    # Benjamini-Hochberg. Bonferroni would control the family-wise error rate,
    # which is the wrong target here: we *expect* many true members, and at
    # K_candidate=20 the per-test threshold becomes alpha/1120 = 4.5e-5, requiring
    # >200k shuffles just to be reachable. BH controls the false discovery rate
    # instead — the fraction of called members that are spurious — which is what
    # actually matters for a downstream group statistic, and its threshold for rank
    # i is alpha*i/m, so it stays well clear of the 1/(n+1) p-value floor.
    flat  = member_p.ravel()
    order = np.argsort(flat)
    ranks = np.arange(1, n_tests + 1)
    passed = flat[order] <= member_alpha * ranks / n_tests
    n_pass = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    p_thresh = float(flat[order][n_pass - 1]) if n_pass else 0.0
    is_member_all = (member_p <= p_thresh) if n_pass else np.zeros_like(member_p, bool)
    print(f"[detect_ensembles_ecker] membership: Benjamini-Hochberg FDR "
          f"q={member_alpha} over {n_tests} tests → p <= {p_thresh:.2e} "
          f"({n_pass} of {n_tests} tests significant, {n_shuffles_member} shuffles)")

    if p_floor > p_thresh and p_thresh > 0:
        n_needed = int(np.ceil(10.0 / p_thresh))
        print(f"[detect_ensembles_ecker] WARNING: n_shuffles_member="
              f"{n_shuffles_member} floors the empirical p-value at {p_floor:.2e}, "
              f"at or above the threshold {p_thresh:.2e} — membership calls are "
              f"resolution-limited. Use n_shuffles_member>={n_needed}.")

    max_members = max_member_frac * N   # size cutoff; max_member_frac=1.0 → never triggers
    kept_V, kept_M, kept_S, kept_dropped, size_dropped = [], [], [], 0, 0
    for j in range(K_candidate):
        is_member = is_member_all[:, j]
        if is_member.sum() < 2:
            kept_dropped += 1
            continue

        member_idx = np.where(is_member)[0]
        miu = np.triu_indices(len(member_idx), k=1)
        member_mean_corr = (float(np.nanmean(pop_corr[np.ix_(member_idx, member_idx)][miu]))
                             if len(miu[0]) else -np.inf)

        if not (member_mean_corr > whole_pop_mean_corr):
            kept_dropped += 1
            continue

        # post-detection size cutoff: a co-firing survivor that still recruits most
        # of the population is a non-specific / global mode, not a targeted assembly.
        if is_member.sum() > max_members:
            size_dropped += 1
            continue

        kept_V.append(corr_all[:, j].astype(np.float32))
        kept_M.append(is_member.astype(np.int8))
        kept_S.append(seq_matrix[:, j])

    K = len(kept_V)
    survived_cofire = K_candidate - kept_dropped
    size_msg = (f"; {size_dropped} then dropped as non-specific (>{max_member_frac:.0%} of N)"
                if size_dropped else "")
    print(f"[detect_ensembles_ecker] {survived_cofire}/{K_candidate} candidate clusters survived "
          f"the co-firing criterion (pop. mean r={whole_pop_mean_corr:.3f}){size_msg} "
          f"→ K={K} functional assemblies  ({kept_dropped} dropped)")

    if K == 0:
        return _empty_result(n_bins=n_bins, n_sig_bins=n_sig, sig_bin_frac=float(sig_frac),
                              davies_bouldin_best=float(best_db),
                              davies_bouldin_curve=db_curve,
                              n_candidate_clusters=K_candidate,
                              member_p_threshold=(float(p_thresh)
                                                  if p_thresh is not None else None),
                              whole_pop_mean_corr=whole_pop_mean_corr,
                              n_size_dropped=size_dropped)

    V = np.stack(kept_V, axis=1).astype(np.float32)                       # (N, K)
    M = np.stack(kept_M, axis=1).astype(np.int8)                          # (N, K)
    S_bins = np.stack(kept_S, axis=1).astype(np.float32)                  # (n_bins, K)

    # broadcast bin-level activation sequences back out to the original T frames
    if bin_frames > 1:
        S = np.repeat(S_bins, bin_frames, axis=0)
        if S.shape[0] < T:
            S = np.vstack([S, np.zeros((T - S.shape[0], K), dtype=np.float32)])
        else:
            S = S[:T]
    else:
        S = S_bins

    # ── build DataFrames (identical construction to `detect_ensembles`) ───────
    root_ids_arr = np.asarray(list(root_ids), dtype=np.int64)
    rows = []
    for k in range(K):
        for i, rid in enumerate(root_ids_arr):
            rows.append({'k': k, 'root_id': int(rid),
                         'weight': float(V[i, k]),
                         'is_member': bool(M[i, k])})
    ensembles_df = pd.DataFrame(rows, columns=['k', 'root_id', 'weight', 'is_member'])

    meta_rows = []
    for k in range(K):
        w = V[:, k]
        meta_rows.append({
            'k':           k,
            'n_members':   int(M[:, k].sum()),
            'weight_mean': float(w.mean()),
            'weight_std':  float(w.std()),
            'weight_max':  float(w.max()),
        })
    ensemble_meta_df = pd.DataFrame(meta_rows)

    population_meta = {
        'K':             K,
        'K_cs':          None,
        'lambda_max_mp': None,
        'lambda_cs':     None,
        'lambda_gap':    None,
        'N':             N,
        'T':             T,
        'trace_type':    trace_type,
        'method':            'ecker',
        'bin_frames':         bin_frames,
        'n_bins':             n_bins,
        'n_sig_bins':         n_sig,
        'sig_bin_frac':       float(sig_frac),
        'k_scanned':          list(k_range),
        'davies_bouldin_best': float(best_db),
        'davies_bouldin_curve': db_curve,
        'n_candidate_clusters': K_candidate,
        'n_shuffles_member':  int(n_shuffles_member),
        'member_alpha':       float(member_alpha),
        'member_p_threshold': (float(p_thresh) if p_thresh is not None else None),
        'whole_pop_mean_corr': whole_pop_mean_corr,
        'max_member_frac':    float(max_member_frac),
        'n_size_dropped':     size_dropped,
    }

    return {
        'ensembles_df':     ensembles_df,
        'ensemble_meta_df': ensemble_meta_df,
        'population_meta':  population_meta,
        'V': V, 'M': M, 'S': S,
    }


def detect_ensembles_dispatch(traces, root_ids, method='lds', **kwargs):
    """
    Single entry point so the notebook flips one flag instead of importing/calling
    a different function. `method` picks the algorithm; everything else in
    `kwargs` is passed straight through to it *unchanged* — the two algorithms
    take different hyperparameters (see their own docstrings), so this wrapper
    does not translate or share kwargs between them. What IS identical between
    them is the (traces, root_ids) input contract and the returned dict schema
    (ensembles_df, ensemble_meta_df, population_meta, V, M, S), which is what
    lets the rest of the notebook stay unchanged.

    Parameters
    ----------
    method : 'lds'   → detect_ensembles_lds(...)
             'ecker' → detect_ensembles_ecker(...)
    """
    if method == 'lds':
        return detect_ensembles_lds(traces, root_ids, **kwargs)
    elif method == 'ecker':
        return detect_ensembles_ecker(traces, root_ids, **kwargs)
    raise ValueError(f"unknown method {method!r}; expected 'lds' or 'ecker'")


def build_activity_matrix(func_data, session, scan_idx, h5_stim_path,
                          trial_residual=False):
    """
    Build (N, T) trace matrix for one (session, scan_idx).

    Every detector downstream treats column `t` as one instant, which is what the
    loader's ms_delay interpolation guarantees: raw traces are on per-unit clocks up
    to one full frame (~159 ms) apart, and that offset is structured by imaging
    depth, so unaligned input would bias ensembles towards same-depth membership.

    Parameters
    ----------
    func_data      : ex_func_data dict keyed by str root_id, from
                     `activity_utils.load_ex_functional_data_by_root_id`
    session        : int
    scan_idx       : int
    h5_stim_path   : path to microns_per_scan_stimuli.h5. Only oracle-clip frames
                     are kept (condition_hashes with ≥5 repeats).
    trial_residual : replace each neuron's trace with
                     its per-clip TRIAL RESIDUAL: within each oracle clip, subtract that
                     clip's average over its ~10 repeats from every repeat. See
                     "Trial-residual mode" below.

    Trial-residual mode
    -------------------
    Every detector's null (the ICA detector's circular shift, the clustering
    detector's bin shuffles) destroys the
    alignment between a neuron's trace and the stimulus. On raw oracle-clip traces that
    makes the test ask "do these neurons fire together more than two *unrelated* neurons
    would?" — and two cells watching the same movie pass that trivially, without ever
    exchanging a spike. Measured on this dataset, co-members of raw-oracle ensembles are
    elevated far more in signal (co-tuning) correlation than in noise correlation
    (lds/oracle: +0.275 vs +0.093), i.e. raw-oracle ensembles are mostly tuning groups.

    Subtracting each clip's trial average removes exactly the stimulus-locked component
    and keeps the trial-to-trial fluctuation around it. Co-fluctuation in the residual
    is not explainable by shared tuning, so it is what a connectome analysis should be
    asked to predict. This is only constructible on repeated stimuli — hence oracle-only.

    Two consequences worth knowing:
      * The residual is SIGNED, while deconvolved spike traces are non-negative (up to
        float noise: min ~-1e-8 against a max of ~900 on s6/sc7). LdS
        z-scores its input so it is unaffected; for the clustering detector a
        "population event" now
        means "the population collectively exceeded its typical response to this frame
        of this clip" rather than "the population fired", which is the intended reading.
      * Columns are ordered (clip, repeat, frame-within-clip) and concatenated, so
        consecutive columns can straddle a repeat or clip boundary. The circular-shift
        surrogate treats the concatenation as one continuous series, which is the same
        approximation the raw-oracle path already makes at clip boundaries.

    Returns
    -------
    Z          (N, T) float32
    root_ids   sorted list of int root_ids (length N)
    fps        float
    """
    trace_key = 'spike_trace'
    traces, fps_seen = {}, None
    for rid_str, units in func_data.items():
        for (sess, scan, _uid), data in units.items():
            if sess != session or scan != scan_idx or trace_key not in data:
                continue
            tr = np.asarray(data[trace_key], dtype=np.float32)
            if fps_seen is None:
                fps_seen = float(data['fps'])
            # one unit per neuron per scan (`break` below). A collision here means
            # this root_id pooled units from >1 nucleus — a merge-error segment,
            # i.e. genuinely different cells — so averaging them would blend two
            # cells into one row. Take the first and say so.
            rid = int(rid_str)
            if rid in traces:
                print(f'[build_activity_matrix] WARNING: root_id {rid} has a second unit '
                      f'in this scan (multi-nucleus segment?) — keeping the first only')
            else:
                traces[rid] = tr
            break
    if not traces:
        raise RuntimeError(f'No traces for session={session}, scan_idx={scan_idx}')
    root_ids_out = sorted(traces.keys())
    Z = np.vstack([traces[r] for r in root_ids_out])
    from activity_utils import get_oracle_condition_hashes, load_scan_trials_df
    oracle_windows = get_oracle_condition_hashes(
        load_scan_trials_df(h5_stim_path, session, scan_idx))
    if not oracle_windows:
        raise ValueError(f'No oracle clips for session={session} scan={scan_idx}')

    if trial_residual:
        # Same per-clip truncation and (clip, repeat, frame) ordering as
        # activity_utils.trial_residual_vector, applied to all N rows at once so
        # every neuron's columns stay index-aligned — that alignment is what makes
        # a cross-neuron correlation on Z a noise correlation.
        parts, n_rep = [], []
        for _ch, windows in oracle_windows:
            min_len = min(int(e) - int(s) for (s, e) in windows)
            if min_len <= 0:
                continue
            rep = np.stack([Z[:, int(s):int(s) + min_len] for (s, _e) in windows],
                           axis=1)                       # (N, n_repeats, min_len)
            parts.append((rep - rep.mean(axis=1, keepdims=True)).reshape(Z.shape[0], -1))
            n_rep.append(len(windows))
        if not parts:
            raise RuntimeError('Oracle trial-residual left no usable clips')
        Z = np.concatenate(parts, axis=1).astype(np.float32)
        if min(n_rep) < 3:
            print(f'[build_activity_matrix] WARNING: s{session} sc{scan_idx} has a '
                  f'clip with only {min(n_rep)} repeats — the trial average is a '
                  f'poor estimate and the residual keeps part of the stimulus response')
        if Z.shape[1] < 100:
            raise RuntimeError(f'Oracle trial-residual left only {Z.shape[1]} frames')
    else:
        keep = np.zeros(Z.shape[1], dtype=bool)
        for _ch, windows in oracle_windows:
            for s, e in windows:
                keep[int(s):int(e)] = True
        if keep.sum() < 100:
            raise RuntimeError(f'Oracle filter left only {keep.sum()} frames')
        Z = Z[:, keep]

    return Z, root_ids_out, fps_seen


def build_ensemble_members(res, rec_root_ids, neurons_df):
    """Rank ensembles by activation energy and build per-ensemble member DataFrames.

    Parameters
    ----------
    res          : result dict from detect_ensembles_dispatch (keys V, M, S, population_meta, ensemble_meta_df)
    rec_root_ids : list of int root_ids length N (same order as res['V'] rows)
    neurons_df   : DataFrame with root_id index and clf_type/cell_type/mtype columns

    Ensembles left with fewer than two members are dropped: a group of one has no
    internal wiring and no shared input to measure.

    Returns
    -------
    ensemble_members : dict  k → DataFrame[root_id, weight, clf_type, cell_type, mtype]
    k_list           : list of ensemble indices kept
    K                : len(k_list)
    """
    activation_energy = (res['S'] ** 2).sum(axis=0)
    res['ensemble_meta_df']['activation_energy'] = activation_energy
    res['ensemble_meta_df']['rank'] = (-activation_energy).argsort().argsort()

    K_full = res['population_meta']['K']
    k_list = list(range(K_full))
    K = len(k_list)

    V = res['V']
    M = res['M']
    n_with_membership = int(M[:, k_list].any(axis=1).sum())
    print(f'Spike ensembles: K={K} (of {K_full} detected)  |  '
          f'{n_with_membership} of {len(rec_root_ids)} neurons have ≥1 membership')

    if K == 0:
        raise RuntimeError('No ensembles detected — try a different scan or check the traces.')

    ct_lookup = neurons_df.set_index('root_id')[['clf_type', 'cell_type', 'mtype']]
    ensemble_members = {}
    for k in k_list:
        member_ids = [rec_root_ids[i] for i in range(len(rec_root_ids)) if M[i, k]]
        df_k = ct_lookup.reindex(member_ids).reset_index()
        df_k.insert(1, 'weight', [float(V[rec_root_ids.index(r), k]) for r in member_ids])
        df_k = df_k.sort_values('weight', ascending=False).reset_index(drop=True)
        ensemble_members[k] = df_k
        print(f'  k={k}: {len(member_ids)} members')

    singleton_ks = [k for k, df_k in ensemble_members.items() if len(df_k) < 2]
    if singleton_ks:
        print(f'  dropping {len(singleton_ks)} singleton ensemble(s): {singleton_ks}')
        for k in singleton_ks:
            del ensemble_members[k]
        k_list = [k for k in k_list if k not in set(singleton_ks)]
    K = len(k_list)

    return ensemble_members, k_list, K


def build_ex_pool_arrays(spine_df, all_ex_root_ids):
    """Build integer-index arrays over the ex-neuron pool for fast edge counting.

    Built once and reused by every control draw and every bootstrap. Index arrays
    avoid per-group set operations and speed up the rejection sampling by ~100×.

    Parameters
    ----------
    spine_df         : full spine table (pre_pt_root_id, post_pt_root_id, tag columns)
    all_ex_root_ids  : 1-D array of all excitatory root_ids (sets the index ordering)

    Returns
    -------
    dict with keys n_ex, rid_to_idx, edge_pre, edge_post,
        syn_pre_idx, syn_post_idx, syn_is_spine
    """
    all_ex_set  = set(all_ex_root_ids)
    ex_edges_df = (spine_df[spine_df['pre_pt_root_id'].isin(all_ex_set) &
                            spine_df['post_pt_root_id'].isin(all_ex_set)]
                   .drop_duplicates(['pre_pt_root_id', 'post_pt_root_id']))
    rid_to_idx = {rid: i for i, rid in enumerate(all_ex_root_ids)}
    edge_pre   = ex_edges_df['pre_pt_root_id'].map(rid_to_idx).values
    edge_post  = ex_edges_df['post_pt_root_id'].map(rid_to_idx).values
    n_ex       = len(all_ex_root_ids)
    print(f'Ex-pool edge list: {len(ex_edges_df):,} unique pre→post pairs among {n_ex} ex neurons')

    ex_syn_internal = spine_df[spine_df['pre_pt_root_id'].isin(all_ex_set) &
                               spine_df['post_pt_root_id'].isin(all_ex_set)]
    syn_pre_idx  = ex_syn_internal['pre_pt_root_id'].map(rid_to_idx).values
    syn_post_idx = ex_syn_internal['post_pt_root_id'].map(rid_to_idx).values
    syn_is_spine = (ex_syn_internal['tag'] == 'spine').values

    return {
        'n_ex': n_ex, 'rid_to_idx': rid_to_idx,
        'edge_pre': edge_pre, 'edge_post': edge_post,
        'syn_pre_idx': syn_pre_idx, 'syn_post_idx': syn_post_idx,
        'syn_is_spine': syn_is_spine,
    }


def run_spine_targeting_bootstrap(ensembles_df, control_df, label_col, n_null=1000, seed=0):
    """Paired bootstrap null for pooled internal spine targeting (figure 6C).

    For each null replicate, draws one matched control per ensemble (independently)
    and pools spine/synapse counts the same way as the real ensembles.

    Parameters
    ----------
    ensembles_df : DataFrame with columns 'n_spine', 'n_synapses', and label_col
    control_df   : DataFrame with columns label_col, 'n_spine_internal',
                   'n_synapses_internal' (output of sample_matched_controls)
    label_col    : grouping key — 'k' for single-scan, 'scan_k' for multi-scan
    n_null       : bootstrap replicates (default 1000)
    seed         : RNG seed (default 0)

    Returns
    -------
    dict with keys: p_obs, valid_null (array), p_emp, star, n_spine_obs, n_syn_obs,
        usable_labels, usable_mask
    """
    from stats_corr import p_to_stars

    p_obs = ensembles_df['n_spine'].sum() / ensembles_df['n_synapses'].sum()

    per_label_pools = {
        lbl: control_df.loc[control_df[label_col] == lbl,
                             ['n_spine_internal', 'n_synapses_internal']].to_numpy()
        for lbl in ensembles_df[label_col]
    }
    empty_labels = [lbl for lbl, arr in per_label_pools.items() if len(arr) == 0]
    if empty_labels:
        print(f'WARNING: ensembles with no sampled controls (skipped): {empty_labels}')
    usable_labels = [lbl for lbl in per_label_pools if len(per_label_pools[lbl]) > 0]

    rng    = np.random.default_rng(seed)
    null_p = np.empty(n_null)
    for r in range(n_null):
        spine_sum, total_sum = 0, 0
        for lbl in usable_labels:
            arr  = per_label_pools[lbl]
            draw = arr[rng.integers(0, len(arr))]
            spine_sum += draw[0]
            total_sum += draw[1]
        null_p[r] = spine_sum / total_sum if total_sum > 0 else np.nan

    valid_null = null_p[~np.isnan(null_p)]
    p_emp      = (1 + np.sum(valid_null >= p_obs)) / (1 + len(valid_null))
    star       = p_to_stars(p_emp)

    usable_mask   = ensembles_df[label_col].isin(usable_labels)
    n_spine_obs   = int(ensembles_df.loc[usable_mask, 'n_spine'].sum())
    n_syn_obs     = int(ensembles_df.loc[usable_mask, 'n_synapses'].sum())

    print(f'Spine targeting — pooled internal spine fraction  ({usable_mask.sum()} ensembles)')
    print(f'  Observed: {n_spine_obs}/{n_syn_obs} = {p_obs:.3f}')
    print(f'  Null (paired bootstrap, n={len(valid_null)}): '
          f'mean={np.nanmean(valid_null):.3f}  '
          f'[{np.nanpercentile(valid_null, 2.5):.3f}, {np.nanpercentile(valid_null, 97.5):.3f}]')
    print(f'  Empirical p (one-sided, obs ≥ null) = {p_emp:.4f}  {star}')

    return {
        'p_obs': p_obs, 'valid_null': valid_null, 'p_emp': p_emp, 'star': star,
        'n_spine_obs': n_spine_obs, 'n_syn_obs': n_syn_obs,
        'usable_labels': usable_labels, 'usable_mask': usable_mask,
    }


def _mean_pairwise_dist(idx, pos_array):
    """Mean Euclidean soma distance (µm) over ALL pairs in a group of pool indices."""
    P = pos_array[np.asarray(idx, dtype=int)]
    P = P[~np.isnan(P).any(axis=1)]
    if len(P) < 2:
        return np.nan
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return float(d[np.triu_indices(len(P), 1)].mean())


def _batch_size(n_size):
    """How many candidate groups to test at once, for a group of `n_size` neurons.

    The distance test builds an (m, n_size, n_size) array, so the batch shrinks as the
    groups grow to hold peak memory roughly constant (a few tens of MB).
    """
    return int(max(1024, min(65_536, 4_000_000 // max(n_size * n_size, 1))))


def _draw_candidate_groups(rng, draw_from, n_size, batch):
    """`batch` candidate groups of distinct pool indices, as an (m, n_size) array.

    Drawn with replacement; rows that repeat an index are dropped, so m <= batch. That
    leaves a uniform draw over distinct-index groups — every ordered tuple of distinct
    indices is equally likely and each group has the same n_size! of them — which is
    what `rng.choice(..., replace=False)` gives one group at a time, only vectorised.
    With groups of 2-10 drawn from a pool of ~1,000, fewer than 5% of rows repeat.
    """
    pos = rng.integers(0, len(draw_from), size=(batch, n_size))
    if n_size > 1:
        ordered = np.sort(pos, axis=1)
        pos = pos[(np.diff(ordered, axis=1) != 0).all(axis=1)]
    return draw_from[pos]


def _mean_pairwise_dist_batch(groups, pos_array):
    """`_mean_pairwise_dist` over a whole (m, k) batch → (m,) means, NaN where < 2 somata.

    Somata with no recorded position are dropped pair-wise, exactly as the per-group
    version drops them before taking all pairs of what is left.
    """
    P = pos_array[groups]                                   # (m, k, 3)
    valid = ~np.isnan(P).any(axis=2)                        # (m, k)
    k = groups.shape[1]
    iu, ju = np.triu_indices(k, 1)
    diff = P[:, iu, :] - P[:, ju, :]                        # (m, n_pairs, 3)
    d = np.sqrt((diff * diff).sum(axis=-1))                 # (m, n_pairs)
    pair_ok = valid[:, iu] & valid[:, ju]
    n_pairs = pair_ok.sum(axis=1)
    total = np.where(pair_ok, np.nan_to_num(d), 0.0).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(n_pairs > 0, total / n_pairs, np.nan)
    return out


def compute_avg_dist_targets(label_to_members, pool, pos_array, dist_tol, abs_floor_um):
    """Per-label soma-distance target (µm) for distance-matched null sampling.

    The target is the group's **compactness**: mean soma-soma distance over ALL member
    pairs, n(n-1)/2 of them. It is always defined for n >= 2, and it is the quantity that
    actually drives the shared-input and connection-density metrics — ensembles are
    0.63-0.84x as spread out as size-matched random groups, and nearby neurons both
    connect more and share inputs more (local interneurons especially) for purely
    geometric reasons.

    An earlier mode matched only the mean distance over internally CONNECTED pairs. It is
    undefined (NaN -> constraint silently skipped) for any group with no internal synapse,
    which is >50% of detected ensembles here, so it failed to bind on exactly the
    ensembles the shared-input measure depends on, so it is not offered.

    Returns dict {label: float} — labels with missing positions get NaN. Prints a summary.
    """
    rid_to_idx = pool['rid_to_idx']
    avg_dist_targets = {
        lbl: _mean_pairwise_dist([rid_to_idx[r] for r in members if r in rid_to_idx],
                                 pos_array)
        for lbl, members in label_to_members.items()
    }
    valid_d = [d for d in avg_dist_targets.values() if not np.isnan(d) and d > 0]
    n_skip  = sum(1 for d in avg_dist_targets.values() if np.isnan(d) or d == 0)
    if valid_d:
        print(f'Distance matching (all-pair compactness): avg soma distance — '
              f'mean={np.mean(valid_d):.1f} µm, '
              f'range=[{np.min(valid_d):.1f}, {np.max(valid_d):.1f}] µm, '
              f'tol=±{dist_tol*100:.0f}% (floor ±{abs_floor_um:.0f} µm)'
              + (f'  [{n_skip} ensembles skipped: zero/NaN dist]' if n_skip else ''))
    return avg_dist_targets


def build_pool_pos_arrays(pool, verbose=True):
    """Soma positions indexed to the pool, for the distance-matched nulls.

    Returns an (n_ex, 3) array; rows for neurons with no position stay NaN.
    """
    from utils import load_neuron_position_transformed
    neurons_pos = load_neuron_position_transformed(use_column_manual_ct=True)
    pos_lookup  = neurons_pos.set_index('root_id')[
        ['pt_position_xt', 'pt_position_yt', 'pt_position_zt']]

    arr = np.full((pool['n_ex'], 3), np.nan)
    for rid, idx in pool['rid_to_idx'].items():
        if rid in pos_lookup.index:
            arr[idx] = pos_lookup.loc[rid].values
    if verbose:
        n_have = pool['n_ex'] - int(np.isnan(arr[:, 0]).sum())
        print(f'pos_array: {n_have}/{pool["n_ex"]} neurons have positions')
    return arr


def build_control_eligible_idx(pool, label_to_members, verbose=True):
    """Pool indices a control group may be drawn from: everything MINUS every ensemble member.

    Returns (eligible_idx, eligible_root_ids).

    Why this exists — null contamination
    ------------------------------------
    Controls used to be drawn from the whole pool, with only the *exact* real group
    rejected. Nothing stopped one ensemble's members from landing in another
    ensemble's null. With ~476 recorded neurons, 50+ ensembles and 1000 draws each,
    that is not a rare event: a control for ensemble E2 routinely contains a pair
    {a, b} that are themselves co-members of E1. That pair's synapse then counts on
    BOTH sides of the pooled comparison — as ensemble signal via E1, and as null via
    E2 — which drags the null toward the observed value and biases every pooled
    metric toward the null hypothesis.

    The same leak occurs *within* an ensemble: rejecting only the exact member set
    still admits every proper subset, so a control for {a,b,c,d} could be {a,b,c,x}.

    Barring all members from all nulls makes the observed and null populations
    disjoint by construction, so no synapse can be counted on both sides.

    Parameters
    ----------
    pool              : dict from build_ex_pool_arrays — defines the index space
    label_to_members  : dict label -> list[root_id]; the union over all labels
                        (all ensembles, all scans) is what gets excluded
    """
    rid_to_idx  = pool['rid_to_idx']
    member_rids = {rid for members in label_to_members.values() for rid in members}
    member_idx  = {rid_to_idx[r] for r in member_rids if r in rid_to_idx}

    candidate_idx = np.arange(pool['n_ex'])
    eligible_idx = np.array([i for i in candidate_idx if i not in member_idx], dtype=int)
    idx_to_rid   = {v: k for k, v in rid_to_idx.items()}
    eligible_root_ids = np.array([idx_to_rid[i] for i in eligible_idx])

    if verbose:
        print(f'Control eligibility: {len(eligible_idx)} of {len(candidate_idx)} pool neurons  '
              f'(excluded {len(member_idx)} neurons that are a member of ≥1 ensemble '
              f'in any scan, out of {len(member_rids)} distinct member root_ids)')
    return eligible_idx, eligible_root_ids


def build_pool_indegree_array(pool, syn_df, all_ex_root_ids, verbose=True):
    """In-degree array for degree-matched null sampling — (pool['n_ex'], 2) int.

    Column 0 = number of distinct E pre-synaptic partners, column 1 = distinct I
    partners, both counted over the within-column `syn_df` and indexed by pool index.
    Matching both columns at once keeps the E and I shared-input counts honest under a
    single sampling pass, and since E and I are disjoint, matching both within ±tol also
    bounds the pooled E+I in-degree within ±tol.

    Why this matters: shared input is an INTERSECTION, so if each member has k input
    partners the expected intersection over a group of n scales roughly like k**n.
    Ensemble members carry a median E+I in-degree of ~61 against a pool median of ~43 —
    only 1.4x — but 1.4**4 is ~4x. Measured on this dataset with no ensemble structure
    involved at all, random groups drawn from the high-in-degree third of the pool show
    5.4x (size 3) to 10.3x (size 5) more shared input than groups drawn from the low
    third. That is larger than the shared-input enrichment being attributed to ensembles,
    so an unmatched null mostly measures how well-connected the members happen to be.
    """
    ex_set  = set(all_ex_root_ids)
    sub_all = syn_df[syn_df['post_id'].isin(ex_set)]
    deg_ee  = sub_all[sub_all['pre_clf_type'] == 'E'].groupby('post_id')['pre_id'].nunique().to_dict()
    deg_ii  = sub_all[sub_all['pre_clf_type'] == 'I'].groupby('post_id')['pre_id'].nunique().to_dict()

    arr = np.zeros((pool['n_ex'], 2), dtype=np.int64)
    for rid, idx in pool['rid_to_idx'].items():
        arr[idx] = (deg_ee.get(rid, 0), deg_ii.get(rid, 0))
    if verbose:
        print(f'in-degree array: E median={np.median(arr[:, 0]):.0f}  '
              f'I median={np.median(arr[:, 1]):.0f}  over {pool["n_ex"]} pool neurons')
    return arr


def run_connection_probability_bootstrap(ensembles_df, label_col, syn_df, all_ex_root_ids, pool,
                           n_controls=1000, seed=0, max_attempts=500_000, n_boot=None,
                           label_to_members=None, pos_array=None, dist_tol=0.2,
                           abs_floor_um=15.0, eligible_idx=None):
    """Connection probability among ensemble members, against the E→E baseline (figure 6B).

    Observed: pooled n_synapses (incl. multi-contacts) / pooled N*(N-1) across ensembles.
    Null: size-only matched random groups (no edge constraint) — unconstrained random
          groups converge to the global E→E probability, giving a meaningful baseline.
          If label_to_members and pos_array are provided, additionally matches avg soma
          distance of connected pairs within dist_tol (size + distance matched null).
    Baseline: global E→E synapse rate across all micro-column ex neurons — computed over
          `all_ex_root_ids`, so pass the eligible (ensemble-free) root_ids when
          `eligible_idx` is set, to keep the baseline describing the pool the null
          actually draws from.
    eligible_idx: optional restriction on which pool indices controls may use — see
          build_control_eligible_idx.
    """
    from stats_corr import p_to_stars

    ex_set   = set(all_ex_root_ids)
    ee_total = int(((syn_df['pre_id'].isin(ex_set)) & (syn_df['post_id'].isin(ex_set))).sum())
    N_ex     = len(all_ex_root_ids)
    p_global = ee_total / (N_ex * (N_ex - 1))
    print(f'Global E→E synapse probability: {ee_total} / {N_ex*(N_ex-1)} = {p_global:.6f}')

    ens = ensembles_df.copy()
    ens['n_possible'] = ens['n_members'] * (ens['n_members'] - 1)
    p_obs = ens['n_synapses'].sum() / ens['n_possible'].sum()

    match_label = ('size + distance' if (label_to_members is not None and pos_array is not None)
                   else 'size-only')
    print(f'Sampling {match_label} matched controls for the connection-probability null...')
    def _seed_fn(row, lc=label_col, s=seed):
        # crc32, not hash(): CPython randomises hash(str) per process.
        # See the module docstring.
        return int(s * 997 + zlib.crc32(str(row[lc]).encode()) % 100_000)

    if n_boot is None:
        n_boot = n_controls
    control_df = sample_matched_controls(
        ensembles_df, label_col=label_col, seed_fn=_seed_fn, pool=pool,
        n_controls=n_controls, max_attempts=max_attempts,
        match_edges=False, verbose_short_only=True,
        label_to_members=label_to_members, pos_array=pos_array,
        dist_tol=dist_tol, abs_floor_um=abs_floor_um, eligible_idx=eligible_idx,
    )

    n_members_map = ens.set_index(label_col)['n_members'].to_dict()
    ctrl = control_df.copy()
    ctrl['n_possible'] = ctrl[label_col].map(n_members_map) * (ctrl[label_col].map(n_members_map) - 1)

    per_label = {
        lbl: ctrl.loc[ctrl[label_col] == lbl, ['n_synapses_internal', 'n_possible']].to_numpy()
        for lbl in ens[label_col]
    }
    usable = [lbl for lbl in per_label if len(per_label[lbl]) > 0]

    rng    = np.random.default_rng(seed)
    null_p = np.empty(n_boot)
    for r in range(n_boot):
        syn_s, pos_s = 0, 0
        for lbl in usable:
            arr  = per_label[lbl]
            draw = arr[rng.integers(0, len(arr))]
            syn_s += draw[0]
            pos_s += draw[1]
        null_p[r] = syn_s / pos_s if pos_s > 0 else np.nan

    valid_null = null_p[~np.isnan(null_p)]
    p_emp      = (1 + np.sum(valid_null >= p_obs)) / (1 + len(valid_null))
    star       = p_to_stars(p_emp)

    n_syn_obs = int(ens.loc[ens[label_col].isin(usable), 'n_synapses'].sum())
    n_pos_obs = int(ens.loc[ens[label_col].isin(usable), 'n_possible'].sum())
    print(f'Connection probability  ({len(usable)} ensembles)')
    print(f'  Observed: {n_syn_obs}/{n_pos_obs} = {p_obs:.6f}')
    print(f'  Null ({match_label} matched, n={len(valid_null)}): '
          f'mean={np.nanmean(valid_null):.6f}  '
          f'[{np.nanpercentile(valid_null, 2.5):.6f}, {np.nanpercentile(valid_null, 97.5):.6f}]')
    print(f'  Empirical p (one-sided, obs ≥ null) = {p_emp:.4f}  {star}')

    # real_df / ctrl_df carry the per-label numerator+denominator behind the pooled
    # ratio. Everything above is a pooled summary over all ensembles at once; keeping
    # the per-label rows is what lets shared_input_by_size re-aggregate the same statistic
    # within one ensemble size without resampling the connectome.
    return {
        'p_obs': p_obs, 'p_global': p_global,
        'valid_null': valid_null, 'p_emp': p_emp, 'star': star,
        'real_df': ens[[label_col, 'n_synapses', 'n_possible']].copy(),
        'ctrl_df': ctrl[[label_col, 'n_synapses_internal', 'n_possible']].copy(),
        'label_col': label_col,
    }


def run_shared_input_bootstrap(label_to_members, all_ex_root_ids, syn_df,
                            ensembles_df, pool,
                            label_col='k', n_controls=1000, seed=0,
                            max_attempts=500_000, n_boot=None,
                            pos_array=None, dist_tol=0.2, abs_floor_um=15.0,
                            degree_array=None, degree_tol=0.2, eligible_idx=None,
                            syn_with_tags=None):
    """Shared presynaptic input to ensemble members — the intersection of their
    presynaptic partner sets (figures 6D and 6E).

    Parameters
    ----------
    label_to_members : dict  label → list[root_id]
        Single-scan: {k: list(ensemble_members[k]['root_id']) for k in k_list}
        Multi-scan:  ensemble_members_by_scank  (scan_k → list[root_id])
    ensembles_df : DataFrame with columns label_col, 'n_members', 'n_edges'
    pool         : dict from build_ex_pool_arrays (needs 'n_ex', 'edge_pre',
                   'edge_post', 'rid_to_idx')
    pos_array    : float array (n_ex, 3) soma positions in µm, indexed by pool idx.
                   When provided, controls must also match the ensemble's all-pair
                   soma compactness — the control that matters most here, since the
                   shared input is overwhelmingly inhibitory and local interneurons
                   contact whatever sits inside their arbor.
    dist_tol     : fractional tolerance (default 0.2 = ±20%).
    abs_floor_um : absolute tolerance floor in µm (default 15.0).
    degree_array : int array (n_ex, 2) from build_pool_indegree_array. When provided,
                   controls must also match the ensemble's SUMMED in-degree (E and I,
                   both within degree_tol). Strongly recommended — see that function's
                   docstring for why an unmatched null mostly measures in-degree.
    degree_tol   : fractional tolerance for summed in-degree (default 0.2 = ±20%).
    eligible_idx : optional restriction on which pool indices controls may be drawn
                   from — see build_control_eligible_idx. Note `all_ex_root_ids` must
                   still span the FULL column: it defines which post-synaptic neurons
                   have a pre-synaptic set at all, so restricting it would zero out
                   the real ensembles' own shared-input counts.
    syn_with_tags : optional spine-tagged synapse table (syn_df rows that appear in the
                   spine table, with a 'tag' column). When given, each group also gets
                   a spine fraction *for the shared input specifically*: of the synapses
                   the shared pre-synaptic neurons make onto the group's members, what
                   fraction land on spines. Returned under 'shared_ex_spine_frac' /
                   'shared_inh_spine_frac'. This is the "do shared inputs preferentially target
                   spines?" question, which is distinct from spine targeting (synapses *between*
                   members) — shared input comes from outside the group.

    Two flavours built from the within-column syn_df, split by the pre-synaptic cell's
    type rather than pooled:
      'shared_ex' — shared input from E cells only (pre_clf_type == 'E')
      'shared_inh' — shared input from I cells only (pre_clf_type == 'I')

    A pre-synaptic cell has one clf_type, so a cell in the pooled (E+I) intersection is
    necessarily in exactly one of these two: shared_ex + shared_inh == the old
    `shared_all`, which is kept as a column for composition reporting but is no longer
    tested on its own. Splitting matters because the two behave oppositely with ensemble
    size — the E-only intersection empties out at n>=4 while the I-only one carries the
    whole effect.

    Real ensembles are compared to size- AND edge-matched random control groups
    (the same constraint spine targeting uses). If pos_array is given, also compactness-matched.
    """
    from stats_corr import p_to_stars

    # How many times the paired bootstrap resamples the drawn control groups. Separate
    # from `n_controls`, which is how many groups get drawn: resampling is cheap and
    # under-doing it leaves the reported p-value jittering by a few 1/1000ths, which is
    # enough to move a significance star when p lands near a threshold.
    if n_boot is None:
        n_boot = n_controls

    ex_set = set(all_ex_root_ids)
    sub_post = syn_df[syn_df['post_id'].isin(ex_set)]
    sub_shared_ex   = sub_post[sub_post['pre_clf_type'] == 'E']
    sub_shared_inh   = sub_post[sub_post['pre_clf_type'] == 'I']

    pre_by_post_ex = sub_shared_ex.groupby('post_id')['pre_id'].apply(set).to_dict()
    pre_by_post_inh = sub_shared_inh.groupby('post_id')['pre_id'].apply(set).to_dict()

    def _shared_set(root_ids, pre_by_post):
        sets = [pre_by_post.get(rid, set()) for rid in root_ids]
        if not sets:
            return set()
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return result

    # ── spine tags for the shared-input synapses (optional) ───────────────────
    # tag_by_post[post_id][pre_id] = (n_tagged_synapses, n_spine_synapses)
    tag_by_post = None
    if syn_with_tags is not None:
        _t = syn_with_tags[syn_with_tags['post_id'].isin(ex_set)]
        _g = (_t.assign(_is_spine=(_t['tag'] == 'spine').astype(int))
                .groupby(['post_id', 'pre_id'])
                .agg(n_syn=('_is_spine', 'size'), n_spine=('_is_spine', 'sum')))
        tag_by_post = {}
        for (post, pre), row in _g.iterrows():
            tag_by_post.setdefault(post, {})[pre] = (int(row['n_syn']), int(row['n_spine']))
        print(f'Shared-input spine tags: {len(_t):,} tagged synapses onto {len(tag_by_post):,} '
              f'post-synaptic neurons')

    def _shared_spine_counts(root_ids, shared_pre):
        """(n_syn, n_spine) over synapses from `shared_pre` onto `root_ids`."""
        if tag_by_post is None or not shared_pre:
            return 0, 0
        n_syn = n_spine = 0
        for rid in root_ids:
            per_pre = tag_by_post.get(rid)
            if not per_pre:
                continue
            for pre in shared_pre:
                hit = per_pre.get(pre)
                if hit is not None:
                    n_syn   += hit[0]
                    n_spine += hit[1]
        return n_syn, n_spine

    def _group_row(label, root_ids, with_n_members=False):
        sh_shared_ex = _shared_set(root_ids, pre_by_post_ex)
        sh_shared_inh = _shared_set(root_ids, pre_by_post_inh)
        row = {label_col: label, 'shared_ex': len(sh_shared_ex), 'shared_inh': len(sh_shared_inh),
               # E and I sets are disjoint by cell identity, so this is exactly the
               # pooled E+I intersection the earlier `shared_all` flavour tested.
               'shared_all': len(sh_shared_ex) + len(sh_shared_inh)}
        if with_n_members:
            row['n_members'] = len(root_ids)
        if tag_by_post is not None:
            row['shared_ex_syn'], row['shared_ex_spine'] = _shared_spine_counts(root_ids, sh_shared_ex)
            row['shared_inh_syn'], row['shared_inh_spine'] = _shared_spine_counts(root_ids, sh_shared_inh)
        return row

    # Real ensembles
    real_rows = [_group_row(label, members, with_n_members=True)
                 for label, members in label_to_members.items()]
    real_df = pd.DataFrame(real_rows)
    print('Shared input (observed ensembles):')
    for _, row in real_df.iterrows():
        print(f'  {label_col}={row[label_col]}  n={row["n_members"]}  '
              f'shared_ex={row["shared_ex"]}  shared_inh={row["shared_inh"]}')

    idx_to_rid = {v: k for k, v in pool['rid_to_idx'].items()}

    # The same edge-matched (+ optionally compactness-matched) controls spine targeting uses —
    # driven by the shared iter_matched_groups sampler.
    match_label = 'size + edge'
    if pos_array is not None:
        match_label += ' + compactness'
    if degree_array is not None:
        match_label += ' + in-degree'
    print(f'Sampling {match_label} matched controls for shared input...')

    def _seed_fn(row, lc=label_col, s=seed):
        # crc32, not hash(): CPython randomises hash(str) per process.
        # See the module docstring.
        return int(s * 997 + zlib.crc32(str(row[lc]).encode()) % 100_000)

    # iter_matched_groups iterates ensembles_df rows; align it to label_to_members
    # order by selecting the matching subset (preserving n_members/n_edges columns).
    ens_for_iter = (ensembles_df.set_index(label_col)
                                .loc[list(label_to_members.keys()),
                                     ['n_members', 'n_edges']]
                                .reset_index())

    ctrl_rows = []
    for label, group_idx in iter_matched_groups(
            ens_for_iter, label_col=label_col, seed_fn=_seed_fn, pool=pool,
            n_controls=n_controls, max_attempts=max_attempts,
            verbose_short_only=True,
            label_to_members=label_to_members, match_edges=True,
            pos_array=pos_array, dist_tol=dist_tol, abs_floor_um=abs_floor_um,
            degree_array=degree_array, degree_tol=degree_tol,
            eligible_idx=eligible_idx):
        ctrl_rows.append(_group_row(label, [idx_to_rid[i] for i in group_idx]))
    ctrl_df = pd.DataFrame(ctrl_rows)

    # Paired bootstrap per flavor
    rng_boot = np.random.default_rng(seed)
    def _bootstrap(col):
        obs   = int(real_df[col].sum())
        pools = {row[label_col]: ctrl_df.loc[ctrl_df[label_col] == row[label_col], col].values
                 for _, row in real_df.iterrows()}
        usable = [lbl for lbl, arr in pools.items() if len(arr) > 0]
        null_vals = np.empty(n_boot)
        for r in range(n_boot):
            null_vals[r] = sum(pools[lbl][rng_boot.integers(0, len(pools[lbl]))] for lbl in usable)
        valid = null_vals[~np.isnan(null_vals)]
        p_emp = (1 + np.sum(valid >= obs)) / (1 + len(valid))
        star  = p_to_stars(p_emp)
        print(f'  {col}: obs={obs}  null mean={np.nanmean(valid):.1f} '
              f'[{np.nanpercentile(valid,2.5):.1f}, {np.nanpercentile(valid,97.5):.1f}]  '
              f'p={p_emp:.4f} {star}')
        return {'obs': obs, 'null': valid, 'p_emp': p_emp, 'star': star}

    print('Shared-input bootstrap:')
    res_shared_ex = _bootstrap('shared_ex')
    res_shared_inh = _bootstrap('shared_inh')

    out = {'real_df': real_df, 'ctrl_df': ctrl_df, 'shared_ex': res_shared_ex, 'shared_inh': res_shared_inh,
           'label_col': label_col}

    # Spine fraction of the shared input itself — pooled Σspine/Σsyn, paired bootstrap
    # (the same shape as spine targeting, but over synapses arriving FROM the shared pre-synaptic
    # neurons rather than synapses between members).
    if tag_by_post is not None:
        print('Spine fraction of the shared input:')
        for flavour, num_col, den_col in (('shared_ex', 'shared_ex_spine', 'shared_ex_syn'),
                                          ('shared_inh', 'shared_inh_spine', 'shared_inh_syn')):
            out[f'{flavour}_spine_frac'] = _bootstrap_ratio(
                real_df, ctrl_df, label_col, num_col, den_col,
                n_controls=n_boot, seed=seed, name=f'  {flavour}')
    return out


def _bootstrap_ratio(real_df, ctrl_df, label_col, num_col, den_col,
                     n_controls=1000, seed=0, name=''):
    """Paired bootstrap for a pooled ratio Σnum/Σden over labels. Shared by the
    shared-input spine fraction and the outgoing-targeting metric."""
    from stats_corr import p_to_stars

    num_obs = int(real_df[num_col].sum())
    den_obs = int(real_df[den_col].sum())
    obs     = num_obs / den_obs if den_obs > 0 else np.nan

    pools  = {lbl: ctrl_df.loc[ctrl_df[label_col] == lbl, [num_col, den_col]].to_numpy()
              for lbl in real_df[label_col]}
    usable = [lbl for lbl, arr in pools.items() if len(arr) > 0]

    rng  = np.random.default_rng(seed)
    null = np.empty(n_controls)
    for r in range(n_controls):
        n_s = d_s = 0
        for lbl in usable:
            arr  = pools[lbl]
            draw = arr[rng.integers(0, len(arr))]
            n_s += draw[0]
            d_s += draw[1]
        null[r] = n_s / d_s if d_s > 0 else np.nan

    valid = null[~np.isnan(null)]
    if np.isnan(obs) or len(valid) == 0:
        print(f'{name}: no synapses from shared input — skipped')
        return {'obs': obs, 'null': valid, 'p_emp': np.nan, 'star': '',
                'n_num': num_obs, 'n_den': den_obs}
    p_emp = (1 + np.sum(valid >= obs)) / (1 + len(valid))
    star  = p_to_stars(p_emp)
    print(f'{name}: obs={num_obs}/{den_obs}={obs:.3f}  null mean={np.nanmean(valid):.3f} '
          f'[{np.nanpercentile(valid, 2.5):.3f}, {np.nanpercentile(valid, 97.5):.3f}]  '
          f'p={p_emp:.4f} {star}')
    return {'obs': obs, 'null': valid, 'p_emp': p_emp, 'star': star,
            'n_num': num_obs, 'n_den': den_obs}


def build_pool_feature_array(pool, values, label='feature', verbose=True):
    """(n_ex,) float array of a per-neuron value indexed by pool index; NaN where missing.

    `values` : pandas Series indexed by root_id (or dict root_id -> value).
    """
    values = values if isinstance(values, dict) else values.to_dict()
    arr = np.full(pool['n_ex'], np.nan)
    for rid, idx in pool['rid_to_idx'].items():
        v = values.get(rid)
        if v is not None:
            arr[idx] = v
    if verbose:
        n_have = int(np.isfinite(arr).sum())
        print(f'{label}: {n_have}/{pool["n_ex"]} pool neurons have a value  '
              f'(median={np.nanmedian(arr):.4g})')
    return arr


def build_member_cell_type_df(label_to_members, neurons_df):
    """One row per ensemble member slot: scan_k, root_id, cell_type.

    Stored in the results pickle by the run so `results_analysis.ipynb` can show the
    cell-type composition of the detected ensembles without loading the connectome.

    A neuron in two ensembles appears twice, matching how every member-level metric
    counts slots.
    """
    ct = neurons_df.set_index('root_id')['cell_type'].to_dict()
    rows = [{'scan_k': lbl, 'root_id': rid, 'cell_type': ct.get(rid)}
            for lbl, members in label_to_members.items() for rid in members]
    df = pd.DataFrame(rows)
    n_missing = int(df['cell_type'].isna().sum()) if len(df) else 0
    if n_missing:
        print(f'build_member_cell_type_df: {n_missing}/{len(df)} member slots have no '
              f'cell_type in neurons_df')
    return df


def run_shared_input_strength_bootstrap(label_to_members, pool, ctrl_groups, feature_array,
                                n_null=1000, seed=0,
                                name='shared input strength', value_label='value'):
    """Paired bootstrap comparing a per-neuron feature of ensemble members vs controls.

    Parameters
    ----------
    label_to_members : dict label -> list[root_id]
    pool             : dict from build_ex_pool_arrays (defines the index space)
    ctrl_groups      : dict label -> list[np.ndarray of pool indices], from
                       sample_matched_controls(..., return_groups=True). Must be keyed
                       by the same labels as label_to_members.
    feature_array    : (n_ex,) per-neuron values, indexed like the pool. The
                       observed statistic is their mean over all pooled members.

    A neuron in two ensembles is counted twice, on both the observed and the null side
    (each replicate draws one control group per ensemble), so the pairing holds.

    Reports a two-sided empirical p — unlike Metrics A-C there is no a-priori direction
    for "are ensemble neurons spinier" — alongside both one-sided tails.
    """
    from stats_corr import p_to_stars

    rid_to_idx = pool['rid_to_idx']

    def _num_den(idx):
        """(Σvalue, n_finite) — the group's contribution to the pooled mean.

        Keeping the pair rather than the ratio is what makes the statistic
        re-poolable over any subset of ensembles (see shared_input_by_size).
        """
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            return 0.0, 0.0
        vals = feature_array[idx]
        vals = vals[np.isfinite(vals)]
        return float(vals.sum()), float(len(vals))

    def _agg(idx):
        num, den = _num_den(idx)
        return num / den if den > 0 else np.nan

    usable = [lbl for lbl in label_to_members if ctrl_groups.get(lbl)]
    missing = [lbl for lbl in label_to_members if not ctrl_groups.get(lbl)]
    if missing:
        print(f'{name}: no control groups for {len(missing)} ensembles (skipped): {missing}')

    member_idx = {lbl: np.array([rid_to_idx[r] for r in label_to_members[lbl]
                                 if r in rid_to_idx], dtype=int)
                  for lbl in usable}
    obs_idx = np.concatenate([member_idx[lbl] for lbl in usable]) if usable else np.array([], int)
    obs = _agg(obs_idx)

    rng  = np.random.default_rng(seed)
    null = np.empty(n_null)
    for r in range(n_null):
        drawn = [ctrl_groups[lbl][rng.integers(0, len(ctrl_groups[lbl]))] for lbl in usable]
        null[r] = _agg(np.concatenate(drawn)) if drawn else np.nan

    valid     = null[~np.isnan(null)]
    p_greater = (1 + np.sum(valid >= obs)) / (1 + len(valid))
    p_less    = (1 + np.sum(valid <= obs)) / (1 + len(valid))
    p_emp     = min(1.0, 2 * min(p_greater, p_less))
    star      = p_to_stars(p_emp)

    null_mean = float(np.nanmean(valid))
    print(f'{name} — {value_label}  ({len(usable)} ensembles, {len(obs_idx)} member slots)')
    print(f'  Observed: {obs:.4g}')
    print(f'  Null (paired bootstrap, n={len(valid)}): mean={null_mean:.4g}  '
          f'[{np.nanpercentile(valid, 2.5):.4g}, {np.nanpercentile(valid, 97.5):.4g}]')
    print(f'  fold = {obs / null_mean:.3f}×   two-sided p={p_emp:.4f} {star}  '
          f'(greater={p_greater:.4f}, less={p_less:.4f})')

    # Per-label (numerator, denominator) rows behind the pooled statistic, on both
    # sides — the same role real_df/ctrl_df play for Metrics B and C, and what
    # shared_input_by_size re-aggregates per ensemble size.
    real_df = pd.DataFrame([dict(zip(('num', 'den'), _num_den(member_idx[lbl])),
                                 **{'label': lbl}) for lbl in usable])
    ctrl_df = pd.DataFrame([dict(zip(('num', 'den'), _num_den(g)), **{'label': lbl})
                            for lbl in usable for g in ctrl_groups[lbl]])

    return {'obs': obs, 'valid_null': valid, 'null_mean': null_mean,
            'p_emp': p_emp, 'p_greater': p_greater, 'p_less': p_less, 'star': star,
            'fold': obs / null_mean if null_mean else np.nan,
            'n_members': len(obs_idx), 'usable_labels': usable,
            'value_label': value_label,
            'real_df': real_df, 'ctrl_df': ctrl_df, 'label_col': 'label'}


def iter_matched_groups(
    ensembles_df,
    label_col,
    seed_fn,
    pool,
    n_controls=1000,
    max_attempts=500_000,
    verbose_short_only=True,
    real_member_idx_sets=None,
    match_edges=True,
    label_to_members=None,
    pos_array=None,
    dist_tol=0.2,
    abs_floor_um=15.0,
    degree_array=None,
    degree_tol=0.2,
    eligible_idx=None,
):
    """Generator over reject-sampled matched control groups.

    Yields (label, group_idx) per accepted group. Handles size + edge (+ optional
    distance, + optional in-degree) matching and rejects draws equal to the real
    ensemble. Callers plug in their own per-group stat computation.

    Parameters mirror sample_matched_controls (see its docstring). When
    real_member_idx_sets is None and label_to_members is provided, real-ensemble
    rejection sets are auto-built from pool['rid_to_idx'].

    `eligible_idx` restricts *which* pool indices may be drawn without changing the
    index space: all matching targets (edges, distance, in-degree) are still computed
    over the full `pool`, so ensemble members stay addressable even when they are
    barred from the controls. See build_control_eligible_idx.
    """
    n_ex         = pool['n_ex']
    edge_pre     = pool['edge_pre']
    edge_post    = pool['edge_post']
    rid_to_idx   = pool['rid_to_idx']

    # Dense pre→post adjacency over the pool, built once: counting a candidate group's
    # internal edges is then a slice, not a pass over the whole edge list. `edge_pre` /
    # `edge_post` are already unique pairs, so this carries the same counts.
    adj = None
    if match_edges:
        adj = np.zeros((n_ex, n_ex), dtype=np.int8)
        adj[edge_pre, edge_post] = 1

    # Draw from the eligible subset if given, else the whole pool. Kept as an index
    # array (not a reduced pool) so every *_array below stays indexed by pool index.
    draw_from = np.arange(n_ex) if eligible_idx is None else np.asarray(eligible_idx, dtype=int)

    if real_member_idx_sets is None and label_to_members is not None:
        real_member_idx_sets = {
            lbl: frozenset(rid_to_idx[r] for r in members if r in rid_to_idx)
            for lbl, members in label_to_members.items()
        }

    avg_dist_targets = None
    if label_to_members is not None and pos_array is not None:
        avg_dist_targets = compute_avg_dist_targets(
            label_to_members, pool, pos_array, dist_tol, abs_floor_um)

    # Summed group in-degree, one target per column of degree_array (E, then I).
    degree_targets = None
    if label_to_members is not None and degree_array is not None:
        degree_targets = {
            lbl: degree_array[[rid_to_idx[r] for r in members if r in rid_to_idx]].sum(axis=0)
            for lbl, members in label_to_members.items()
        }
        _tg = np.array([v for v in degree_targets.values() if v is not None])
        print(f'Degree matching: summed group in-degree — E mean={_tg[:, 0].mean():.0f}, '
              f'I mean={_tg[:, 1].mean():.0f}, tol=±{degree_tol*100:.0f}%')

    for _, erow in ensembles_df.iterrows():
        label    = erow[label_col]
        n_size   = int(erow['n_members'])
        target_x = int(erow['n_edges']) if match_edges else None
        rng      = np.random.default_rng(int(seed_fn(erow)))
        real_set = (real_member_idx_sets or {}).get(label)
        kept     = 0
        attempts = 0
        if len(draw_from) < n_size:
            print(f'  {label_col}={label}  size={n_size}  eligible pool has only '
                  f'{len(draw_from)} neurons — cannot sample  [SKIP]')
            continue
        # Candidates are tested in batches. The accept/reject rules below are exactly
        # the one-at-a-time rules this sampler used before, evaluated over a whole
        # batch at once; `_batch_size` only trades memory for speed. See
        # `_draw_candidate_groups` for why drawing with replacement and dropping the
        # rows that repeat an index is the same uniform draw over distinct-index
        # groups that `rng.choice(..., replace=False)` makes.
        target_d = None
        if avg_dist_targets is not None:
            _td = avg_dist_targets.get(label)
            if _td is not None and not np.isnan(_td) and _td > 0:
                target_d = float(_td)
        target_g = degree_targets.get(label) if degree_targets is not None else None
        real_mask = None
        if real_set is not None and len(real_set) == n_size:
            real_mask = np.zeros(n_ex, dtype=bool)
            real_mask[list(real_set)] = True

        batch = _batch_size(n_size)
        while kept < n_controls and attempts < max_attempts:
            groups = _draw_candidate_groups(rng, draw_from, n_size, batch)
            if len(groups) == 0:
                continue
            room = max_attempts - attempts
            if len(groups) > room:
                groups = groups[:room]
            attempts += len(groups)

            keep = np.ones(len(groups), dtype=bool)

            # A draw that is exactly the real ensemble is not a control. Rows hold
            # distinct indices, so "every member is in real_set" means "equals it".
            if real_mask is not None:
                keep &= ~real_mask[groups].all(axis=1)

            if match_edges and keep.any():
                g = groups[keep]
                n_edges = adj[g[:, :, None], g[:, None, :]].sum(axis=(1, 2))
                keep[keep] = n_edges == target_x

            if target_d is not None and keep.any():
                g = groups[keep]
                avg_d = _mean_pairwise_dist_batch(g, pos_array)
                threshold = max(dist_tol * target_d, abs_floor_um)
                # NaN (fewer than two placed somata) fails this test, as it did before.
                keep[keep] = np.abs(avg_d - target_d) <= threshold

            if target_g is not None and keep.any():
                g = groups[keep]
                cand_g = degree_array[g].sum(axis=1)
                keep[keep] = ~np.any(np.abs(cand_g - target_g) >
                                     degree_tol * np.maximum(target_g, 1), axis=1)

            for group_idx in groups[keep]:
                yield label, group_idx
                kept += 1
                if kept == n_controls:
                    break
        status = 'OK' if kept == n_controls else 'SHORT'
        if not verbose_short_only or status == 'SHORT':
            parts = []
            if match_edges:
                parts.append(f'target n_edges={target_x:3d}')
            if avg_dist_targets is not None:
                td     = avg_dist_targets.get(label)
                td_str = f'{td:.1f} µm' if (td is not None and not np.isnan(td)) else 'N/A'
                parts.append(f'target_dist={td_str}')
            if degree_targets is not None:
                tg = degree_targets.get(label)
                if tg is not None:
                    parts.append(f'target_deg=E{int(tg[0])}/I{int(tg[1])}')
            extras = '  '.join(parts) + ('  ' if parts else '')
            print(f'  {label_col}={label}  size={n_size}  {extras}'
                  f'kept={kept:4d}/{n_controls}  attempts={attempts:>6}  [{status}]')


def sample_matched_controls(
    ensembles_df,
    label_col,
    seed_fn,
    pool,
    n_controls=1000,
    max_attempts=500_000,
    verbose_short_only=True,
    real_member_idx_sets=None,
    match_edges=True,
    label_to_members=None,
    pos_array=None,
    dist_tol=0.2,
    abs_floor_um=15.0,
    degree_array=None,
    degree_tol=0.2,
    eligible_idx=None,
    return_groups=False,
):
    """Reject-sample matched control groups; return per-group spine/synapse counts.

    Thin wrapper over iter_matched_groups.

    Parameters
    ----------
    ensembles_df      : DataFrame with columns 'n_members', <label_col>;
                        also 'n_edges' when match_edges=True
    label_col         : column in ensembles_df carried into each output row
    seed_fn           : callable(row) -> int — RNG seed per ensemble row
    pool              : dict from build_ex_pool_arrays
    n_controls        : controls to sample per ensemble row (default 1000)
    max_attempts      : rejection-sampling budget per row (default 500_000)
    verbose_short_only: if True, only print rows that fell short of n_controls
    real_member_idx_sets : dict label -> frozenset[int]; auto-built from label_to_members
                        via pool['rid_to_idx'] if not provided.
    match_edges       : if True (default), reject-sample until group has same n_edges as
                        the ensemble (spine targeting, shared input). False → size-only
                        (the connection-probability null).
    label_to_members  : dict label -> list[root_id]. Enables auto real-ensemble rejection
                        and, when pos_array is also given, distance matching on the
                        group's all-pair compactness (see compute_avg_dist_targets).
    pos_array         : float array (pool['n_ex'], 3) of soma positions in µm.
    dist_tol          : fractional tolerance for distance (default 0.2 = ±20%);
                        effective window = max(dist_tol * target_d, abs_floor_um).
    abs_floor_um      : absolute tolerance floor in µm (default 15.0).
    degree_array      : int array (pool['n_ex'], 2) from build_pool_indegree_array.
                        When given (with label_to_members), also matches the group's
                        SUMMED in-degree, per column (E, then I), within degree_tol.
    degree_tol        : fractional tolerance for summed in-degree (default 0.2 = ±20%).
    eligible_idx      : optional int array of pool indices controls may be drawn from.
                        Use build_control_eligible_idx to bar every ensemble member from
                        every null, which is what keeps one ensemble's members out of
                        another ensemble's control groups.
    return_groups     : if True, return (control_df, groups_by_label) where
                        groups_by_label is a dict {label: list[np.ndarray]} of the
                        accepted pool-index arrays. Lets shared input reuse
                        these groups instead of resampling.

    Returns
    -------
    DataFrame with columns: <label_col>, n_synapses_internal, n_spine_internal.
    When return_groups=True, returns (DataFrame, groups_by_label) instead.
    """
    n_ex         = pool['n_ex']
    syn_pre_idx  = pool['syn_pre_idx']
    syn_post_idx = pool['syn_post_idx']
    syn_is_spine = pool['syn_is_spine']

    target_edges_by_label = (ensembles_df.set_index(label_col)['n_edges'].to_dict()
                             if match_edges else {})
    rows = []
    groups_by_label = {} if return_groups else None
    total_multi_contacts = 0
    for label, group_idx in iter_matched_groups(
            ensembles_df, label_col, seed_fn, pool,
            n_controls=n_controls, max_attempts=max_attempts,
            verbose_short_only=verbose_short_only,
            real_member_idx_sets=real_member_idx_sets, match_edges=match_edges,
            label_to_members=label_to_members, pos_array=pos_array,
            dist_tol=dist_tol, abs_floor_um=abs_floor_um,
            degree_array=degree_array, degree_tol=degree_tol,
            eligible_idx=eligible_idx):
        mask = np.zeros(n_ex, dtype=bool)
        mask[group_idx] = True
        syn_mask = mask[syn_pre_idx] & mask[syn_post_idx]
        n_syn    = int(syn_mask.sum())
        rows.append({
            label_col:             label,
            'n_synapses_internal': n_syn,
            'n_spine_internal':    int(syn_is_spine[syn_mask].sum()),
        })
        if return_groups:
            groups_by_label.setdefault(label, []).append(np.asarray(group_idx, dtype=np.int32))
        if match_edges:
            total_multi_contacts += n_syn - target_edges_by_label[label]
    n_samples = len(rows)
    if match_edges and n_samples:
        print(f'Controls: {total_multi_contacts} multi-contact synapses across {n_samples} sampled groups '
              f'(avg {total_multi_contacts/n_samples:.2f} per group)')
    control_df = pd.DataFrame(rows)
    if return_groups:
        return control_df, groups_by_label
    return control_df

# ═════════════════════════════════════════════════════════════════════════════
# Reading the results back: per-neuron features, cell-type composition, per-size
# aggregation. Used by the figure notebooks.
# ═════════════════════════════════════════════════════════════════════════════

def build_shared_input_strength_series(df, verbose=True):
    """Per-neuron `mean_input_outdegree_full` — figure4 panel E's y axis.

    "Shared input strength" = the mean within-column out-degree of a neuron's
    pre-synaptic partners. A neuron fed by broadly-projecting cells scores high, so it
    is the per-neuron counterpart of shared input: that asks whether a group's members
    intersect on specific pre-cells, this asks whether a member is individually wired
    to the kind of cell that feeds many targets at once. Both E and I inputs count
    (the `_full` flavour, over the whole binarised E+I matrix) — which is the flavour
    figure4 plots against `outgoing_spine_ratio_all`.

    Rebuilt here rather than read off `connectome_neurons.csv` because it is a property
    of the *filtered* matrix, not of a neuron: it needs the filter → rebuild → recompute
    pass, exactly as figure4 cells 2 and 5 do it.

    `df` : the calc_spines_features frame (pre-filter). Returns a Series root_id → value,
    covering only the neurons that survive `filter_valid_neuron_w_spines` — pool neurons
    outside that population get NaN downstream and drop out of the bootstrap.
    """
    from spines_utils import (filter_valid_neuron_w_spines,
                                                     split_syn_mat_by_type_four)
    from spine_pref_utils import mean_input_outdegree

    (fdf, _syn_mat, filtered_bin_mat, filtered_mapping,
     _rev, _ex, _inh) = filter_valid_neuron_w_spines(df)
    neuron_clf_type = fdf[['root_id', 'clf_type']].set_index('root_id').to_dict(orient='index')

    EE, EI, IE, II, ex_idx, inh_idx = split_syn_mat_by_type_four(
        filtered_bin_mat, filtered_mapping, neuron_clf_type)
    ex_od_df, _inh_od_df = mean_input_outdegree(
        {'EE': EE, 'EI': EI, 'IE': IE, 'II': II},
        {'EE': ex_idx, 'IE': ex_idx, 'EI': inh_idx, 'II': inh_idx},
        filtered_mapping, fdf,
        full_mat=filtered_bin_mat, all_idx=sorted(filtered_mapping.keys()))

    s = ex_od_df.set_index('root_id')['mean_input_outdegree_full'].dropna()
    if verbose:
        print(f'Shared input strength: {len(s)} E neurons in the filtered population '
              f'(median={s.median():.4g})')
    return s


def member_cell_type_order(*member_dfs):
    """Shared x ordering for cell-type panels: canonical E order, observed types only.

    Pass every run's `member_cell_type_df` so the panels of one figure share an axis —
    a type missing from one detector still gets its (empty) slot.
    """
    from connectome_types import col_cell_types_ordered
    seen = set()
    for d in member_dfs:
        seen |= set(d['cell_type'].dropna().unique())
    return [ct for ct in col_cell_types_ordered if ct in seen] + \
           sorted(seen - set(col_cell_types_ordered))


def recorded_ex_root_ids(neurons_df, h5_path=None):
    """Column E neurons with calcium in >= 1 scan — the population the detector saw.

    The same set `ensemble_run.load_shared` derives from `ex_func_data`, but read off
    the H5's key list instead of its traces: as a *denominator* (which cell types could
    have been detected at all) it needs no signal, and loading ~1k neurons' traces to
    count them costs minutes.
    """
    import h5py
    from connectome_types import CALCIUM_H5_PATH
    from activity_utils import build_nucleus_to_root_id

    ex_root_ids = neurons_df.loc[neurons_df.clf_type == 'E', 'root_id']
    nuc_to_root = build_nucleus_to_root_id(neurons_df, ex_root_ids)
    with h5py.File(h5_path or CALCIUM_H5_PATH, 'r') as f:
        recorded_nuc = {int(k) for k in f['neurons'].keys()}
    return np.array(sorted(nuc_to_root[n] for n in recorded_nuc if n in nuc_to_root))


def member_cell_type_stats(member_cell_type_df, reference_cell_types, order=None):
    """Member cell-type counts against the population they were drawn from.

    `reference_cell_types` : the cell_type Series of that population — for figure 6
    the *recorded* E neurons, not the whole column: a type the detector never had the
    chance to pick cannot be under-represented.

    Members are distinct neurons drawn without replacement from that population, so
    the count of type c is hypergeometric, not binomial — with 401 of ~1.1k recorded
    neurons detected (Ecker), the finite-population correction shrinks the SD by ~20%
    and the binomial z would overstate every deviation. Returns a frame indexed by
    cell type with obs/expected counts, both fractions, `enrichment` (obs/expected)
    and `z`. Types absent from the reference get z = NaN.
    """
    d = member_cell_type_df.drop_duplicates('root_id')
    ref = pd.Series(reference_cell_types).dropna()
    order = order if order is not None else member_cell_type_order(member_cell_type_df)

    obs = d['cell_type'].value_counts().reindex(order).fillna(0).astype(float)
    K   = ref.value_counts().reindex(order).fillna(0).astype(float)
    M   = float(len(ref))
    N   = float(obs.sum())

    p   = K / M
    exp = N * p
    var = N * p * (1 - p) * (M - N) / (M - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(var > 0, (obs - exp) / np.sqrt(var), np.nan)
        enr = np.where(exp > 0, obs / exp, np.nan)

    return pd.DataFrame({'obs': obs, 'expected': exp,
                         'obs_frac': obs / N, 'ref_frac': p,
                         'enrichment': enr, 'z': z}, index=pd.Index(order, name='cell_type'))


def recorded_coverage_table(subset_cell_types, reference_cell_types, order=None,
                            total_label='All E'):
    """Per cell type: how much of a population a subset caught — the table form.

    The counting behind figure 6's recorded-vs-column panel, as numbers rather than
    bars: `coverage` (subset / reference within a type) is the one column that says
    whether the calcium recording under-sampled a type, and it is the column a bar
    chart of two fractions makes the reader compute by eye.

    Returns a frame indexed by cell type — `n_reference`, `n_subset`, `coverage`, and
    both share-of-population fractions — with a `total_label` row appended.
    """
    sub = pd.Series(subset_cell_types).dropna()
    ref = pd.Series(reference_cell_types).dropna()
    if order is None:
        from connectome_types import col_cell_types_ordered
        seen = set(ref.unique())
        order = [ct for ct in col_cell_types_ordered if ct in seen] + \
                sorted(seen - set(col_cell_types_ordered))
    order = list(order)

    n_ref = ref.value_counts().reindex(order).fillna(0).astype(int)
    n_sub = sub.value_counts().reindex(order).fillna(0).astype(int)
    out = pd.DataFrame({'n_reference': n_ref, 'n_subset': n_sub},
                       index=pd.Index(order, name='cell_type')).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['coverage'] = np.where(out.n_reference > 0,
                                   out.n_subset / out.n_reference, np.nan)
    out['ref_frac']    = out.n_reference / len(ref)
    out['subset_frac'] = out.n_subset / len(sub)
    out.loc[total_label] = [len(ref), len(sub), len(sub) / len(ref), 1.0, 1.0]
    return out


def _per_size_tables(results):
    """Per-label (numerator, denominator) tables for every metric, on both sides.

    Returns a list of specs, one per metric:
        (key, pretty, mode, real_df, ctrl_df, label_col, num_col, den_col, two_sided)

    `mode` is 'sum' (the statistic is a pooled count — the shared-input intersections)
    or 'ratio' (a pooled Σnum/Σden — every other measure). Measures whose per-label
    tables are absent from the pickle are skipped.
    """
    ens = results['all_ensembles_df']
    specs = []

    # Spine targeting: its per-label tables are the ensemble table itself and the
    # size + edge (+ compactness) matched control draws.
    ctrl_a = results.get('matched_controls_df')
    if ctrl_a is not None:
        specs.append(('spine_targeting', 'spine targeting', 'ratio',
                      ens, ctrl_a, 'scan_k',
                      ('n_spine', 'n_spine_internal'),
                      ('n_synapses', 'n_synapses_internal'), False))

    # Connection probability.
    b = results.get('connection_probability') or {}
    if 'real_df' in b:
        specs.append(('connection_probability', 'connection probability', 'ratio',
                      b['real_df'], b['ctrl_df'], b['label_col'],
                      ('n_synapses', 'n_synapses_internal'),
                      ('n_possible', 'n_possible'), False))

    # Shared input, and the spine fraction of that shared input, split E / I.
    c = results.get('shared_input') or {}
    if 'real_df' in c:
        lc = c['label_col']
        for flavour, pretty in (('shared_ex', 'E-only'), ('shared_inh', 'I-only')):
            col = flavour
            if col in c['real_df']:
                specs.append((flavour, f'shared input ({pretty})', 'sum',
                              c['real_df'], c['ctrl_df'], lc, (col, col), None, False))
            if f'{flavour}_syn' in c['real_df']:
                specs.append((f'{flavour}_spine_frac',
                              f'spine frac of shared input ({pretty})', 'ratio',
                              c['real_df'], c['ctrl_df'], lc,
                              (f'{flavour}_spine', f'{flavour}_spine'),
                              (f'{flavour}_syn', f'{flavour}_syn'), False))

    # Shared input strength is a per-neuron feature of the members, so its test is
    # two-sided: nothing predicts a direction for "do ensemble neurons receive from
    # better-connected partners".
    m = results.get('shared_input_strength') or {}
    if 'real_df' in m:
        specs.append(('shared_input_strength', 'shared input strength', 'ratio',
                      m['real_df'], m['ctrl_df'], m.get('label_col', 'label'),
                      ('num', 'num'), ('den', 'den'), True))
    return specs


def shared_input_by_size(results, n_boot=1000, seed=0, cumulative=(3, 4),
                    min_null=1.0, min_den=20):
    """Every metric split by ensemble size — paired bootstrap within each size class.

    The pooled figures aggregate ensembles of all sizes into one comparison. This runs the identical paired bootstrap separately over the ensembles
    of each size, so a size-2 ensemble is only ever compared with size-2 controls. It
    re-uses the control draws already stored in the pickle, so it needs no resampling
    and no connectome access.

    Every measure is a pooled statistic over labels — either Σnum (the shared-input
    intersection counts) or Σnum/Σden (all the rest) — so restricting the label set to
    one size and re-running the same bootstrap is exactly the per-size version of the
    same test. Metrics D and E get the two-sided p their pooled versions use.

    Returns a long DataFrame with one row per (metric, size):
      metric, pretty, mode, size ('2', '3', … plus 'n>=3', 'n>=4'), n_ens,
      obs, obs_num, obs_den, null, fold, value, value_null, p, star, reliable

    `value` / `value_null` are the numbers to plot per size: the statistic itself for
    ratio metrics, and the per-ensemble mean (Σ/n_ens) for count metrics, so sizes with
    different numbers of ensembles stay comparable.

    `reliable` is False where the fold is arithmetic rather than biology: for count
    metrics when the null mean is below `min_null` (a fold against a null that is
    essentially always empty says "the null never does this", not "5000x"), and for
    ratio metrics when the observed denominator is under `min_den` synapses.
    """
    from stats_corr import p_to_stars

    ens = results['all_ensembles_df']
    size_by_label = ens.set_index('scan_k')['n_members'].to_dict()

    def _groups(labels_all):
        out = [(str(int(sz)), [l for l in labels_all if size_by_label.get(l) == sz])
               for sz in sorted({size_by_label[l] for l in labels_all
                                 if l in size_by_label})]
        for thr in cumulative:
            sel = [l for l in labels_all if size_by_label.get(l, 0) >= thr]
            if sel:
                out.append((f'n>={thr}', sel))
        return out

    rows = []
    for (key, pretty, mode, real_df, ctrl_df, label_col,
         (rnum, cnum), den_cols, two_sided) in _per_size_tables(results):
        rden, cden = den_cols if den_cols else (None, None)
        real = real_df.set_index(label_col)
        ctrl_by_label = {lbl: sub for lbl, sub in ctrl_df.groupby(label_col)}
        rng = np.random.default_rng(seed)

        for size_label, labels in _groups(list(real.index)):
            usable = [l for l in labels if len(ctrl_by_label.get(l, ())) > 0]
            if not usable:
                continue
            obs_num = float(real.loc[usable, rnum].sum())
            # count metrics have no denominator — obs is the sum itself
            obs_den = float(real.loc[usable, rden].sum()) if rden else np.nan
            obs = obs_num if mode == 'sum' else (obs_num / obs_den if obs_den else np.nan)

            pools = {l: (ctrl_by_label[l][cnum].to_numpy(dtype=float),
                         ctrl_by_label[l][cden].to_numpy(dtype=float)
                         if cden else np.ones(len(ctrl_by_label[l])))
                     for l in usable}
            null = np.empty(n_boot)
            for r in range(n_boot):
                n_s = d_s = 0.0
                for l in usable:
                    num_arr, den_arr = pools[l]
                    j = rng.integers(0, len(num_arr))
                    n_s += num_arr[j]
                    d_s += den_arr[j]
                null[r] = n_s if mode == 'sum' else (n_s / d_s if d_s > 0 else np.nan)

            valid = null[~np.isnan(null)]
            if not len(valid) or not np.isfinite(obs):
                continue
            p_greater = (1 + np.sum(valid >= obs)) / (1 + len(valid))
            p_less    = (1 + np.sum(valid <= obs)) / (1 + len(valid))
            p = min(1.0, 2 * min(p_greater, p_less)) if two_sided else p_greater
            mu = float(np.nanmean(valid))
            n_ens = len(usable)
            rows.append({
                'metric': key, 'pretty': pretty, 'mode': mode,
                'size': size_label, 'n_ens': n_ens,
                'obs': obs, 'obs_num': obs_num, 'obs_den': obs_den,
                'null': mu, 'fold': (obs / mu) if mu else np.nan,
                'value':      obs / n_ens if mode == 'sum' else obs,
                'value_null': mu / n_ens if mode == 'sum' else mu,
                'p': p, 'star': p_to_stars(p),
                'reliable': bool(mu >= min_null) if mode == 'sum'
                            else bool(obs_den >= min_den),
            })
    return pd.DataFrame(rows)
