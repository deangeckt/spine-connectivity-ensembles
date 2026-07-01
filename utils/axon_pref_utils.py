"""
Utility functions for axon spine preference analysis.
Used by supplementary figures fig_s4 and fig_s8.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def idan_fit(x, rho_inf, d0):
    """Saturating model: rho_inf * (d / (d + d0))"""
    return rho_inf * (x / (x + d0))


def biphasic_spine_fit(d, rho0, rho_inf, d0, A, mu, sigma):
    """
    Saturating + Gaussian bump model.
    d: distance (x-axis). rho0: baseline. rho_inf: asymptote.
    d0: half-saturation distance. A: bump amplitude. mu: bump center. sigma: bump width.
    """
    saturation = rho0 + (rho_inf - rho0) * (d / (d + d0))
    bump = A * np.exp(-((d - mu) ** 2) / (2 * sigma ** 2))
    return saturation + bump


def calc(node_id, outgoing_syn, dist_col='dist_to_pre_syn_soma',
         log=True, min_synapses_per_bin=0, single_bins=80, single_bin_centers=None):
    if single_bin_centers is None:
        single_bin_centers = []
    single_syn = outgoing_syn[outgoing_syn['pre_id'] == node_id]
    counts_all_single, _ = np.histogram(single_syn[dist_col], bins=single_bins)
    spines_only_single = single_syn[single_syn.tag == 'spine']
    global_spine_avg = spines_only_single.shape[0] / single_syn.shape[0]
    if log:
        print(f'Global spine fraction for neuron {node_id}: {global_spine_avg:.4f}')
    counts_spines_single, _ = np.histogram(spines_only_single[dist_col], bins=single_bins)
    valid_bins_single = counts_all_single > min_synapses_per_bin
    spine_fraction_single = np.full(len(counts_all_single), np.nan)
    spine_fraction_single[valid_bins_single] = (
        counts_spines_single[valid_bins_single] / counts_all_single[valid_bins_single]
    )
    x_coords = single_bin_centers[valid_bins_single]
    y_coords = spine_fraction_single[valid_bins_single]
    counts = counts_all_single[valid_bins_single]
    return (x_coords, y_coords, counts, single_bin_centers,
            spine_fraction_single, valid_bins_single, counts_all_single, global_spine_avg)


def fit(x_coords, y_coords, single_bin_centers, counts_all_single, valid_bins_single,
        ax=None, log=True, use_basic_fit=True):
    counts = counts_all_single[valid_bins_single]
    sigmas = 1.0 / np.sqrt(counts)

    if use_basic_fit:
        fit_func = idan_fit
        popt, _ = curve_fit(
            fit_func, x_coords, y_coords,
            bounds=([0, 1e-5], [1.0, np.inf]),
            maxfev=10000, sigma=sigmas, absolute_sigma=False
        )
        rho_inf_opt, d0_opt = popt
        fit_label = f'Basic Fit: $\\rho_\\infty$={rho_inf_opt:.2f}, $d_0$={d0_opt:.1f}'
    else:
        fit_func = biphasic_spine_fit
        popt, _ = curve_fit(
            fit_func, x_coords, y_coords,
            bounds=([0.0, 0.5, 1.0, 0.0, 10.0, 5.0], [0.5, 1.0, 1200.0, 0.5, 200.0, 150.0]),
            p0=[0.2, 0.8, 150, 0.15, 50, 30],
            maxfev=10000, sigma=sigmas, absolute_sigma=False
        )
        rho0_opt, rho_inf_opt, d0_opt, A_opt, mu_opt, sigma_opt = popt
        fit_label = f'Biphasic: $\\rho_\\infty$={rho_inf_opt:.2f}, Bump@ {mu_opt:.0f}µm'

    smooth_xs = np.linspace(0, single_bin_centers.max(), 300)
    if ax is not None:
        ax.plot(smooth_xs, fit_func(smooth_xs, *popt),
                color='gray', lw=2, linestyle='--', alpha=0.8, label=fit_label)

    y_pred = fit_func(x_coords, *popt)
    W = counts
    weighted_mean_y = np.sum(W * y_coords) / np.sum(W)
    ss_tot_w = np.sum(W * (y_coords - weighted_mean_y) ** 2)
    ss_res_w = np.sum(W * (y_coords - y_pred) ** 2)
    r_squared_w = 1.0 if ss_res_w == 0 else (
        (1 - ss_res_w / ss_tot_w) if ss_tot_w != 0 else 0.0
    )
    w_mse = np.mean(W * (y_coords - y_pred) ** 2)

    if log:
        print(f"  rho_inf_opt: {rho_inf_opt:.4f}")
        print(f"  d0_opt:      {d0_opt:.4f}")
        print(f"  W-R^2:       {r_squared_w:.4f}")
        print(f"  W-MSE:       {w_mse:.4e}")

    return w_mse, r_squared_w


def fit_pop(outgoing_syn, num_pop_bins=120, max_distance=1200, ax=None,
            log=False, use_basic_fit=True, dist_col='dist_to_pre_syn_soma', pop_color=None):
    """
    Fit population-average spine fraction vs. distance and plot.
    pop_color: line color (defaults to first matplotlib cycle color).
    """
    if pop_color is None:
        pop_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    pop_bins = np.linspace(0, max_distance, num_pop_bins)
    pop_bin_centers = (pop_bins[:-1] + pop_bins[1:]) / 2
    counts_all_pop = np.zeros(len(pop_bin_centers))
    unique_ids = outgoing_syn['pre_id'].unique()
    all_fractions = []

    for nid in unique_ids:
        axon_data = outgoing_syn[outgoing_syn['pre_id'] == nid]
        c_all, _ = np.histogram(axon_data[dist_col], bins=pop_bins)
        c_spines, _ = np.histogram(axon_data[axon_data.tag == 'spine'][dist_col], bins=pop_bins)
        counts_all_pop += c_all
        frac = np.full(len(c_all), np.nan)
        valid_mask = c_all > 0
        frac[valid_mask] = c_spines[valid_mask] / c_all[valid_mask]
        all_fractions.append(frac)

    all_fractions = np.array(all_fractions)
    valid_bins_pop = np.sum(~np.isnan(all_fractions), axis=0) > 0
    spine_fraction_pop = np.full(all_fractions.shape[1], np.nan)
    sd_pop = np.full(all_fractions.shape[1], np.nan)

    if np.any(valid_bins_pop):
        spine_fraction_pop[valid_bins_pop] = np.nanmean(all_fractions[:, valid_bins_pop], axis=0)
        sd_pop[valid_bins_pop] = np.nanstd(all_fractions[:, valid_bins_pop], axis=0)

    n_neurons_per_bin = np.sum(~np.isnan(all_fractions), axis=0)
    err_pop = np.zeros_like(sd_pop)
    err_pop[valid_bins_pop] = sd_pop[valid_bins_pop] / np.sqrt(n_neurons_per_bin[valid_bins_pop])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=150)

    ax.plot(pop_bin_centers[valid_bins_pop], spine_fraction_pop[valid_bins_pop],
            color=pop_color, lw=2.0, label='Population average', zorder=5., alpha=0.8)
    ax.fill_between(pop_bin_centers[valid_bins_pop],
                    spine_fraction_pop[valid_bins_pop] - err_pop[valid_bins_pop],
                    spine_fraction_pop[valid_bins_pop] + err_pop[valid_bins_pop],
                    color=pop_color, alpha=0.2, edgecolor='none', zorder=4)
    ax.set_xlabel('Axonal path length from Soma (µm)')
    ax.set_ylabel('Ratio of synapses on spines (in bin)')
    ax.set_xlim(0, max_distance)

    x_coords_pop = pop_bin_centers[valid_bins_pop]
    y_coords_pop = spine_fraction_pop[valid_bins_pop]
    r_squared_w = fit(
        x_coords=x_coords_pop,
        y_coords=y_coords_pop,
        single_bin_centers=pop_bin_centers,
        counts_all_single=counts_all_pop,
        valid_bins_single=valid_bins_pop,
        ax=ax, log=log, use_basic_fit=use_basic_fit
    )
    return r_squared_w


def fit_single_wrap(outgoing_syn_df, single_bin_width, max_distance, ax,
                    root_id=864691135617152361, color='red', log=False,
                    min_synapses_per_bin=0, axon_label='',
                    dist_col='dist_to_pre_syn_soma'):
    single_bins = np.linspace(0, max_distance, single_bin_width)
    single_bin_centers = (single_bins[:-1] + single_bins[1:]) / 2

    (x_coords, y_coords, counts, single_bin_centers, spine_fraction_single,
     valid_bins_single, counts_all_single, global_spine_avg) = calc(
        root_id, outgoing_syn_df, dist_col=dist_col,
        min_synapses_per_bin=min_synapses_per_bin,
        single_bins=single_bins, single_bin_centers=single_bin_centers
    )

    ax.plot(single_bin_centers[valid_bins_single], spine_fraction_single[valid_bins_single],
            marker='o', markersize=4, color=color, lw=1.5, alpha=0.7, label=f'{axon_label}')

    for x, y, count in zip(x_coords, y_coords, counts):
        ax.text(x, y + 0.015, str(count), color=color, fontsize=12,
                ha='center', va='bottom', alpha=0.8)

    w_mse, r_squared_w = fit(x_coords, y_coords, single_bin_centers, counts_all_single,
                              valid_bins_single, ax, log=log, use_basic_fit=True)
    return w_mse, r_squared_w


def plot_raw_bin(outgoing_syn, ax, is_ei=False,
                 spiny_color='#7C3AED', aspiny_color='#059669',
                 spiny_color_ei='#BE98FF', aspiny_color_ei='#2CFFB1',
                 dist_col='dist_to_pre_syn_soma'):
    """Plot histogram of synapse distances split by spine/shaft (and optionally E/I target)."""
    outgoing_syn_onto_spines = outgoing_syn[outgoing_syn.tag == 'spine']
    outgoing_syn_onto_nonspines = outgoing_syn[outgoing_syn.tag != 'spine']

    if is_ei:
        curves = [
            (outgoing_syn_onto_spines[outgoing_syn_onto_spines.post_clf_type == 'E'],
             f'E→E spines (N={len(outgoing_syn_onto_spines[outgoing_syn_onto_spines.post_clf_type == "E"])})'),
            (outgoing_syn_onto_spines[outgoing_syn_onto_spines.post_clf_type == 'I'],
             f'E→I spines (N={len(outgoing_syn_onto_spines[outgoing_syn_onto_spines.post_clf_type == "I"])})'),
            (outgoing_syn_onto_nonspines[outgoing_syn_onto_nonspines.post_clf_type == 'E'],
             f'E→E shaft/soma (N={len(outgoing_syn_onto_nonspines[outgoing_syn_onto_nonspines.post_clf_type == "E"])})'),
            (outgoing_syn_onto_nonspines[outgoing_syn_onto_nonspines.post_clf_type == 'I'],
             f'E→I shaft/soma (N={len(outgoing_syn_onto_nonspines[outgoing_syn_onto_nonspines.post_clf_type == "I"])})'),
        ]
        colors = [spiny_color, spiny_color_ei, aspiny_color, aspiny_color_ei]
    else:
        curves = [
            (outgoing_syn_onto_spines, f'onto spines (N={len(outgoing_syn_onto_spines)})'),
            (outgoing_syn_onto_nonspines, f'onto shaft/soma (N={len(outgoing_syn_onto_nonspines)})'),
        ]
        colors = [spiny_color, aspiny_color]

    bins = np.arange(0, 1200, 40)
    for (data, label), c in zip(curves, colors):
        sns.histplot(data[dist_col], discrete=False, fill=False, element='step',
                     bins=bins, lw=1.75, label=label, alpha=0.85, ax=ax, color=c)


def plot_single_neuron(ax, sk_a, sk_d, syn_spines, syn_nonspines, dend_color,
                       z_spine=5, z_shaft=4, z_order_dend=1, circle_size=6,
                       alpha_synapse=0.6, axon_lw=4, dend_lw=4, axon_alpha=0.8,
                       dend_alpha=0.8, axon_color='b',
                       spiny_color='#7C3AED', aspiny_color='#059669'):
    """Plot axon skeleton with synapse scatter colored by spine/shaft."""
    from plot_utils import plot_skeleton_continuous
    plot_skeleton_continuous(ax=ax, sk=sk_a, lw=axon_lw, alpha=axon_alpha,
                             color=axon_color, ignore_vertex_zero=True, coords=('x', 'y'))
    ax.scatter(syn_nonspines.pt_position_xt, syn_nonspines.pt_position_yt,
               s=circle_size, alpha=alpha_synapse, marker='o', color=aspiny_color, zorder=z_shaft)
    ax.scatter(syn_spines.pt_position_xt, syn_spines.pt_position_yt,
               s=circle_size, alpha=alpha_synapse, marker='o', color=spiny_color, zorder=z_spine)
    plot_skeleton_continuous(ax=ax, sk=sk_d, lw=dend_lw, alpha=dend_alpha,
                             color=dend_color, ignore_vertex_zero=True, coords=('x', 'y'),
                             zorder=z_order_dend)
    ax.invert_yaxis()
