import numpy as np
import pandas as pd

def per_neuron_spine_ratio(syn_subset, root_ids, group_by='pre_id', main_feature='spine'):
    # sanity: ee_syn_tags[ee_syn_tags.pre_id == 864691136330407914].tag.value_counts(normalize=True)
    grouped  = syn_subset.groupby(group_by)
    records  = []
    for rid in root_ids:
        if rid not in grouped.groups:
            records.append({'root_id': rid, 'ratio': np.nan, 'n_syn': 0})
            continue
        g = grouped.get_group(rid)
        records.append({'root_id': rid,
                        'ratio':  (g.tag == main_feature).sum() / len(g),
                        'n_syn':  len(g)})
    return pd.DataFrame(records).set_index('root_id')


def compute_per_neuron_spine_ratio(syn_with_tags, pre_root_ids, pre_clf, post_clf, main_feature='spine'):
    """
    For each neuron in pre_root_ids, compute fraction of outgoing tagged
    synapses onto post_clf targets that land on spines.
    Returns DataFrame indexed by root_id: ratio, n_syn, binary_degree.
    """
    sub = syn_with_tags[
        (syn_with_tags.pre_clf_type == pre_clf) &
        (syn_with_tags.post_clf_type == post_clf)
    ]
    grouped = sub.groupby('pre_id')
    records = []
    for rid in pre_root_ids:
        if rid not in grouped.groups:
            records.append({'root_id': rid, 'ratio': np.nan,
                            'n_syn': 0, 'binary_degree': 0})
            continue
        g = grouped.get_group(rid)
        n_spine = (g.tag == main_feature).sum()
        records.append({'root_id': rid,
                        'ratio':         n_spine / len(g),
                        'n_syn':         len(g),
                        'binary_degree': g['post_id'].nunique()})
    return pd.DataFrame(records).set_index('root_id')


def plot_spine_along_axon(ax,
                          outgoing_syn,
                          node_id_list,
                          axon_colors=None,
                          pop_color='black',
                          show_xlabel=False,
                          show_legend=False,
                          cross_dist_textsize=10,
                          show_cross_distance=True,
                          title='',
                          aver_by='micro',
                          err_type='sem',
                          num_single_bins=10,
                          num_pop_bins=100,
                          max_distance=1200,
                          min_synapses_per_bin=1,
                          axon_bin_fontsize=10,
                          inh_pref=False):
    """
    Plot spine fraction vs. distance for a population average and individual axons.

    Parameters
    ----------
    aver_by : str, 'micro' or 'macro'
        'micro': Pooled average (synapse-weighted).
        'macro': Mean of means (neuron-weighted).
    err_type : str, 'sem' or 'sd'
        'sem': Standard Error of the Mean. Shaded area shrinks as N increases.
        'sd': Standard Deviation. Shaded area shows the diversity of the population.
    if inh_pref is true - we calculate the prefrence to targeting E v I (not related to spines)
    """

    dist_col = 'dist_to_pre_syn_soma'

    # --- Binning Strategies ---
    pop_bins = np.linspace(0, max_distance, num_pop_bins)
    pop_bin_centers = (pop_bins[:-1] + pop_bins[1:]) / 2

    single_bins = np.linspace(0, max_distance, num_single_bins)
    single_bin_centers = (single_bins[:-1] + single_bins[1:]) / 2


    if axon_colors is None:
        axon_colors = [f'C{j}' for j in range(len(node_id_list))]

    # ==========================================
    # STEP A: POPULATION AVERAGE
    # ==========================================
    if aver_by == 'micro':
        counts_all_pop, _ = np.histogram(outgoing_syn[dist_col], bins=pop_bins)
        if inh_pref:
            spines_only_pop = outgoing_syn[outgoing_syn.post_clf_type == 'E']
        else:
            spines_only_pop = outgoing_syn[outgoing_syn.tag == 'spine']
        counts_spines_pop, _ = np.histogram(spines_only_pop[dist_col], bins=pop_bins)

        valid_bins_pop = counts_all_pop > 0
        spine_fraction_pop = np.full(len(counts_all_pop), np.nan)
        
        # Only divide where there are actually synapses
        spine_fraction_pop[valid_bins_pop] = counts_spines_pop[valid_bins_pop] / counts_all_pop[valid_bins_pop]
        
        sd_pop = np.sqrt(spine_fraction_pop * (1 - spine_fraction_pop))
        
        if err_type == 'sem':
            err_pop = sd_pop / np.sqrt(counts_all_pop)
        else:
            err_pop = sd_pop

    elif aver_by == 'macro':
        unique_ids = outgoing_syn['pre_id'].unique()
        all_fractions = []

        for node_id in unique_ids:
            axon_data = outgoing_syn[outgoing_syn['pre_id'] == node_id]
            c_all, _ = np.histogram(axon_data[dist_col], bins=pop_bins)
            if inh_pref:
                c_spines, _ = np.histogram(axon_data[axon_data.post_clf_type == 'E'][dist_col], bins=pop_bins)
            else:
                c_spines, _ = np.histogram(axon_data[axon_data.tag == 'spine'][dist_col], bins=pop_bins)
            
            frac = np.full(len(c_all), np.nan)
            valid_mask = c_all > 0
            frac[valid_mask] = c_spines[valid_mask] / c_all[valid_mask]
            all_fractions.append(frac)

        all_fractions = np.array(all_fractions)
        
        # First, find which columns (bins) have at least one valid number
        valid_bins_pop = np.sum(~np.isnan(all_fractions), axis=0) > 0
        
        spine_fraction_pop = np.full(all_fractions.shape[1], np.nan)
        sd_pop = np.full(all_fractions.shape[1], np.nan)
        
        # Only calculate mean/std on columns that are NOT entirely NaN
        if np.any(valid_bins_pop):
            spine_fraction_pop[valid_bins_pop] = np.nanmean(all_fractions[:, valid_bins_pop], axis=0)
            sd_pop[valid_bins_pop] = np.nanstd(all_fractions[:, valid_bins_pop], axis=0)
            
        if err_type == 'sem':
            n_neurons_per_bin = np.sum(~np.isnan(all_fractions), axis=0)
            # Avoid divide by zero for err_pop
            err_pop = np.zeros_like(sd_pop)
            err_pop[valid_bins_pop] = sd_pop[valid_bins_pop] / np.sqrt(n_neurons_per_bin[valid_bins_pop])
        else:
            err_pop = sd_pop
        

    # --- Plotting the Population Line ---
    label_suffix = "SEM" if err_type == 'sem' else "SD"
    ax.plot(pop_bin_centers[valid_bins_pop], spine_fraction_pop[valid_bins_pop],
            color=pop_color, lw=2.0, label=f'Pop. Avg ({aver_by}, {label_suffix})', zorder=5)
    
    ax.fill_between(pop_bin_centers[valid_bins_pop],
                    spine_fraction_pop[valid_bins_pop] - err_pop[valid_bins_pop],
                    spine_fraction_pop[valid_bins_pop] + err_pop[valid_bins_pop],
                    color=pop_color, alpha=0.2, edgecolor='none', zorder=4)

    # --- Crossing distance annotation ---
    y_vals_pop = spine_fraction_pop[valid_bins_pop]
    x_vals_pop = pop_bin_centers[valid_bins_pop]

    if show_cross_distance:

        if np.any(y_vals_pop > 0.5):
            idx1 = np.where(y_vals_pop > 0.5)[0][0]
            if idx1 > 0:
                x0, y0 = x_vals_pop[idx1 - 1], y_vals_pop[idx1 - 1]
                x1, y1 = x_vals_pop[idx1], y_vals_pop[idx1]
                crossing_distance = x0 + (x1 - x0) * ((0.5 - y0) / (y1 - y0 + 1e-9))
            else:
                crossing_distance = x_vals_pop[idx1]

            ax.vlines(x=crossing_distance, ymin=0, ymax=0.5,
                    color='gray', linestyle='--', lw=1.5, alpha=0.7)
            ax.text(crossing_distance + 10, 0.01, f'{crossing_distance:.0f} µm',
                    color='gray', fontsize=cross_dist_textsize)
            ax.axhline(0.5, color='gray', linestyle='--', lw=1.5, alpha=0.7)

    # ==========================================
    # STEP B: INDIVIDUAL AXONS
    # ==========================================
    for j, node_id in enumerate(node_id_list):
        single_syn = outgoing_syn[outgoing_syn['pre_id'] == node_id]

        counts_all_single, _ = np.histogram(single_syn[dist_col], bins=single_bins)
        if inh_pref:
            spines_only_single = single_syn[single_syn.post_clf_type == 'E']
        else:
            spines_only_single = single_syn[single_syn.tag == 'spine']

        counts_spines_single, _ = np.histogram(spines_only_single[dist_col], bins=single_bins)

        valid_bins_single = counts_all_single > min_synapses_per_bin
        spine_fraction_single = np.full(len(counts_all_single), np.nan)
        spine_fraction_single[valid_bins_single] = counts_spines_single[valid_bins_single] / counts_all_single[valid_bins_single]

        # Annotate the synapse count on top of each valid bin ---
        x_coords = single_bin_centers[valid_bins_single]
        y_coords = spine_fraction_single[valid_bins_single]
        counts = counts_all_single[valid_bins_single]
    

        # Plot the axon line
        ax.plot(single_bin_centers[valid_bins_single], spine_fraction_single[valid_bins_single],
                marker='o', markersize=4,
                color=axon_colors[j], lw=1.5, alpha=0.7, label=f'Axon {node_id}', zorder=10)
        
        for x, y, count in zip(x_coords, y_coords, counts):
            # Added a +0.015 offset to 'y' so the text floats slightly above the line point
            ax.text(x, y + 0.015, str(count), color=axon_colors[j], 
                    fontsize=axon_bin_fontsize, ha='center', va='bottom', alpha=0.8, zorder=10)

    # Formatting
    if title:
        ax.set_title(title)
    ax.set_ylabel('Fraction of synapses on spines')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1200)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_xlabel: ax.set_xlabel('Distance along axon (µm)')
    if show_legend: ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)


def build_subnet_spine_ratio_df(mat, row_idx, col_idx, mapping, pre_clf, post_clf,
                                syn_with_tags, neurons_df, main_feature='spine'):
    """
    Build per-neuron spine ratio df for one subnetwork.
    neurons_df must already be filtered (ex_neurons or inh_neurons).
    Hubs defined by rank-based hard count on binary out-degree from mat.
    """
    pre_root_ids = [mapping[i] for i in row_idx]

    mat_outdegree = {
        mapping[row_idx[node]]: int(mat[node, :].sum())
        for node in range(mat.shape[0])
    }

    ratio_df = compute_per_neuron_spine_ratio(syn_with_tags, pre_root_ids, pre_clf, post_clf)

    out = neurons_df[neurons_df.root_id.isin(pre_root_ids)].copy()
    out[f'outgoing_{main_feature}_ratio']   = out.root_id.map(ratio_df['ratio'])
    out['n_syn']         = out.root_id.map(ratio_df['n_syn'])
    out['binary_degree'] = out.root_id.map(mat_outdegree).fillna(0).astype(int)

    return out


def mean_input_outdegree(real_blocks, col_idx_map, filtered_mapping, df,
                            full_mat=None, all_idx=None):
        """
        ...
        Parameters (new)
        ----------------
        full_mat : np.ndarray, optional
            Full (all × all) binarised adjacency matrix. If provided, also
            computes mean_input_outdegree_full for every neuron.
        all_idx : list[int], optional
            Ordered graph indices corresponding to rows/cols of full_mat,
            used to recover root_ids. Required when full_mat is given.
        """
        NAMES = ['EE', 'IE', 'EI', 'II']

        node_mean_od = {}
        for name in NAMES:
            block        = (real_blocks[name] > 0).astype(int)
            pre_od       = block.sum(axis=1)
            weighted_sum = (block * pre_od[:, None]).sum(axis=0)
            in_degree    = block.sum(axis=0).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_od      = np.where(in_degree > 0, weighted_sum / in_degree, np.nan)

            col_gidx  = col_idx_map[name]
            root_ids  = [filtered_mapping[g] for g in col_gidx]
            node_mean_od[name] = pd.Series(mean_od, index=root_ids,
                                        name=f'mean_input_outdegree_{name}')

        # ── Full-matrix metric ────────────────────────────────────────────────────
        full_series = None
        if full_mat is not None:
            assert all_idx is not None, "all_idx required when full_mat is provided"
            full_block        = (full_mat > 0).astype(int)          # (n_all, n_all)
            pre_od_full       = full_block.sum(axis=1)              # out-degree in full graph
            weighted_sum_full = full_block.T @ pre_od_full          # (n_all,)
            in_degree_full    = full_block.sum(axis=0).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_od_full      = np.where(in_degree_full > 0, weighted_sum_full / in_degree_full, np.nan)
            root_ids_all = [filtered_mapping[g] for g in all_idx]
            full_series = pd.Series(mean_od_full, index=root_ids_all,  name='mean_input_outdegree_full')

        # ── Merge into E / I DataFrames ───────────────────────────────────────────
        ex_df  = df[df['clf_type'] == 'E'].copy().set_index('root_id')
        inh_df = df[df['clf_type'] == 'I'].copy().set_index('root_id')

        for name, series in node_mean_od.items():
            target = ex_df if name in ('EE', 'IE') else inh_df
            target[f'mean_input_outdegree_{name}'] = series

        if full_series is not None:
            ex_df['mean_input_outdegree_full']  = full_series
            inh_df['mean_input_outdegree_full'] = full_series

        return ex_df.reset_index(), inh_df.reset_index()


def get_crossing_distances_by_neuron(outgoing_syn, max_distance=1200, num_pop_bins=100,
                                      dist_col='dist_to_pre_syn_soma'):
    """
    For each neuron (pre_id), compute the axonal distance at which spine fraction crosses 0.5.
    Returns 0.0 if always above 0.5, np.inf if never crosses.
    Used for fig_s7.ipynb.
    """
    import numpy as np
    pop_bins = np.linspace(0, max_distance, num_pop_bins)
    pop_bin_centers = (pop_bins[:-1] + pop_bins[1:]) / 2
    crossing_distances = {}

    for neuron_id in outgoing_syn['pre_id'].unique():
        df_neuron = outgoing_syn[outgoing_syn['pre_id'] == neuron_id]
        counts_all, _ = np.histogram(df_neuron[dist_col], bins=pop_bins)
        spines_only = df_neuron[df_neuron['tag'] == 'spine']
        counts_spines, _ = np.histogram(spines_only[dist_col], bins=pop_bins)
        valid_bins = counts_all > 0

        if not np.any(valid_bins):
            crossing_distances[neuron_id] = np.inf
            continue

        y_vals = counts_spines[valid_bins] / counts_all[valid_bins]
        x_vals = pop_bin_centers[valid_bins]
        above_half_idx = np.where(y_vals > 0.5)[0]

        if len(above_half_idx) == 0:
            crossing_distances[neuron_id] = np.inf
        elif above_half_idx[0] == 0:
            crossing_distances[neuron_id] = 0.0
        else:
            idx1 = above_half_idx[0]
            idx0 = idx1 - 1
            x0, y0 = x_vals[idx0], y_vals[idx0]
            x1, y1 = x_vals[idx1], y_vals[idx1]
            crossing_distances[neuron_id] = x0 + (x1 - x0) * ((0.5 - y0) / (y1 - y0 + 1e-9))

    return crossing_distances