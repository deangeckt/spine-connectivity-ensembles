import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.stats import pearsonr
from scipy.optimize import curve_fit


def add_reg_line(x, y, ax, color, add_reg_text='none', reg_text_font_size = 16,
                 linestyle=':', linewidth=1, line_alpha=1, only_stars=True, log=True):
    valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    x = x[valid_mask]
    y = y[valid_mask]

    pearson_r, p_value = pearsonr(x, y)
    p_str  = f'p={p_value:.1e}' if p_value < 0.001 else f'p={p_value:.3f}'

    if only_stars:
        full_only_r_text = f'$R$={pearson_r:.2f} {p_to_stars(p_value)}'
    else:
        full_only_r_text = f'$R$={pearson_r:.2f}, {p_str} {p_to_stars(p_value)}'

    slope, intercept = np.polyfit(x, y, 1)
    r_squared = pearson_r ** 2

    x_fit = np.array([np.min(x), np.max(x)])
    y_fit = slope * x_fit + intercept

    ax.plot(x_fit, y_fit, linestyle=linestyle, color=color, linewidth=linewidth, alpha=line_alpha)

    x_text, y_text = 0.98, 0.02

    _orig_text = ax.text
    def _text_once(*args, **kwargs):
        ax.text = _orig_text
        kwargs['transform'] = ax.transAxes
        kwargs['ha'] = 'right'
        kwargs['va'] = 'bottom'
        return _orig_text(*args, **kwargs)

    if add_reg_text == 'slope_and_r2':
        ax.text = _text_once
        ax.text(x_text, y_text, f'm={slope:.2f}\n$R^2$={r_squared:.2f}',
                color=color, fontsize=reg_text_font_size, ha='left', va='top', alpha=0.9,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    if add_reg_text == 'r_only':
        ax.text = _text_once
        ax.text(x_text, y_text, full_only_r_text,
                color=color, fontsize=reg_text_font_size, ha='left', va='top', alpha=0.9,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
    if log:
        print(f'R={pearson_r:.2f} R^2: {r_squared:.2f} slope: {slope:.2f} intercept: {intercept:.2f}, p={p_str}')


def add_log_curve(df, x_col, y_col, ax, lw=1.2, alpha=1, linestyle='--', color='gray',
                  fontsize=12, add_rho_text=True, only_stars=True, log=True):
    clean = df.dropna(subset=[x_col, y_col])
    clean = clean[clean[x_col] > 0]
    xs = np.linspace(clean[x_col].min(), clean[x_col].max(), 300)

    try:
        popt, _ = curve_fit(log_fit, clean[x_col], clean[y_col], maxfev=5000)
        a, b = popt
        ax.plot(xs, log_fit(xs, *popt), color=color, lw=lw, alpha=alpha, linestyle=linestyle)
        if log:
            print(f' log fit: a={a:.4f}, b={b:.4f}')
    except RuntimeError:
        print(f'log fit failed')

    spearman_rho, p = sp_stats.spearmanr(clean[x_col], clean[y_col])
    x_text, y_text = 0.98, 0.02

    p_str  = f'p={p:.1e}' if p < 0.001 else f'p={p:.3f}'
    if only_stars:
        full_text = f'$ρ$={spearman_rho:.2f} {p_to_stars(p)}'
    else:
        full_text = f'$ρ$={spearman_rho:.2f}, {p_str} {p_to_stars(p)}'

    if add_rho_text:
        _orig_text = ax.text
        def _text_once(*args, **kwargs):
            ax.text = _orig_text
            kwargs['transform'] = ax.transAxes
            kwargs['ha'] = 'right'
            kwargs['va'] = 'bottom'
            return _orig_text(*args, **kwargs)

        ax.text = _text_once
        ax.text(x_text, y_text , full_text,
                color=color, fontsize=fontsize, ha='left', va='top', alpha=1,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
    if log:
        print(f'ρ={spearman_rho:.2f}, p={p_str}')
    return full_text


def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def log_fit(x, a, b):
    return a * np.log(x + 1) + b


def binned_trend(x, y, bins=8, error_type='sem', min_n=1):
    """
    Bin x into `bins`, and compute mean and error (SEM or STD) of y in each bin.
    `bins` can be an integer (number of equally spaced bins) or an array of bin edges.
    `min_n` is the minimum number of data points required to keep a bin.
    """
    if error_type not in ['sem', 'std']:
        raise ValueError("error_type must be 'sem' or 'std'")

    # Allow custom bin edges or an integer for equal-width bins
    if isinstance(bins, int):
        bin_edges = np.linspace(x.min(), x.max(), bins + 1)
        num_bins = bins
    else:
        bin_edges = np.asarray(bins)
        num_bins = len(bin_edges) - 1

    bin_idx = np.digitize(x, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    bin_centers, y_mean, y_err, bin_counts = [], [], [], []

    for b in range(num_bins):
        mask = bin_idx == b

        # Skip the bin if it doesn't meet the minimum N threshold
        if np.sum(mask) < min_n:
            continue

        x_b = x[mask]
        y_b = y[mask]

        bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        y_mean.append(y_b.mean())

        if len(y_b) > 1:
            std_val = y_b.std(ddof=1)
            if error_type == 'sem':
                y_err.append(std_val / np.sqrt(len(y_b)))
            else:
                y_err.append(std_val)
        else:
            y_err.append(0.0)

        bin_counts.append(len(x_b))

    return np.array(bin_centers), np.array(y_mean), np.array(y_err), bin_edges, np.array(bin_counts)


def binned_mul_plot(dfs, names, x_list, y_list, cmap, n_bins=12, min_n=1, ax=None, plot_error_bars=True,
                    plot_polygons=True, markers=None, ls=None, markersize=5, bin_amount=[0], bin_amount_text_size=12, error_type='sem',
                    add_reg_line=False, add_reg_text=False, add_r_to_legend=True, add_r_parentesis=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    markers_ = markers if markers is not None else ['o']*len(dfs)
    linestyles_ = ls if ls is not None else ['-']*len(dfs)

    for i, (df, name, x_str, y_str, color, marker, linestyle) in enumerate(zip(dfs, names, x_list, y_list, cmap, markers_, linestyles_)):
        x = df[x_str].values
        y = df[y_str].values
        pearson_r, p_value = pearsonr(x, y)

        # --- PASS n_bins AND min_n TO binned_trend ---
        bin_centers, y_mean, y_err, bin_edges, bin_counts = binned_trend(x, y, bins=n_bins, error_type=error_type, min_n=min_n)

        if plot_error_bars:
            ax.errorbar(bin_centers, y_mean, yerr=y_err, fmt='o-', capsize=3, color=color, alpha=0.2)

        if add_r_parentesis:
            leg_label = f'{name} (R={pearson_r:.2f} {p_to_stars(p_value)})' if add_r_to_legend else name
        else:
            leg_label = f'{name} R={pearson_r:.2f} {p_to_stars(p_value)}' if add_r_to_legend else name

        if plot_polygons:
            ax.plot(bin_centers, y_mean,
                color=color,
                marker=marker,
                ls=linestyle,
                markersize=markersize,
                linewidth=2,
                alpha=1,
                label=leg_label)

            ax.fill_between(bin_centers,
                            y_mean - y_err,
                            y_mean + y_err,
                            color=color,
                            alpha=0.15,
                            edgecolor=None)

        # --- PLOT ANNOTATED REGRESSION LINE ---
        if add_reg_line:
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = pearson_r ** 2

            x_fit = np.array([np.min(x), np.max(x)])
            y_fit = slope * x_fit + intercept

            ax.plot(x_fit, y_fit, linestyle=':', color=color, linewidth=2, alpha=0.8)
            if add_reg_text:
                x_text = x_fit[0] + 0.75 * (x_fit[1] - x_fit[0])
                y_text = slope * x_text + intercept

                ax.text(x_text, y_text - 0.15, f'm={slope:.2f}\n$R^2$={r_squared:.2f}',
                        color=color, fontsize=10, ha='left', va='top', alpha=0.9,
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        # ----------------------------------------

        if bin_amount and i in bin_amount:
            for b, bc in enumerate(bin_centers):
                count = int(bin_counts[b])
                ax.text(bc, y_mean[b] + y_err[b], f'{count}',
                        ha='center', va='bottom', fontsize=bin_amount_text_size, alpha=0.7)
            print(f"Sum of bin counts for {name}: {int(np.sum(bin_counts))}")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.legend(frameon=False)
    return ax.get_figure(), ax
