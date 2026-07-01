import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import skeleton_plot as skelplot
from meshparty.skeleton import Skeleton
import matplotlib as mpl
from neuron import Neuron
from matplotlib.colors import LogNorm
from connectome_types import m_types, col_cell_types_ordered, cell_types
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection

comp_colors = {
    1: 'Gray',
    2: 'Blue',
    3: 'Black',
    4: 'Red'
}

comp_labels = {
    1: 'Soma',
    2: 'Axon',
    3: 'Basal Dendrite',
    4: 'Apical Dendrite',
}

full_legend_elements = []
for comp_id, color in comp_colors.items():
    if comp_id in comp_labels and 1 <= comp_id <= 4:
        full_legend_elements.append(Line2D([], [], color=color, lw=3, label=comp_labels[comp_id]))

semi_legend_elements = [
    Line2D([], [], color=comp_colors[3], lw=3, label='Dendrite'),
    Line2D([], [], color=comp_colors[2], lw=3, label='Axon'),
]

inv_comp_labels = {v: k for k, v in comp_labels.items()}

depths = {
    '2/3': 115.1112491335,
    '4': 333.4658190171,
    '5': 453.6227158132,
    '6a': 687.6482650269,
    '6b': 883.1308910545,
    'wm': 922.5861720311
}

ex_color =  "#FF0000" # red, dark red: "#A2142F"
inh_color =  "#0072BD"
uk_color = '#7f395e'
ei_palette = {"I": inh_color, "E": ex_color}

colors = [
    "#E41A1C",  # Red
    "#377EB8",  # Blue
    "#4DAF4A",  # Green
    "#984EA3",  # Purple
    "#FF7F00",  # Orange
    "#FFFF33",  # Yellow
    "#A65628",  # Brown
    "#F781BF",  # Pink
    "#999999",  # Gray
    "#66C2A5",  # Teal
    "#FC8D62",   # Coral
    "#000000",  # Black (Highest contrast against the existing list)
    "#104E8B",  # Dark Blue / Navy (Distinct from the existing mid-Blue #377EB8)
    "#6B8E23",  # Olive Drab (Distinct from Green #4DAF4A and Yellow #FFFF33)
    "#8B008B",  # Dark Magenta (Distinct from Purple #984EA3 and Pink #F781BF)
]

auto_ct_color_dict = {ct: colors[i] for i, ct in enumerate(cell_types[:11])}
ct_color_dict = {ct: colors[i] for i, ct in enumerate(col_cell_types_ordered)}
mtype_color_dict =  {mt: mpl.colormaps['tab20'](i) for i, mt in enumerate(m_types)}
cell_types = ['L23', 'L4', 'L5', 'L6', 'BC', 'MC', 'BPC', 'NGC']
ex_cell_types = ['L23', 'L4', 'L5', 'L6']
inh_cell_types = ['BC', 'MC', 'BPC', 'NGC']
four_syn_colors_on_dark_bg = ['#12AFED', '#13B413', '#E22929', '#FFC000']
ex_colors = {ct: color for ct, color in zip(ex_cell_types, four_syn_colors_on_dark_bg)}
inh_colors = {ct: color for ct, color in zip(inh_cell_types, four_syn_colors_on_dark_bg)}

microns_ex_depths = {'L2/3': 250, 'L4': 350, 'L5': 510, 'L6': 740}

    
def plot_layers(ax, color='gray', override_l6_with_y_lim=None, is_horiz=False, fontsize=12):
    prev_height = 0
    for line_label, height in microns_ex_depths.items():
        if override_l6_with_y_lim is not None and line_label == 'L6':
            height = override_l6_with_y_lim
        if is_horiz:
            ax.axvline(x=height, color=color, linestyle='--', alpha=0.55)
            ax.text((height+prev_height)/2, ax.get_ylim()[1], f' {line_label}', color=color,
                       verticalalignment='top', horizontalalignment='center', weight="bold", rotation=45, fontsize=fontsize)
        else:
            ax.axhline(y=height, color=color, linestyle='--', alpha=0.55)
            ax.text(ax.get_xlim()[1], (height+prev_height)/2, f' {line_label}', color=color, fontsize=fontsize,
                        verticalalignment='center', horizontalalignment='right', weight="bold")
        prev_height = height

def plot_synapse_on_sk(df, ax, color_by='cell_type', cmap=None, alpha=0.65,
                       min_scatter_size=3,
                       max_scatter_size=15,
                       z_order=1,
                       colors=None,
                       marker='o'):

    def syn_df_fix(df, min_=3, max_=15):
        min_size, max_size = min_, max_
        df['scatter_size'] = np.array((df['size']))
        df['scatter_size'] = (df['scatter_size'] - df['scatter_size'].min()) / (
                df['scatter_size'].max() - df['scatter_size'].min())
        df['scatter_size'] = df['scatter_size'] * (max_size - min_size) + min_size
        df['pos'] = df['pos'].apply(np.array)

    syn_df_fix(df, min_scatter_size, max_scatter_size)

    if cmap is None:
        cmap = mpl.colormaps['tab10']

    unique_values = df[color_by].unique()

    if colors is None:
        num_colors = min(len(unique_values), 10)  # Limit to 10 colors (tab10 has 10 colors)
        colors = {value: cmap(i / 10) for i, value in enumerate(unique_values[:num_colors])}

    for value in unique_values:
        subset = df[df[color_by] == value]
        ax.scatter(subset['pos'].apply(lambda x: x[0]),
                   subset['pos'].apply(lambda x: x[1]),
                   zorder=z_order, marker=marker,
                   c=[colors[value]], s=subset['scatter_size'], alpha=alpha, label=value)

    lgnd = ax.legend(scatterpoints=1, title=color_by)
    for handle in lgnd.legend_handles:
        handle.set_sizes([40.0])

    return ax

def _polylines_from_edges(vertices, edges):
    """
    Returns a list of vertex INDICES for each continuous trace.
    """
    edges = np.asarray(edges, dtype=int)
    n = len(vertices)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    deg = np.array([len(nbrs) for nbrs in adj])

    visited = set()
    polys = []

    def trace(u, v):
        line = [u, v]
        e = (min(u, v), max(u, v))
        visited.add(e)
        prev, cur = u, v
        while True:
            # stop at branch/endpoints
            if len(adj[cur]) != 2:
                break
            nxt = adj[cur][0] if adj[cur][1] == prev else adj[cur][1]
            e = (min(cur, nxt), max(cur, nxt))
            if e in visited:
                break
            line.append(nxt)
            visited.add(e)
            prev, cur = cur, nxt
        return np.asarray(line, int)

    # start traces from non-degree-2 nodes
    for u in range(n):
        if deg[u] != 2:
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in visited:
                    polys.append(trace(u, v))

    # leftover cycles (all nodes deg==2)
    for u, v in edges:
        e = (min(u, v), max(u, v))
        if e not in visited:
            polys.append(trace(u, v))

    # Return indices, not coordinates, so we can look up properties later
    return polys

def _get_trace_ribbon(xy, radii):
    """
    Calculates the polygon coordinates for a variable-width ribbon.
    """
    # 1. Calculate segment vectors
    dxy = xy[1:] - xy[:-1] 
    
    # 2. Calculate normals (perpendicular vectors)
    # Rotate 90 degrees: (x, y) -> (-y, x)
    normals = np.stack([-dxy[:, 1], dxy[:, 0]], axis=1)
    
    # Normalize
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lens[norm_lens == 0] = 1.0 
    normals = normals / norm_lens

    # 3. Calculate vertex normals (average of incoming/outgoing)
    n_start = normals[0:1]
    n_end = normals[-1:]
    n_internal = (normals[:-1] + normals[1:]) / 2.0
    
    # Re-normalize internal vectors
    n_int_lens = np.linalg.norm(n_internal, axis=1, keepdims=True)
    n_int_lens[n_int_lens == 0] = 1.0
    n_internal = n_internal / n_int_lens
    
    vertex_normals = np.concatenate([n_start, n_internal, n_end], axis=0)

    # 4. Extrude vertices by radius along the normal
    r = radii[:, None]
    left_side = xy + vertex_normals * r
    right_side = xy - vertex_normals * r
    
    # 5. Create closed polygon (Up left, down right)
    return np.concatenate([left_side, right_side[::-1]], axis=0)

def plot_skeleton_continuous(ax, sk, lw=1.0, color='#e6e6e6', alpha=1.0,
                             ignore_vertex_zero=True, coords=('x', 'y'), zorder=1):
    """
    Plots the skeleton as a continuous ribbon, excluding vertex 0.
    """
    # 1. Get trace indices
    poly_indices = _polylines_from_edges(sk.vertices, sk.edges)
    
    # 2. Get radius data (Handle Pandas Series vs Numpy safely)
    if hasattr(sk.vertex_properties['radius'], 'values'):
         all_radii = sk.vertex_properties['radius'].values
    else:
         all_radii = np.asarray(sk.vertex_properties['radius'])

    # 3. Generate Polygons
    polygons = []
    
    for indices in poly_indices:
        # --- NEW: Filter out vertex 0 ---
        # We keep only indices that are NOT 0
        if ignore_vertex_zero:
            indices = indices[indices != 0]
        
        # If filtering removed points such that we can't make a line, skip
        if len(indices) < 2:
            continue
            
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        col_indices = [axis_map[ax.lower()] for ax in coords]
        trace_coords = sk.vertices[indices][:, col_indices]
            
        # trace_xy = sk.vertices[indices][:, :2]
        
        # Apply scaling factor (lw) to the radius here
        # Note: Since we are drawing polygons, this width is in DATA UNITS (e.g., microns)
        trace_r = all_radii[indices] * lw
        
        poly = _get_trace_ribbon(trace_coords, trace_r)
        polygons.append(poly)

    # 4. Plot
    # edgecolors='none' ensures the ribbon is smooth with no border lines
    pc = PolyCollection(polygons, facecolors=color, edgecolors='none', alpha=alpha, zorder=zorder) #rasterized=True
    
    ax.add_collection(pc)
    
    # 5. Standard axes handling
    ax.autoscale_view()
    ax.set_aspect('equal')