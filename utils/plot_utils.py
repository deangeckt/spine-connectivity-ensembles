import numpy as np
from matplotlib.collections import PolyCollection

depths = {
    '2/3': 115.1112491335,
    '4': 333.4658190171,
    '5': 453.6227158132,
    '6a': 687.6482650269,
    '6b': 883.1308910545,
    'wm': 922.5861720311
}

ex_color =  "#FF0000"
inh_color =  "#0072BD"
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

cell_types = ['L23', 'L4', 'L5', 'L6', 'BC', 'MC', 'BPC', 'NGC']
inh_cell_types = ['BC', 'MC', 'BPC', 'NGC']
four_syn_colors_on_dark_bg = ['#12AFED', '#13B413', '#E22929', '#FFC000']
inh_colors = {ct: color for ct, color in zip(inh_cell_types, four_syn_colors_on_dark_bg)}

microns_ex_depths = {'L2/3': 250, 'L4': 350, 'L5': 510, 'L6': 740}

    


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