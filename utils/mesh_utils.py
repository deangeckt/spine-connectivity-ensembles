import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from meshparty import trimesh_io
from scipy.spatial import cKDTree
from standard_transform import minnie_ds


def load_full_mesh(neuron_id, mesh_path):
    local_file = os.path.join(mesh_path, f"{neuron_id}.h5")
    if os.path.exists(local_file):
        print(f"Loading mesh for neuron {neuron_id} from local cache.")
        res = trimesh_io.read_mesh_h5(local_file)
        return trimesh_io.Mesh(
            vertices=res[0],
            faces=res[1],
            edges=res[2] if len(res) > 2 else None,
            seg_id=neuron_id
        )
    raise FileNotFoundError(f"Mesh not found: {local_file}")


def plot_mesh_shaded(ax, mesh, transform,
                     color=(0.2, 0.7, 0.3),
                     light_dir=np.array([0.3, 0.5, 1.0]),
                     ambient=0.3):
    verts_um = transform.apply(mesh.vertices)
    faces    = mesh.faces
    x, y     = verts_um[:, 0], verts_um[:, 1]

    v0, v1, v2 = verts_um[faces[:, 0]], verts_um[faces[:, 1]], verts_um[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(float)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms

    light   = np.array(light_dir, dtype=float)
    light  /= np.linalg.norm(light)
    shading = ambient + (1 - ambient) * np.clip(normals @ light, 0, 1)

    r, g, b = color
    cmap = LinearSegmentedColormap.from_list(
        'mesh_color',
        [(r * 0.2, g * 0.2, b * 0.2),
         (r,       g,       b      ),
         (min(r*1.5,1), min(g*1.5,1), min(b*1.5,1))],
    )
    triang = mtri.Triangulation(x, y, faces)
    ax.tripcolor(triang, facecolors=shading, cmap=cmap,
                 vmin=0, vmax=1, edgecolors='none', rasterized=True)


def crop_triangulation(mesh_triang, xlim, ylim):
    x = mesh_triang.x
    y = mesh_triang.y
    triangles = mesh_triang.triangles
    tx = x[triangles]
    ty = y[triangles]
    in_x = (tx >= xlim[0]) & (tx <= xlim[1])
    in_y = (ty >= ylim[0]) & (ty <= ylim[1])
    mask = np.any(in_x & in_y, axis=1)
    return mtri.Triangulation(x, y, triangles[mask])


def crop_mesh(mesh, transform, xlim, ylim):
    verts_um = transform.apply(mesh.vertices)
    faces = mesh.faces
    x, y = verts_um[:, 0], verts_um[:, 1]
    fx = x[faces]
    fy = y[faces]
    in_x = (fx >= xlim[0]) & (fx <= xlim[1])
    in_y = (fy >= ylim[0]) & (fy <= ylim[1])
    mask = np.any(in_x & in_y, axis=1)

    class CroppedMesh:
        pass
    m = CroppedMesh()
    m.vertices = mesh.vertices
    m.faces = faces[mask]
    return m


def plot_branch(xlim, ylim, mesh_triang, mesh, mesh_color, syn_df=None, plot_shaded_mesh=False, ax=None, add_scale_bar=True):
    synapse_color = '#00E676'
    synapse_edge_color = 'white'
    syn_size = 50
    syn_alpha = 0.85
    mesh_lw = 0.1

    if syn_df is not None:
        branch_syn = filter_synapse_position(xlim=xlim, ylim=ylim, syn_df=syn_df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    if plot_shaded_mesh:
        r, g, b = mcolors.to_rgb(mesh_color)
        cropped_mesh = crop_mesh(mesh, minnie_ds.transform_nm, xlim, ylim)
        plot_mesh_shaded(ax, cropped_mesh, minnie_ds.transform_nm, color=(r, g, b))
    else:
        cropped_triang = crop_triangulation(mesh_triang, xlim, ylim)
        ax.triplot(cropped_triang, color=mesh_color, alpha=1, lw=mesh_lw)

    if syn_df is not None:
        ax.scatter(branch_syn.position_xt, branch_syn.position_yt, marker='o', linewidth=1.2,
                   c=synapse_color, edgecolors=synapse_edge_color, s=syn_size, alpha=syn_alpha)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    if add_scale_bar:
        fontprops = fm.FontProperties(size=12, family='Arial')
        scalebar = AnchoredSizeBar(transform=ax.transData,
                                   size=2,
                                   label='2 µm',
                                   loc='lower right',
                                   pad=1,
                                   color='black',
                                   frameon=False,
                                   size_vertical=0.05,
                                   sep=5,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)


def filter_synapse_position(xlim, ylim, syn_df):
    return syn_df[
        (syn_df.position_xt >= xlim[0]) & (syn_df.position_xt <= xlim[1]) &
        (syn_df.position_yt >= ylim[0]) & (syn_df.position_yt <= ylim[1])
    ]


def extract_dendrite_mesh_via_sk(sk, full_mesh):
    comp_series = sk.vertex_properties['compartment']
    axon_small_mask = np.zeros(sk.n_vertices, dtype=bool)
    axon_indices = comp_series[comp_series == 2].index.values
    axon_small_mask[axon_indices] = True

    tree = cKDTree(sk.vertices * 1000)  # µm → nm
    _, idx = tree.query(full_mesh.vertices, workers=-1)
    axon_full_mask = axon_small_mask[idx]

    dendrite_mesh = full_mesh.apply_mask(~axon_full_mask)
    return dendrite_mesh
