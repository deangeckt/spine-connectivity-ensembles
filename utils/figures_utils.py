from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
from stats_corr import binned_mul_plot


def add_panel_label(target, text, xy=(-0.01, 1.01), fontsize=18):
    letter_props = {'family': 'Arial', 'weight': 'bold', 'size': fontsize, 'va': 'bottom', 'ha': 'right'}
    if hasattr(target, 'transFigure'):
        target.text(xy[0], xy[1], text, transform=target.transFigure, **letter_props)
    
    elif hasattr(target, 'text2D'):
        # Route to text2D if it's a 3D axis (like your cylinders)
        target.text2D(xy[0], xy[1], text, transform=target.transAxes, **letter_props, clip_on=False)
    else:
        # Route to standard text for normal 2D axes
        target.text(xy[0], xy[1], text, transform=target.transAxes, **letter_props, clip_on=False)

def plot_clf_type_triple_binned(x_axis, x_label, df,
                                main_x_feature='ds_spine_density',
                                err_type='sem', bin_amount_text_size=12,
                         colors=None, names=None,
                         ax_list=None, use_density=False):
    df = df.copy()
    created_fig = False
    if ax_list is None:
        fig, ax_list = plt.subplots(3, 1, figsize=(5, 4*3), sharex=True, dpi=100)
        created_fig = True

    # Syn plot
    if use_density:
        y_list=['ex_incoming_synapses_density', 'inh_incoming_synapses_density']
    else:
        y_list=['num_of_ex_incoming_synapses', 'num_of_inh_incoming_synapses']
    
    # max_density = df['ds_incoming_synapses_density'].max()
    if main_x_feature == 'spine_density':
        custom_bins = [0.0, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08]
    elif main_x_feature == 'ds_spine_density':
        custom_bins = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375]
        custom_bins_centers = [0.25, 0.5, 0.75, 1.0, 1.25]
    elif main_x_feature == 'number_syn_on_spines':
        custom_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200]


    print(custom_bins)
    binned_mul_plot(
        markers=['o', ','],
        ls=['-', ':'],
        dfs=[df] * 2,
        names=names,
        x_list=[x_axis] * 2,
        y_list=y_list,
        cmap=colors,
        n_bins=custom_bins,
        bin_amount=[0],
        bin_amount_text_size=bin_amount_text_size,
        ax=ax_list[0],
        error_type=err_type
    )
    y_label = 'Density of local synapses\n(syn/μm)' if use_density else '# of local synapses/neuron'
    ax_list[0].set_ylabel(y_label)

    # Multiple contact plot
    if use_density:
        df['ex_incoming_multiple_contacts_density'] = df['ex_incoming_multiple_contacts_ratio'] / df['dendrite_length']
        df['inh_incoming_multiple_contacts_density'] = df['inh_incoming_multiple_contacts_ratio'] / df['dendrite_length']
        y_list=['ex_incoming_multiple_contacts_density', 'inh_incoming_multiple_contacts_density']
    else:
        y_list=['ex_incoming_multiple_contacts_ratio', 'inh_incoming_multiple_contacts_ratio']
    binned_mul_plot(
        markers=['o', ','],
        ls=['-', ':'],
        dfs=[df] * 2,
        names=names,
        x_list=[x_axis] * 2,
        y_list=y_list,
        cmap=colors,
        n_bins=custom_bins,
        bin_amount=[0],
        bin_amount_text_size=bin_amount_text_size,
        ax=ax_list[1],
        error_type=err_type

    )
    y_label = 'Density of synapses per connection' if use_density else 'Synapses per connection'

    ax_list[1].set_ylabel(y_label)

    # Partners plot
    if use_density:
        y_list=['ex_incoming_contacts_density', 'inh_incoming_contacts_density']
    else:
        y_list=['num_of_ex_incoming_neurons', 'num_of_inh_incoming_neurons']
    binned_mul_plot(
        markers=['o', ','],
        ls=['-', ':'],
        dfs=[df] * 2,
        names=names,
        x_list=[x_axis] * 2,
        y_list=y_list,
        cmap=colors,
        n_bins=custom_bins,
        bin_amount=[0],
        bin_amount_text_size=10,
        ax=ax_list[2],
        error_type=err_type
    )
    y_label = 'Density of local partners\n(neurons/μm)' if use_density else '# of local partners/neuron'
    ax_list[2].set_ylabel(y_label)
    ax_list[2].set_xlabel(x_label)


    if main_x_feature == 'number_syn_on_spines':
        for ax in ax_list:
            ax.set_xticks(custom_bins)
    
    if main_x_feature == 'ds_spine_density':
        for ax in ax_list:
            ax.set_xticks(custom_bins_centers)
            ax.set_xticklabels([str(b) for b in custom_bins_centers])
    if created_fig:
        return fig, ax_list

def make_cylinder_surface(r, height, n=60):
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.linspace(0, height, 2)
    T, Z = np.meshgrid(theta, z)
    return r * np.cos(T), r * np.sin(T), Z

def make_disk_verts(r, z_pos, n=60):
    theta = np.linspace(0, 2 * np.pi, n)
    # Remove the [0] center point so it only traces the perimeter
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, z_pos)
    return [list(zip(x, y, z))]

def plot_nested_cylinders(
    ax,
    R_outer=1.8,
    R_inner=0.55,
    H_outer=4.0,
    H_inner=3.8,
    n_dots=300,
    frac_red=0.80,
    seed=42,
    dot_size=1,
    show_outer=True,
    scatter_outside=False,
    ex_color='#FF0000',
    inh_color="#0072BD"
):
    rim_theta = np.linspace(0, 2 * np.pi, 300)
    z0_inner = (H_outer - H_inner) / 2

    if show_outer:
        Xo, Yo, Zo = make_cylinder_surface(R_outer, H_outer)
        ax.plot_surface(Xo, Yo, Zo, color='#c0c0c0', alpha=0.05, linewidth=0, antialiased=True)
        for z_cap in [0, H_outer]:
            poly = Poly3DCollection(make_disk_verts(R_outer, z_cap), alpha=0.06, facecolor='#c0c0c0', edgecolor='black', linewidth=0.7)
            ax.add_collection3d(poly)
            ax.plot(R_outer * np.cos(rim_theta), R_outer * np.sin(rim_theta), z_cap, color='black', linewidth=0.7, alpha=0.6)

    Xi, Yi, Zi = make_cylinder_surface(R_inner, H_inner)
    Zi = Zi + z0_inner
    ax.plot_surface(Xi, Yi, Zi, color='#e0e0e0', alpha=0.15, linewidth=0, antialiased=True)

    for z_cap in [z0_inner, z0_inner + H_inner]:
        poly = Poly3DCollection(make_disk_verts(R_inner, z_cap), alpha=0.15, facecolor='#e0e0e0', edgecolor='black', linewidth=0.8)
        ax.add_collection3d(poly)
        ax.plot(R_inner * np.cos(rim_theta), R_inner * np.sin(rim_theta), z_cap, color='black', linewidth=0.8, alpha=0.85)

    rng = np.random.default_rng(seed)
    
    # --- SCATTER INSIDE ---
    r_dot_in = R_inner * np.sqrt(rng.uniform(0, 0.85 ** 2, n_dots))
    phi_in   = rng.uniform(0, 2 * np.pi, n_dots)
    x_dot_in = r_dot_in * np.cos(phi_in)
    y_dot_in = r_dot_in * np.sin(phi_in)
    z_dot_in = rng.uniform(z0_inner + 0.05, z0_inner + H_inner - 0.05, n_dots)

    # Create a boolean mask for excitatory dots
    is_ex_in = rng.uniform(0, 1, n_dots) < frac_red

    # Plot excitatory dots inside if ex_color is provided
    if ex_color is not None:
        ax.scatter(x_dot_in[is_ex_in], y_dot_in[is_ex_in], z_dot_in[is_ex_in], 
                   c=ex_color, s=dot_size, alpha=0.85, depthshade=True)
        
    # Plot inhibitory dots inside if inh_color is provided
    if inh_color is not None:
        ax.scatter(x_dot_in[~is_ex_in], y_dot_in[~is_ex_in], z_dot_in[~is_ex_in], 
                   c=inh_color, s=dot_size, alpha=0.85, depthshade=True)


    # --- SCATTER OUTSIDE ---
    if show_outer and scatter_outside:
        # Buffer slightly so dots don't clip through the inner or outer walls
        r_min = R_inner * 1.05 
        r_max = R_outer * 0.95
        
        # Area-uniform sampling for the annulus
        r_dot_out = np.sqrt(rng.uniform(r_min**2, r_max**2, n_dots))
        phi_out   = rng.uniform(0, 2 * np.pi, n_dots)
        x_dot_out = r_dot_out * np.cos(phi_out)
        y_dot_out = r_dot_out * np.sin(phi_out)
        
        # Buffer slightly inside the top/bottom caps of the outer cylinder
        z_dot_out = rng.uniform(0.05, H_outer - 0.05, n_dots)
        
        # Create a boolean mask for excitatory dots
        is_ex_out = rng.uniform(0, 1, n_dots) < frac_red

        # Plot excitatory dots outside if ex_color is provided
        if ex_color is not None:
            ax.scatter(x_dot_out[is_ex_out], y_dot_out[is_ex_out], z_dot_out[is_ex_out], 
                       c=ex_color, s=dot_size, alpha=0.85, depthshade=True)
            
        # Plot inhibitory dots outside if inh_color is provided
        if inh_color is not None:
            ax.scatter(x_dot_out[~is_ex_out], y_dot_out[~is_ex_out], z_dot_out[~is_ex_out], 
                       c=inh_color, s=dot_size, alpha=0.85, depthshade=True)

    # --- AXIS FORMATTING ---
    ax.set_xlim(-R_outer, R_outer)
    ax.set_ylim(-R_outer, R_outer)
    ax.set_zlim(0, H_outer)
    ax.set_box_aspect([1, 1, 1.1], zoom=1.05) 
    ax.set_axis_off()
    ax.view_init(elev=15, azim=-60)

def plot_3d_box(ax_top, layer_fontsize=10):
    # 3D BOX around panel — solid block feel

    dx, dy = 0.05, 0.06  # depth offset — increase for more perspective

    front = [(0,0), (1,0), (1,1), (0,1)]  # BL, BR, TR, TL
    back  = [(0+dx, 0-dy), (1+dx, 0-dy), (1+dx, 1-dy), (0+dx, 1-dy)]

    box_color_front = '#4a4a4a'
    box_color_back  = '#888888'
    face_fill_color = '#c8c8c8'
    box_lw_front = 1.2
    box_lw_back  = 0.7
    box_alpha = 0.6
    face_alpha = 0.13  # subtle fill — adjust to taste

    trans = ax_top.transAxes

    # ---- Filled side faces (drawn BEFORE edges so edges appear on top) ----

    # Right face: front-BR, back-BR, back-TR, front-TR
    right_face = MplPolygon(
        [front[1], back[1], back[2], front[2]],
        closed=True, transform=trans, clip_on=False,
        facecolor=face_fill_color, edgecolor='none', alpha=face_alpha, zorder=0
    )
    ax_top.add_patch(right_face)

    # Top face: front-TL, front-TR, back-TR, back-TL
    top_face = MplPolygon(
        [front[3], front[2], back[2], back[3]],
        closed=True, transform=trans, clip_on=False,
        facecolor=face_fill_color, edgecolor='none', alpha=face_alpha * 1.4, zorder=0
    )
    ax_top.add_patch(top_face)

    # Bottom face: front-BL, front-BR, back-BR, back-BL
    bottom_face = MplPolygon(
        [front[0], front[1], back[1], back[0]],
        closed=True, transform=trans, clip_on=False,
        facecolor=face_fill_color, edgecolor='none', alpha=face_alpha, zorder=0
    )
    ax_top.add_patch(bottom_face)

    # ---- Back face edges (dashed, lighter) ----
    back_loop = back + [back[0]]
    bx, by = zip(*back_loop)
    ax_top.plot(bx, by, transform=trans, clip_on=False,
                color=box_color_back, lw=box_lw_back, alpha=0.4,
                linestyle='--', zorder=1)

    # ---- Depth lines (back corners to front corners) ----
    for (fx, fy), (bkx, bky) in zip(front, back):
        ax_top.plot([fx, bkx], [fy, bky], transform=trans, clip_on=False,
                    color=box_color_back, lw=box_lw_back, alpha=0.45,
                    linestyle='-', zorder=1)

    # ---- Front face edges (solid, darker — drawn last so they're on top) ----
    front_loop = front + [front[0]]
    fx_list, fy_list = zip(*front_loop)
    ax_top.plot(fx_list, fy_list, transform=trans, clip_on=False,
                color=box_color_front, lw=box_lw_front, alpha=box_alpha,
                linestyle='-', zorder=2)


    # ==========================================
    # LAYER DEPTH LINES — right face of 3D box only
    # ==========================================
    microns_ex_depths = {'L2/3': 250, 'L4': 350, 'L5': 510, 'L6': 750}
    y_data_bottom = 750  # matches ax_top.set_ylim(750, 0)

    prev_depth = 0
    depths_items = list(microns_ex_depths.items())

    for i, (layer_label, depth_um) in enumerate(depths_items):
        y_ax = 1.0 - depth_um / y_data_bottom

        # --- boundary line across the right face ---
        x_f, y_f = 1.0,      y_ax
        x_b, y_b = 1.0 + dx, y_ax - dy

        ax_top.plot([x_f, x_b], [y_f, y_b],
                    transform=trans, clip_on=False,
                    color=box_color_front, lw=0.9, alpha=0.65,
                    linestyle='--', zorder=3)

        # --- label at vertical midpoint of this layer ---
        # next boundary is either the next layer's depth or the bottom of the panel
        next_depth = depths_items[i + 1][1] if i + 1 < len(depths_items) else y_data_bottom
        mid_depth  = (prev_depth + depth_um) / 2
        y_mid_ax   = 1.0 - mid_depth / y_data_bottom

        ax_top.text(x_b + 0.012, y_mid_ax - dy * (mid_depth / depth_um if depth_um else 0),
                    layer_label,
                    transform=trans, clip_on=False,
                    fontsize=layer_fontsize, family='Arial',
                    ha='left', va='center',
                    color=box_color_front, alpha=0.9)

        prev_depth = depth_um

def plot_branch_from_image(ax, branch_idx, branches_images, original_xlim, original_ylim, y_trim=0.2, x_trim=0.2, move_lower=None):
    x0, x1 = original_xlim
    y0, y1 = original_ylim
    ax.imshow(branches_images[branch_idx], extent=[x0, x1, y1, y0])
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min + y_trim * y_range, y_max - y_trim * y_range)
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    ax.set_xlim(x_min + x_trim * x_range, x_max - x_trim * x_range)
    if move_lower is not None:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - move_lower, y_max - move_lower)
    ax.set_yticklabels("")
    ax.set_xticklabels("")
    ax.set_xticks([])
    ax.set_yticks([])


def debug_letter_placement(fig):
    grid_ax = fig.add_axes([0, 0, 1, 1], zorder=100)
    grid_ax.patch.set_alpha(0.0)
    for spine in grid_ax.spines.values():
        spine.set_visible(False)
    grid_ax.tick_params(which='both', length=0, labelsize=0)
    ticks_major = [x / 20.0 for x in range(21)]
    ticks_minor = [x / 100.0 for x in range(101)]
    grid_ax.set_xticks(ticks_major)
    grid_ax.set_yticks(ticks_major)
    grid_ax.set_xticks(ticks_minor, minor=True)
    grid_ax.set_yticks(ticks_minor, minor=True)
    grid_ax.grid(which='major', color='blue', alpha=0.3, linewidth=0.8)
    grid_ax.grid(which='minor', color='gray', alpha=0.15, linewidth=0.4)
    grid_ax.set_xlim(0, 1)
    grid_ax.set_ylim(0, 1)