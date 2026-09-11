"""Drawing kit for the ensemble / spine-targeting schematics of figures 6 and 7.

The cartoon cells — a pyramidal soma with spiny dendrites, an inhibitory one with
smooth arms — plus the synapse dots, the wiring panels, the legend, and the two
assembled figures. Figure 6 imports `plot_inh_target_panel` for its panel E; figure 7
imports the rest.

`plot_figure7` places the panels for the paper's figure; `plot_figure7_with_data` is the
wide graphical-abstract page, which puts a data half (the column's somata, a few
reconstructed cells, their activity traces) beside the whole of `plot_figure7`. Both
place axes only — each panel's own kwargs pass straight through.

Every function has its own docstring; this module is long because the geometry is
specified in numbers, not because the drawing is subtle.
"""

import numpy as np
import plot_utils



# Defaults mirror the constants the figure notebooks already use, so the panel
# can be drawn on its own without threading five colors through every call.
_EX_COLOR      = plot_utils.ex_color   # plot_utils.ex_color
_INH_COLOR     = plot_utils.inh_color   # plot_utils.inh_color
_SPINY_COLOR   = '#7C3AED'   # activity_utils.SPINE_COLOR
_SHAFT_COLOR   = '#059669'   # activity_utils.ASPINY_COLOR

# How strongly a cell's interior is tinted with its own (wall) color, when the
# caller leaves `fill_color=None`. The cell then reads as a body washed in its
# own ink rather than as a hollow outline; 1.0 is the flat cell color, 0 the
# bare page. Both `plot_ex_neuron` and `plot_inh_neuron` take it from here, so
# every cell on the figure — the panels' own and the small ones panel d and the
# column draw — is washed to the same strength from one place. Pass
# `fill_color='white'` for the old hollow look.
NEURON_FILL_ALPHA = 0.15


def _shade(color, factor):
    """Multiply a color's RGB by `factor` (<1 darkens, >1 lightens toward white)."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    if factor <= 1:
        return (r * factor, g * factor, b * factor)
    return tuple(min(1.0, c + (1 - c) * (factor - 1)) for c in (r, g, b))


def _blend(color, alpha, bg='white'):
    """`color` composited onto `bg` at `alpha`, as an *opaque* RGB.

    The hollow cell is faked from overlapping opaque patches (see
    `_render_hollow`), so a real `alpha` on those patches would let the pass-1
    strokes show through the pass-2 fill and the seams the whole trick exists to
    hide would come back. Pre-blending onto the page color instead gives the
    identical result on a `bg`-colored background while keeping every patch
    opaque.
    """
    import matplotlib.colors as mcolors
    a = float(alpha)
    if a >= 1.0:
        return color
    c, b = mcolors.to_rgb(color), mcolors.to_rgb(bg)
    return tuple(ci * a + bi * (1 - a) for ci, bi in zip(c, b))


def _neuron_fill(fill_color, fill_alpha, edge_color, bg_color):
    """The interior color of a cartoon cell.

    `fill_color=None` (the default on both cell functions) means *tint it with
    the cell's own ink*: `edge_color` pre-blended onto the page at `fill_alpha`
    (`NEURON_FILL_ALPHA` when that is `None` too). Anything else is taken
    literally — `'white'` gives back the hollow outline the panels used to draw.
    """
    if fill_color is not None:
        return fill_color
    a = NEURON_FILL_ALPHA if fill_alpha is None else float(fill_alpha)
    return _blend(edge_color, a, bg_color)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# One spine, traced off the hand-drawn reference (vectorised screenshot), then
# normalised: base chord midpoint at the origin, `x` along the base chord's normal
# with the tip at x=1, `y` across. Every attempt to synthesise this profile from
# ellipses and smoothsteps read as a cone, a bead on a stick, or a leaf — the
# distinguishing features are the *concave* flare (constant thin neck for the first
# ~35%, then an accelerating widening) and the blunt, slightly up-tilted head. The
# gentle asymmetry is the original's hand-drawn wobble and is kept on purpose.
_SPINE_UNIT = np.array([
    (0.0000, +0.1503), (0.0487, +0.1349), (0.0997, +0.1293), (0.1510, +0.1282),
    (0.2022, +0.1292), (0.2535, +0.1311), (0.3047, +0.1342), (0.3553, +0.1426),
    (0.4031, +0.1611), (0.4489, +0.1840), (0.4931, +0.2101), (0.5351, +0.2395),
    (0.5754, +0.2713), (0.6142, +0.3048), (0.6522, +0.3393), (0.6907, +0.3733),
    (0.7307, +0.4053), (0.7738, +0.4331), (0.8225, +0.4477), (0.8706, +0.4327),
    (0.9089, +0.3988), (0.9383, +0.3569), (0.9591, +0.3101), (0.9734, +0.2608),
    (0.9833, +0.2105), (0.9899, +0.1596), (0.9945, +0.1085), (0.9974, +0.0573),
    (0.9993, +0.0060), (1.0000, -0.0453), (0.9990, -0.0965), (0.9957, -0.1477),
    (0.9891, -0.1986), (0.9779, -0.2486), (0.9604, -0.2968), (0.9348, -0.3411),
    (0.8946, -0.3719), (0.8442, -0.3794), (0.7936, -0.3721), (0.7478, -0.3495),
    (0.7055, -0.3204), (0.6646, -0.2894), (0.6243, -0.2577), (0.5833, -0.2269),
    (0.5411, -0.1977), (0.4976, -0.1706), (0.4510, -0.1492), (0.4019, -0.1345),
    (0.3516, -0.1242), (0.3008, -0.1173), (0.2496, -0.1135), (0.1984, -0.1128),
    (0.1471, -0.1152), (0.0961, -0.1207), (0.0454, -0.1282), (0.0000, -0.1503),
])
# Where along the unit spine the head is widest, and the stand-off radius to use
# there — both as fractions of `spine_len`. The panel's arrows arrive close to
# head-on, so the radius is nearer the tip distance (0.18) than the half-height.
_SPINE_HEAD_T = 0.82
_SPINE_HEAD_R = 0.28
# Where the constant-width neck ends and the flare into the head begins — the
# stretchable part of the profile, see `_stretch_neck`.
_SPINE_NECK_T = 0.35


def _stretch_neck(x, ext):
    """Unit-spine `x` with `ext` of extra length inserted into the neck.

    `spine_len` scales the traced profile whole, so a spine that stands further
    off the dendrite also has a bigger head — and past a point the heads on a
    densely spined branch start touching each other while the necks stay stubby.
    This lengthens the spine *without* resizing anything: the constant-width
    neck (`x <= _SPINE_NECK_T`) is stretched to hold `ext` more, and everything
    distal to it — the flare and the head, i.e. the whole recognisable part of
    the shape — is carried out rigidly by the same `ext`.

    `ext` is in unit-spine coordinates (fractions of `spine_len`), so the total
    length becomes `1 + ext` and the head's widest point moves from
    `_SPINE_HEAD_T` to `_SPINE_HEAD_T + ext`.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x <= _SPINE_NECK_T, x * (1.0 + ext / _SPINE_NECK_T), x + ext)


def _spine_outline(base, direction, spine_len, mirror=False, root_sink=0.06,
                   neck_ext=0.0):
    """`_SPINE_UNIT` placed on the trunk: rotated onto `direction`, scaled by
    `spine_len`, and sunk `root_sink` of its length back along the axis so the flat
    base chord ends up inside the dendrite stroke instead of showing as a stub.
    `base` is already on the trunk *centreline*, so this only has to cover the
    stroke's half-width — sinking further eats the neck, which is short to begin
    with and is most of what makes the shape read as a spine.

    `mirror` flips the traced wobble across the spine axis, so the two sides of the
    trunk are mirror images rather than 15 copies of the same asymmetry.

    `neck_ext` lengthens the neck only (`_stretch_neck`), in the same local units
    as `spine_len`: the head keeps the size `spine_len` gives it and just stands
    that much further off the dendrite.

    Returns the closed outline in local (pre-`scale`) units.
    """
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])                 # across the spine
    q = _SPINE_UNIT.copy()
    if mirror:
        q[:, 1] *= -1
    if neck_ext:
        q[:, 0] = _stretch_neck(q[:, 0], float(neck_ext) / spine_len)
    q[:, 0] -= root_sink
    return np.asarray(base, dtype=float) + spine_len * (np.outer(q[:, 0], u)
                                                        + np.outer(q[:, 1], v))


# Default dendrite tube width, as a multiple of `spine_len`, so changing the
# spine size rescales the whole drawing coherently instead of leaving fat spines
# on a hairline dendrite.
_TUBE_W_PER_SPINE = 0.50


def _tube_outline(centre, half_w, base_ext=0.0, n_cap=18, open_end=False):
    """Polygon wrapping a polyline: the dendrite drawn as a *tube* with two
    walls, rather than as a stroked line.

    `half_w` is either a scalar or one half-width per centreline point, which is
    what gives the tube its taper. The near end is left as a flat chord and
    pushed `base_ext` back along the starting tangent, so it buries inside
    whatever the tube grows out of (the soma, or the parent branch) instead of
    showing as a butt joint.

    The far end is capped with a semicircle, unless `open_end` — then there is
    no cap and the vertices are re-ordered to run *tip → base → tip*, so the one
    unwalked gap in the ring is the tip chord. Stroke that sequence as an open
    polyline (see `_render_hollow`'s `open_parts`) and the tube ends as two
    walls that simply stop: the un-closed, run-off-the-page end of the
    hand-drawn reference.
    """
    c = np.asarray(centre, dtype=float)
    w = np.broadcast_to(np.asarray(half_w, dtype=float), (len(c),)).copy()

    d = np.gradient(c, axis=0)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    if base_ext:
        c = np.vstack([c[0] - d[0] * base_ext, c])
        w = np.concatenate([w[:1], w])
        d = np.vstack([d[:1], d])
    n = np.column_stack([-d[:, 1], d[:, 0]])

    left, right = c + n * w[:, None], c - n * w[:, None]
    if open_end:
        return np.vstack([right[::-1], left])
    a0  = np.arctan2(n[-1, 1], n[-1, 0])
    ang = a0 - np.linspace(0, np.pi, n_cap)
    cap = c[-1] + w[-1] * np.column_stack([np.cos(ang), np.sin(ang)])
    return np.vstack([left, cap[1:-1], right[::-1]])


def _render_hollow(ax, parts, fill, edge, wall_lw, zorder=3, open_parts=()):
    """Draw a group of outlines as one continuous hollow shape.

    The reference drawing is a single unbroken contour: where a spine meets the
    dendrite, or a branch meets the soma, there is no seam and no line crossing
    the interior. matplotlib has no boolean union, so this fakes one in two
    passes over the same parts:

      1. every part filled *and* stroked in `edge` — a stroke straddles the
         boundary, so each part now covers itself plus `wall_lw/2` beyond it;
      2. every part filled in `fill` with no stroke — which repaints the whole
         union's interior, wiping the pass-1 strokes that fell *inside* a
         neighbouring part.

    What survives is exactly the outer rim of the union: a wall `wall_lw/2`
    points thick with mitre-free joins wherever two parts overlap. Everything
    passed in one call fuses, so parts that must *occlude* each other rather
    than merge have to be rendered as separate groups at different `zorder`.

    `open_parts` are vertex sequences whose wall stops short of closing — the
    open-ended tubes of `_tube_outline(open_end=True)`. They join the union the
    same way, except pass 1 strokes them as a *polyline* instead of a patch, so
    the segment that would close the ring never gets any ink. Their area still
    takes part in both fills, which is what leaves the gap the same `fill` color
    as the interior rather than showing the paper through a notch.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def _path(p):
        return p if isinstance(p, Path) else Path(np.asarray(p, dtype=float),
                                                  closed=True)

    paths = [_path(p) for p in parts]
    opens = [np.asarray(p, dtype=float) for p in open_parts]

    for pth in paths:
        ax.add_patch(PathPatch(pth, facecolor=edge, edgecolor=edge,
                               linewidth=wall_lw * 2, joinstyle='round',
                               capstyle='round', zorder=zorder))
    for xy in opens:
        ax.add_patch(PathPatch(_path(xy), facecolor=edge, edgecolor='none',
                               linewidth=0, zorder=zorder))
        ax.plot(xy[:, 0], xy[:, 1], color=edge, linewidth=wall_lw * 2,
                solid_joinstyle='round', solid_capstyle='round', zorder=zorder)
    for pth in paths + [_path(xy) for xy in opens]:
        ax.add_patch(PathPatch(pth, facecolor=fill, edgecolor='none',
                               linewidth=0, zorder=zorder + 0.05))


def _rot(v, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def _branch(origin, direction, length, n_spines, spine_len,
            first_t=0.30, last_t=0.98, first_side=-1, spine_angle_deg=72,
            bend=0.10, wave_amp=0.055, wave_n=1.6, wave_phase=0.35, n_pts=120,
            neck_ext=0.0):
    """One spiny neurite: a curved centreline leaving `origin` along `direction`,
    with `n_spines` mushroom outlines alternating along it.

    The curve is written in the branch's own frame — `u` along `direction`, `v`
    90° clockwise from it — so the apical trunk and the basal branches are the
    same curve, only rotated and scaled:

        along(t)  = length · t
        across(t) = bend·t^2.2 + wave_amp·t·sin(2π·wave_n·t + 2π·wave_phase)

    i.e. a slow lean (`bend`) with a hand-drawn waver on top, which reads as a
    drawn dendrite without needing a Bezier path. The waver is ramped by `t` so
    it vanishes at the base: an offset there would plant the branch off-centre
    on whatever it grows out of, whatever the phase.

    The frame flips handedness when `direction` is mirrored across x, so a pair
    of mirror-image branches needs `bend`/`wave_amp` negated on one of them (see
    `_pyramidal_geometry`'s basal loop).

    Returns {'centre': (n_pts,2) polyline, 'spines': [...], 'at': t → (xy, unit
    tangent)}; the spine dicts are the ones described in `_pyramidal_geometry`.
    """
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    v = np.array([u[1], -u[0]])
    o = np.asarray(origin, dtype=float)
    w, p0 = 2 * np.pi * wave_n, 2 * np.pi * wave_phase

    def _across(tt):
        return bend * tt ** 2.2 + wave_amp * tt * np.sin(w * tt + p0)

    def _at(tt):
        xy = o + length * tt * u + _across(tt) * v
        d  = (length * u
              + (bend * 2.2 * tt ** 1.2
                 + wave_amp * (np.sin(w * tt + p0)
                               + tt * w * np.cos(w * tt + p0))) * v)
        return xy, d / np.linalg.norm(d)

    t_grid = np.linspace(0, 1, n_pts)
    centre = o + np.outer(length * t_grid, u) + np.outer(_across(t_grid), v)

    spines = []
    for k, tt in enumerate(np.linspace(first_t, last_t, n_spines)
                           if n_spines else []):
        side = first_side * (-1) ** k
        base, tang = _at(tt)
        # +angle rotates a mostly-forward tangent to the left, hence -side.
        sp_dir = _rot(tang, -side * spine_angle_deg)
        spines.append({'base': base, 'dir': sp_dir, 'side': side, 't': float(tt),
                       # Arrow target is the widest point of the head, not the tip,
                       # so `head_r` keeps working as a symmetric stand-off.
                       # `neck_ext` carries the head out rigidly, so it lands on
                       # the offset one-for-one and `head_r` is unchanged.
                       'head': (base + sp_dir
                                * (spine_len * _SPINE_HEAD_T + neck_ext)),
                       'outline': _spine_outline(base, sp_dir, spine_len,
                                                 mirror=(side > 0),
                                                 neck_ext=neck_ext)})

    return {'centre': centre, 'spines': spines, 'at': _at}


def _pyramidal_geometry(n_spines=5, trunk_len=1.55, bend=0.10, soma_w=0.50,
                        soma_top=0.40, soma_bot=0.55, spine_len=0.215,
                        neck_ext=0.0, first_t=0.30, last_t=0.98,
                        first_side=-1, spine_angle_deg=72,
                        wave_amp=0.055, wave_n=1.6, wave_phase=0.35,
                        shaft_ts=(0.02, 0.11), soma_anchor_fracs=(0.40, 0.72),
                        n_basal=0, basal_len=0.70, basal_angle_deg=40,
                        basal_spines=3, basal_hw=0.05, basal_bury=1.15,
                        basal_min_flare=6.0,
                        basal_bend=-0.10,
                        basal_first_t=0.32, basal_last_t=0.70,
                        n_pts=120):
    """Cartoon pyramidal cell in local units: soma triangle + spiny apical trunk,
    optionally with `n_basal` spiny basal branches off the soma's bottom corners.

    Everything is expressed around a soma centered at (0, 0) and 1 unit wide-ish,
    so a panel can place several copies with one `center + local * scale`. Every
    neurite is a `_branch` — see there for the curve (`bend`, `wave_*`); the
    apical trunk leaves the soma apex straight up.

    Spines alternate sides along a branch and stick out at `spine_angle_deg`
    from the local tangent (rotated *toward* the tip, so they angle along the
    branch like the drawing rather than straight out). Each is a copy of
    `_SPINE_UNIT` — the outline traced off the reference drawing — scaled to
    `spine_len`, given `neck_ext` of extra neck (`_stretch_neck`, which leaves
    the head at `spine_len`'s size), and mirrored on alternate sides. Both
    lengths are in the same local units. They start at `first_t`, leaving
    the proximal trunk bare — that is where the shaft arrows land, and a shaft
    arrow arriving next to a spine head reads as a spine contact, which is
    exactly the wrong message for this schematic. So `shaft_ts` must stay below
    `first_t`.

    Basal branches (`n_basal`: 0 = none, 1 = left only, 2 = both) grow out of
    the soma's two bottom corners, splaying `basal_angle_deg` off straight down.
    Each axis runs through its corner and starts back up inside the soma, far
    enough that `basal_bury` × `basal_hw` (the drawn tube's half-width, which
    only the caller knows) of clearance separates the root from the slanted
    edge — see the loop below for why, and for why `basal_angle_deg` is clamped
    to at least `basal_min_flare` degrees wider than that edge. The right branch
    is the mirror image of the left (which is why `basal_bend` is negated per
    side — `_branch`'s frame flips handedness under mirroring).

    Returns a dict of local-space geometry:
      soma   (3,2) triangle vertices, apex up
      trunk  (n_pts,2) polyline — the apical centreline
      spines list of {'base', 'head', 'dir', 'side', 't', 'outline'}, bottom → top
             ('outline' is the closed mushroom polygon, see `_spine_outline`)
             — apical only; each basal carries its own list
      basal  list of {'centre', 'spines', 'side'}, left → right (empty if n_basal=0)
      shaft  list of {'xy', 'normal', 't'} anchor points on the bare trunk
      soma_side {-1/+1: [{'xy', 'normal'}, …]} points down each slanted soma
             edge at `soma_anchor_fracs` of apex → base corner, so several
             arrows can reach the same soma without stacking on one tip
    """
    y0 = soma_top - 0.12          # start inside the soma so no seam shows

    apical = _branch((0.0, y0), (0.0, 1.0), trunk_len, n_spines, spine_len,
                     first_t=first_t, last_t=last_t, first_side=first_side,
                     spine_angle_deg=spine_angle_deg, bend=bend,
                     wave_amp=wave_amp, wave_n=wave_n, wave_phase=wave_phase,
                     n_pts=n_pts, neck_ext=neck_ext)
    trunk, spines = apical['centre'], apical['spines']

    shaft = []
    for tt in shaft_ts:
        xy, tang = apical['at'](tt)
        shaft.append({'xy': xy, 'normal': np.array([-tang[1], tang[0]]), 't': float(tt)})

    apex   = np.array([0.0, soma_top])
    base_l = np.array([-soma_w, -soma_bot])
    base_r = np.array([soma_w, -soma_bot])
    soma_side = {}
    for side, corner in ((-1, base_l), (1, base_r)):
        edge = corner - apex
        n    = np.array([edge[1], -edge[0]]) * side   # outward-pointing
        n    = n / np.linalg.norm(n)
        soma_side[side] = [{'xy': apex + edge * f, 'normal': n, 'frac': float(f)}
                           for f in soma_anchor_fracs]

    basal = []
    # A basal has to flare *wider* than the soma's own slanted edge, or its axis
    # runs back out through that edge instead of up into the soma and the corner
    # can no longer be buried (see below). How much wider is what sets the cost
    # of burying it, so the flare is clamped a few degrees clear of parallel.
    edge_deg = np.degrees(np.arctan2(soma_w, soma_top + soma_bot))
    flare    = max(float(basal_angle_deg) - edge_deg, basal_min_flare)
    a = np.deg2rad(edge_deg + flare)
    for side, corner in ((-1, base_l), (1, base_r))[:int(n_basal)]:
        # Run the branch axis straight *through* the bottom corner and start it
        # far enough back up the axis that the corner is well inside the tube.
        # The tube's fill then erases the vertex and the two soma edges where
        # they enter it, and the branch reads as growing out of the corner. Root
        # it anywhere else and the vertex survives alongside the branch with a
        # sliver of a notch between them — the soma's slanted edge and the
        # branch run nearly parallel, so that sliver is long and very visible.
        d = np.array([side * np.sin(a), -np.cos(a)])
        # Backing up the axis buys clearance from the slanted edge at exactly
        # sin(flare) per unit, so a narrow flare is expensive to bury.
        back = basal_bury * basal_hw / np.sin(np.deg2rad(flare))
        # `back` varies with the angle, so add it on top of the asked-for length
        # and re-map the spine range onto the part that is actually outside the
        # soma. `basal_len` / `basal_first_t` / `basal_last_t` then all mean what
        # they say about the *visible* branch, whatever the burial costs.
        length = basal_len * trunk_len + back
        f0     = back / length
        b = _branch(corner - d * back, d,
                    length, basal_spines, spine_len,
                    first_t=f0 + basal_first_t * (1 - f0),
                    last_t=f0 + basal_last_t * (1 - f0),
                    first_side=first_side * side,
                    spine_angle_deg=spine_angle_deg, bend=basal_bend * side,
                    wave_amp=wave_amp * side, wave_n=wave_n,
                    wave_phase=wave_phase, n_pts=n_pts, neck_ext=neck_ext)
        # `t0` is where the centreline leaves the soma — everything below it is
        # the buried root, so anything placed *on* the branch by parameter (the
        # synapse dots) has to start there and not at t=0.
        basal.append({'centre': b['centre'], 'spines': b['spines'],
                      'side': side, 't0': float(f0)})

    return {'soma': np.vstack([apex, base_l, base_r]), 'trunk': trunk,
            'spines': spines, 'basal': basal, 'shaft': shaft,
            'soma_side': soma_side, 'head_r': spine_len * _SPINE_HEAD_R}


def _bowed_ring(verts, bows, n_per_edge=16):
    """Ring of `verts` with each edge replaced by a shallow parabolic arc.

    `bows[i]` bows edge i (`verts[i]` → `verts[i+1]`) by that fraction of its
    own length, positive = outward, and is at its full value at the midpoint,
    falling to nothing at both ends (4t(1-t)) so the corners stay put. A soma
    drawn with dead-straight edges reads as a drafted polygon; a hair of bow is
    what makes it read as a cell body — in particular the bottom edge wants a
    *negative* bow, arching up between the two basal roots that pull down on it.

    The orientation is taken from the ring's signed area, so "outward" means
    outward whichever way the caller happened to list the vertices.
    """
    v = np.asarray(verts, dtype=float)
    sign = np.sign(np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                          - np.roll(v[:, 0], -1) * v[:, 1])) or 1.0
    t = np.linspace(0, 1, n_per_edge, endpoint=False)
    ring = []
    for i, bow in enumerate(bows):
        a, b = v[i], v[(i + 1) % len(v)]
        d = b - a
        length = np.linalg.norm(d)
        out = np.array([d[1], -d[0]]) / length * sign
        ring.append(a + np.outer(t, d)
                    + np.outer(4 * t * (1 - t) * bow * length, out))
    return np.vstack(ring)


def _rounded_polygon_path(verts, radius):
    """Closed Path through `verts` with each corner cut to a quadratic fillet.

    matplotlib's `joinstyle='round'` only rounds the *stroke*, so a thin-edged
    triangle still has needle-sharp corners. Cutting the corner into a CURVE3
    fillet rounds the filled shape itself, which is what the hand-drawn soma in
    the reference looks like.
    """
    from matplotlib.path import Path
    verts = np.asarray(verts, dtype=float)
    pts, codes = [], []
    n = len(verts)
    for i in range(n):
        p_prev, p, p_next = verts[i - 1], verts[i], verts[(i + 1) % n]
        v_in, v_out = p - p_prev, p_next - p
        l_in, l_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        r = min(radius, l_in / 2, l_out / 2)
        a = p - v_in / l_in * r
        b = p + v_out / l_out * r
        pts.append(a); codes.append(Path.MOVETO if i == 0 else Path.LINETO)
        pts += [p, b]; codes += [Path.CURVE3, Path.CURVE3]
    pts.append(pts[0]); codes.append(Path.CLOSEPOLY)
    return Path(np.array(pts), codes)


def _ex_soma_neck_path(apex, base_l, base_r, hw, neck_r, corner_r,
                       bow_bottom, bow_side, n_arc=24, n_edge=16):
    """Path around a pyramidal soma whose apex has been replaced by a smooth
    *neck* into the apical tube.

    On each side, the slanted soma edge rises from its basal corner up to a
    tangent point `P_s`, then turns tangentially into the vertical tube wall
    via a fillet arc of radius `neck_r` — ending at `P_t` on the wall (x = ±hw).
    The polygon closes with a straight horizontal chord across the top from
    `P_t_right` to `P_t_left`; that chord lies inside the tube's fill and never
    surfaces in the render. Basal corners are rounded by `corner_r` (clamped to
    half of each incident edge), and each straight edge carries the same shallow
    parabolic bow (`4t(1-t)`) as `_bowed_ring`.

    The reason for going to the trouble: cutting the apex flat (`soma_apex_w` in
    `plot_ex_neuron`) leaves two visible corner *stacks* at each shoulder —
    slanted-edge → flat-top, then flat-top → tube-wall — and a scalar fillet
    fed through `_bowed_ring` + `_rounded_polygon_path` gets clamped by the
    ring's sub-edge spacing to nothing at those shoulders. The arc replaces
    both corners with one C¹-smooth transition where the outline itself curves
    from slanted into vertical, and there is no corner left to round.

    Returns (Path, {'P_s_l', 'P_s_r'}). The two tangent points are handed back
    so the caller's `return_geom` can expose a 4-vertex soma polygon
    (`[P_s_l, base_l, base_r, P_s_r]`) — a shape close enough to the drawn
    outline that `_soma_dot_sites` places dots on the correct slanted edges
    without knowing about the neck.
    """
    from matplotlib.path import Path
    apex   = np.asarray(apex,   dtype=float)
    base_l = np.asarray(base_l, dtype=float)
    base_r = np.asarray(base_r, dtype=float)

    bx  = float(-base_l[0])
    by_ = float(base_l[1])
    ay  = float(apex[1])
    L_slant = np.hypot(bx, ay - by_)
    ux, uy  = bx / L_slant, (ay - by_) / L_slant
    px, py  = -uy, ux                                 # perp, 90° CCW of edge

    # Left arc: solve C.x = -hw - r, C = P_s + r * perp, P_s = base_l + s * edge.
    s = (bx - hw - neck_r - neck_r * px) / bx
    P_s_l = np.array([-bx + s * bx, by_ + s * (ay - by_)])
    C_l   = P_s_l + neck_r * np.array([px, py])
    P_t_l = np.array([-hw, C_l[1]])
    # Mirror for the right side.
    P_s_r = np.array([-P_s_l[0], P_s_l[1]])
    C_r   = np.array([-C_l[0],   C_l[1]])
    P_t_r = np.array([-P_t_l[0], P_t_l[1]])

    ang_s_l = np.arctan2(P_s_l[1] - C_l[1], P_s_l[0] - C_l[0])
    ang_t_l = np.arctan2(P_t_l[1] - C_l[1], P_t_l[0] - C_l[0])
    alpha   = (ang_t_l - ang_s_l) % (2 * np.pi)       # CCW sweep, left side
    thetas_l = ang_s_l + np.linspace(0, alpha, n_arc)
    arc_l = C_l + neck_r * np.column_stack([np.cos(thetas_l), np.sin(thetas_l)])
    ang_s_r = np.arctan2(P_s_r[1] - C_r[1], P_s_r[0] - C_r[0])
    thetas_r = ang_s_r + np.linspace(0, -alpha, n_arc)   # CW sweep, right side
    arc_r = C_r + neck_r * np.column_stack([np.cos(thetas_r), np.sin(thetas_r)])

    slant_visible = L_slant * (1.0 - s)               # P_s → base_corner
    r_c = min(float(corner_r), slant_visible / 2, bx)
    u_dn_l = np.array([-ux, -uy])                     # P_s_l → base_l (unit)
    u_dn_r = np.array([ ux, -uy])                     # base_r → P_s_r reversed
    trim_bl_slant = base_l - u_dn_l * r_c             # step back up the slant
    trim_br_slant = base_r + u_dn_r * r_c             # step back up the slant (right side, going up-left)
    trim_bl_bot   = base_l + np.array([ r_c, 0.0])
    trim_br_bot   = base_r + np.array([-r_c, 0.0])

    def _bowed(a, b, bow, out):
        """`n_edge` samples from a to b, bowed by `bow` × edge-length outward
        along `out` (a unit vector). First/last coincide with a/b."""
        d = b - a
        L = np.linalg.norm(d)
        t = np.linspace(0.0, 1.0, n_edge)
        return a[None] + np.outer(t, d) + np.outer(4 * t * (1 - t) * bow * L, out)

    def _out(a, b):
        """Outward normal (rotated 90° CW of a→b direction), which for our CW
        traversal points away from the soma interior."""
        d = b - a
        L = np.linalg.norm(d)
        return np.array([d[1], -d[0]]) / L

    slant_l = _bowed(P_s_l, trim_bl_slant, bow_side, _out(P_s_l, trim_bl_slant))
    bot     = _bowed(trim_bl_bot, trim_br_bot, bow_bottom,
                     _out(trim_bl_bot, trim_br_bot))
    slant_r = _bowed(trim_br_slant, P_s_r, bow_side, _out(trim_br_slant, P_s_r))

    pts, codes = [P_t_l.copy()], [Path.MOVETO]
    # Left arc reversed: P_t_l already placed; walk down to P_s_l.
    for p in arc_l[-2::-1]:
        pts.append(p); codes.append(Path.LINETO)
    # Slanted down to trim near base_l (skip the endpoint we already have).
    for p in slant_l[1:]:
        pts.append(p); codes.append(Path.LINETO)
    # Fillet at base_l.
    pts.append(base_l);       codes.append(Path.CURVE3)
    pts.append(trim_bl_bot);  codes.append(Path.CURVE3)
    # Bowed bottom.
    for p in bot[1:]:
        pts.append(p); codes.append(Path.LINETO)
    # Fillet at base_r.
    pts.append(base_r);        codes.append(Path.CURVE3)
    pts.append(trim_br_slant); codes.append(Path.CURVE3)
    # Slanted up to P_s_r.
    for p in slant_r[1:]:
        pts.append(p); codes.append(Path.LINETO)
    # Right arc P_s_r → P_t_r.
    for p in arc_r[1:]:
        pts.append(p); codes.append(Path.LINETO)
    # CLOSEPOLY draws the straight horizontal top back to P_t_l — the segment
    # that stays hidden inside the apical tube.
    pts.append(pts[0]); codes.append(Path.CLOSEPOLY)

    return Path(np.array(pts), codes), {'P_s_l': P_s_l, 'P_s_r': P_s_r,
                                        'P_t_l': P_t_l, 'P_t_r': P_t_r}


def _draw_pyramidal(ax, center, scale, soma_color, dendrite_color,
                    dendrite_lw=2.8, spine_lw=2.4, soma_edge_lw=1.6,
                    taper=(1.5, 0.75), n_taper=6, soma_round=0.10,
                    zorder=3, geom_kw=None):
    """Draw one cartoon pyramidal cell and return its geometry in data space.

    The trunk is stroked in `n_taper` chunks whose linewidth ramps from
    `taper[0]` to `taper[1]` × `dendrite_lw` — a matplotlib line has one width,
    so this is what gives the dendrite its thick-at-the-soma taper. Round caps
    make the chunk joins invisible.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    g = _pyramidal_geometry(**(geom_kw or {}))
    c = np.asarray(center, dtype=float)

    def W(p):                      # local → world
        return c + np.asarray(p, dtype=float) * scale

    trunk_w = W(g['trunk'])
    edges   = np.linspace(0, len(trunk_w) - 1, n_taper + 1).astype(int)
    for k in range(n_taper):
        seg = trunk_w[edges[k]:edges[k + 1] + 1]
        f   = k / max(n_taper - 1, 1)
        ax.plot(seg[:, 0], seg[:, 1], color=dendrite_color,
                lw=dendrite_lw * (taper[0] + (taper[1] - taper[0]) * f),
                solid_capstyle='round', zorder=zorder)
    # Each spine is one filled mushroom outline (neck flaring into the cap), not a
    # stroked line plus a head patch — the reference drawing has no seam there.
    # A same-color stroke on top of the fill rounds the two base corners and is
    # what `spine_lw` now controls (it fattens the whole shape slightly).
    for sp in g['spines']:
        ax.add_patch(PathPatch(Path(W(sp['outline']), closed=True),
                               facecolor=dendrite_color, edgecolor=dendrite_color,
                               linewidth=spine_lw * 0.4, joinstyle='round',
                               capstyle='round', zorder=zorder))
    ax.add_patch(PathPatch(_rounded_polygon_path([W(v) for v in g['soma']],
                                                 soma_round * scale),
                           facecolor=soma_color, edgecolor=dendrite_color,
                           linewidth=soma_edge_lw, zorder=zorder + 0.1))

    return {'center': c, 'scale': scale, 'head_r': g['head_r'] * scale,
            'spines': [{**sp, 'base': W(sp['base']), 'head': W(sp['head']),
                        'outline': W(sp['outline'])}
                       for sp in g['spines']],
            'shaft': [{**s, 'xy': W(s['xy'])} for s in g['shaft']],
            'soma_side': {k: [{**a, 'xy': W(a['xy'])} for a in v]
                          for k, v in g['soma_side'].items()}}


def plot_ex_neuron(ax=None, n_spines=5, trunk_len=1.55, spine_len=0.24,
                   neck_ext=0.0,
                   n_basal=2, basal_len=0.70, basal_angle_deg=40,
                   basal_spines=3, basal_w=0.86,
                   tube_w=None, tube_taper=0.72, wall_lw=2.2,
                   soma_round=0.03, soma_sink=0.45,
                   soma_neck_r=0.15,
                   soma_bow_bottom=-0.025, soma_bow_side=0.0,
                   fill_color=None, fill_alpha=None, edge_color='#111111',
                   alpha=1.0, bg_color='white',
                   center=(0.0, 0.0), scale=1.0, rotate_deg=0.0,
                   figsize=(3.0, 4.5), dpi=150,
                   geom_kw=None, pad=0.20, return_geom=False):
    """One cartoon pyramidal cell, drawn as a hollow outline: a triangular soma
    with three spiny dendrites growing out of it — the apical one up from the
    flat top, and `n_basal` basal ones down off the bottom edge.

    Everything is a *tube* (a wall of thickness `wall_lw` around a `fill_color`
    interior), not a solid silhouette, and soma + dendrites + spines read as one
    unbroken contour. Every branch ends open: two walls that run out and stop.

    The shape comes from `_pyramidal_geometry`, the same one figure 6's cells use, so
    the spines are the same traced `_SPINE_UNIT` mushrooms; only the rendering differs
    (hollow tubes here, solid fills there).

      n_spines   number of spines along the apical dendrite.
      trunk_len  apical length in local units — larger = a taller neuron. Not µm;
                 the axes are fitted to whatever is drawn.
      spine_len  spine length in local units — really the *head* size, since the whole
                 traced profile scales with it.
      neck_ext   extra neck length added without resizing the head, so the spines stand
                 further off the dendrite. This, not `spine_len`, is the knob for making
                 them pop out — raising `spine_len` inflates the heads with them.

      n_basal    basal branches off the soma's bottom edge: 2 (default), 1 for a single
                 one on the left, 0 for the bare apical-only cell.
      basal_len  visible basal length (soma corner → tip), × `trunk_len`.
      basal_angle_deg  how far a basal leans off straight down. Clamped to a few degrees
                 wider than the soma's own slanted edge, below which the branch cannot be
                 rooted through its corner.
      basal_spines     spines per basal branch.
      basal_w    basal tube width, × the apical `tube_w`.

      tube_w     full width of the dendrite where it leaves the soma, in local units.
                 `None` ties it to `spine_len` via `_TUBE_W_PER_SPINE`.
      tube_taper width at the dendrite tip, × `tube_w`; 1.0 = parallel-sided.
      wall_lw    wall thickness in points. A stroke width, so it does not scale with
                 the axes — bump it for small figures.

      soma_round corner fillet radius of the soma's basal corners, local units. Those
                 corners sit inside the basal tubes (`basal_bury`) and are never seen.
      soma_bow_bottom  bow of the soma's bottom edge, as a fraction of its length.
                 Negative arches it up into the soma, which is what stops the edge
                 reading as a drafted straight line. Past about -0.05 it looks pinched.
      soma_bow_side    the same for the two slanted edges; positive bulges them outward.
      soma_neck_r radius of the fillet arc that turns each slanted soma edge tangentially
                 into the vertical apical tube wall, in local units. This is what closes
                 the top; must be > 0.
      soma_sink  how far the dendrite's base is buried inside the triangle, as a fraction
                 of the soma's height — enough that the tube's flat base never surfaces.

      fill_color the interior. `None` washes it with the cell's own ink (`edge_color` at
                 `fill_alpha`), so the cell reads as a body rather than an empty tube;
                 `'white'` gives the hollow look.
      fill_alpha how strongly that wash is inked, 0-1; `None` takes `NEURON_FILL_ALPHA`.
                 Ignored once `fill_color` is given.
      edge_color the wall.
      alpha      how strongly the cell is inked, 0-1. Applied by pre-blending both
                 colors onto `bg_color` (`_blend`), not as a patch alpha — so it fades
                 against the page like real transparency, but anything drawn *under* it
                 stays hidden.
      bg_color   the page color `alpha` blends against.
      rotate_deg turn the whole cell about `center`, counter-clockwise. Drawn apical-up,
                 so 90 lays it on its side with the apical pointing left — what panel D
                 wants. Applied in the local → world map, so the returned geometry
                 (including each spine's `dir`) comes back rotated with it.
      geom_kw    passed to `_pyramidal_geometry` for the shape knobs this signature does
                 not expose (`bend`, `soma_w`, `soma_top`, `soma_bot`, `first_t`,
                 `spine_angle_deg`, `wave_*`, …).

    `ax` is optional — a new (fig, ax) is created if omitted. Axes are aspect-locked so
    the trunk stays vertical and the spines don't shear.

    Returns (fig, ax), or (fig, ax, geom) with the cell's world-space geometry when
    `return_geom=True`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    kw = dict(geom_kw or {})
    kw.setdefault('n_spines', int(n_spines))
    kw.setdefault('trunk_len', float(trunk_len))
    kw.setdefault('spine_len', float(spine_len))
    kw.setdefault('neck_ext', float(neck_ext))
    kw.setdefault('n_basal', int(n_basal))
    kw.setdefault('basal_len', float(basal_len))
    kw.setdefault('basal_angle_deg', float(basal_angle_deg))
    kw.setdefault('basal_spines', int(basal_spines))
    # A bigger, blunter triangle than figure 6's: the flat top and the two basal
    # roots each eat a corner, so a soma at figure 6's proportions is left with
    # almost no triangle between them.
    kw.setdefault('soma_w', 0.62)
    kw.setdefault('soma_bot', 0.60)
    # Stop the spines short of the tip on every branch, so each one ends on bare
    # wall. A spine sitting on the last of the centreline caps the tube off, and
    # the open end (see `_tube_outline`) is the whole point of the shape.
    kw.setdefault('last_t', 0.86)

    _tw = (_TUBE_W_PER_SPINE * float(kw['spine_len']) if tube_w is None
           else float(tube_w))
    # How wide the basal tube will be drawn, so the geometry can bury each
    # bottom corner deep enough inside its own branch (see `basal_bury`).
    kw.setdefault('basal_hw', 0.5 * _tw * float(basal_w))

    g = _pyramidal_geometry(**kw)
    c = np.asarray(center, dtype=float)

    # The rotation lives in the local → world map, so everything downstream —
    # the tube outlines, the axes fit, the returned geometry — is built from
    # already-rotated points and nothing else has to know about it. `D` is the
    # same map for *directions* (spine axes), which take the turn but not the
    # translation.
    _a = np.deg2rad(float(rotate_deg))
    _R = np.array([[np.cos(_a), -np.sin(_a)], [np.sin(_a), np.cos(_a)]])

    def D(v):                                   # local → world, directions
        return np.asarray(v, dtype=float) @ _R.T

    def W(p):                                   # local → world
        return c + D(p) * scale

    hw  = 0.5 * _tw * scale

    trunk = W(g['trunk'])
    apex, base_l, base_r = g['soma']            # rows: apex, base_l, base_r

    # Each slanted soma edge turns tangentially into the vertical tube wall via a
    # `soma_neck_r` fillet arc, closed with a horizontal chord hidden inside the tube.
    _neck_r = float(soma_neck_r)
    hw_local = 0.5 * _tw

    # Depth to push the tube's flat base down into the soma (see `soma_sink`).
    sink  = float(soma_sink) * (apex[1] - base_l[1]) * scale

    # The three tubes are the `open_parts` of the group: walls that run out and
    # stop, no cap across the tip.
    tubes = [_tube_outline(trunk, np.linspace(hw, hw * float(tube_taper),
                                              len(trunk)), base_ext=sink,
                           open_end=True)]
    parts = [W(sp['outline']) for sp in g['spines']]
    # Basals: same tube + spines, narrower. Their base is already rooted inside
    # the triangle (`basal_bury`), so it only has to be pushed back far enough
    # to clear the wall — the apical's `sink` would drive it out the far side.
    bhw = hw * float(basal_w)
    for b in g['basal']:
        centre = W(b['centre'])
        tubes.append(_tube_outline(centre,
                                   np.linspace(bhw, bhw * float(tube_taper),
                                               len(centre)),
                                   base_ext=1.2 * bhw, open_end=True))
        parts += [W(sp['outline']) for sp in b['spines']]
    soma_path, neck_pts = _ex_soma_neck_path(
        apex, base_l, base_r, hw_local, _neck_r,
        corner_r=float(soma_round), bow_bottom=float(soma_bow_bottom),
        bow_side=float(soma_bow_side))
    # The neck path is built in local (pre-scale, pre-rotate, pre-translate)
    # coordinates, so map each vertex through W the same way the spine outlines and
    # the tube centre are — codes carry through unchanged.
    from matplotlib.path import Path as _Path
    parts.append(_Path(np.array([W(v) for v in soma_path.vertices]),
                       soma_path.codes))
    # 4-vert stand-in for `_soma_dot_sites`: dots land on the two slanted edges.
    soma_verts = [neck_pts['P_s_l'], base_l, base_r, neck_pts['P_s_r']]

    _fill = _neuron_fill(fill_color, fill_alpha, edge_color, bg_color)
    _render_hollow(ax, parts, _blend(_fill, alpha, bg_color),
                   _blend(edge_color, alpha, bg_color), wall_lw,
                   open_parts=tubes)

    pts = np.vstack([p.vertices if hasattr(p, 'vertices') else np.asarray(p)
                     for p in parts + tubes])
    (x0, y0), (x1, y1) = pts.min(axis=0) - pad * scale, pts.max(axis=0) + pad * scale
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')

    if return_geom:
        return fig, ax, {'center': c, 'scale': scale, 'trunk': trunk,
                         'soma': np.array([W(v) for v in soma_verts]),   # neck-arc stand-in
                         # Drawn half-widths of the tubes, so anything placed
                         # *beside* the cell (synapse dots) can clear the wall.
                         'hw': hw, 'basal_hw': bhw,
                         'head_r': g['head_r'] * scale,
                         'spines': [{**sp, 'base': W(sp['base']),
                                     'head': W(sp['head']), 'dir': D(sp['dir']),
                                     'outline': W(sp['outline'])}
                                    for sp in g['spines']],
                         'basal': [{**b, 'centre': W(b['centre']),
                                    'spines': [{**sp, 'base': W(sp['base']),
                                                'head': W(sp['head']),
                                                'dir': D(sp['dir']),
                                                'outline': W(sp['outline'])}
                                               for sp in b['spines']]}
                                   for b in g['basal']]}
    return fig, ax


# ---------------------------------------------------------------------------
# The inhibitory cell: same hollow drawing kit, a different body plan
# ---------------------------------------------------------------------------

# Default dendrite tube width of an inhibitory cell, as a multiple of the soma's
# smaller semi-axis. The excitatory cell ties its tube to `spine_len`
# (`_TUBE_W_PER_SPINE`); an aspiny cell has no spines to stay in proportion
# with, so its own soma sets the proportion instead.
_INH_TUBE_W_PER_SOMA = 0.28

# Which corner each dendrite grows out of, in the order they are taken: the two
# upper ones first, so `n_branches` < 4 still reads as a cell rather than a comb.
_INH_QUADRANTS = ((-1, 1), (1, 1), (-1, -1), (1, -1))


def _superellipse(a, b, squareness=3.0, n_pts=240, wobble=0.0, wobble_n=3,
                  wobble_phase=0.20):
    """Closed ring halfway between an ellipse and a rectangle, semi-axes `a`, `b`.

    |x/a|^s + |y/b|^s = 1: s=2 is an ellipse, s→∞ a rectangle, and the ~2.5-4
    window is the rounded-square soma of the reference drawing — round enough to
    read as a cell body, cornered enough that a dendrite can come out of each
    corner instead of sliding around a circle.

    `wobble` swells and pinches the radius by that fraction over `wobble_n`
    cycles, which is what keeps the ring from reading as a drafted primitive.
    """
    t = np.linspace(0, 2 * np.pi, int(n_pts), endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / float(squareness)
    xy = np.column_stack([a * np.sign(ct) * np.abs(ct) ** e,
                          b * np.sign(st) * np.abs(st) ** e])
    if wobble:
        xy = xy * (1 + wobble * np.sin(wobble_n * t
                                       + 2 * np.pi * wobble_phase))[:, None]
    return xy


def _superellipse_radius(direction, a, b, squareness):
    """Distance from the centre to the `_superellipse` wall along `direction`."""
    d = _unit(direction)
    s = float(squareness)
    return (abs(d[0] / a) ** s + abs(d[1] / b) ** s) ** (-1.0 / s)


def _inh_geometry(n_branches=4, branch_len=1.05, branch_angle_deg=45,
                  branch_lens=None,
                  soma_w=0.42, soma_h=0.40, soma_squareness=3.4,
                  soma_wobble=0.015, soma_pts=240,
                  n_spines=0, spine_len=0.20, spine_angle_deg=72,
                  first_t=0.30, last_t=0.86, first_side=-1,
                  bend=0.045, wave_amp=0.020, wave_n=1.6, wave_phase=0.35,
                  n_pts=120):
    """Cartoon inhibitory cell in local units: a rounded-square soma centred on
    (0, 0) with `n_branches` dendrites leaving it along its corner diagonals.

    The counterpart of `_pyramidal_geometry`, and deliberately *not* built on it:
    the excitatory cell is polar (an apex, a bottom edge, up vs down), while this
    one is symmetric about both axes — no apical, no basals, just four equal arms
    off a body that has no top. What the two share is the neurite itself: every
    arm is a `_branch`, the same curve with the same waver, so the two cells look
    drawn by the same hand.

    Each arm is rooted *on* the soma wall along its own diagonal
    (`_superellipse_radius`) and points straight out along it, so the caller only
    has to sink the tube's flat base back inside the body (`plot_inh_neuron`'s
    `soma_sink`) — with no corner to bury, none of the pyramidal cell's
    `basal_bury` machinery is needed here.

    Mirroring flips `_branch`'s frame handedness once per axis, so the curve is
    negated in the two quadrants reached by an odd number of mirrors (`sgn`
    below); that is what makes the four arms mirror images of each other rather
    than four rotated copies with the lean pointing the wrong way on two of them.

    `bend` and `wave_amp` are gentler than the pyramidal cell's: the same waver
    on four short arms at once reads as four snakes, where on one long apical it
    reads as a hand-drawn line.

    `n_spines` is 0 by default and normally stays there — an inhibitory dendrite
    is aspiny, which beside the spiny pyramidal cell is half the point of the
    figure. It is exposed only because `_branch` gives it for free.

    Returns a dict of local-space geometry:
      soma     (soma_pts,2) closed ring, counter-clockwise
      branches list of {'centre' (n_pts,2), 'spines', 'dir', 'quad', 'root_r'},
               in `_INH_QUADRANTS` order; 'spines' is the same dict list as
               `_pyramidal_geometry`'s (empty while the cell is aspiny),
               'root_r' the centre → wall distance the arm starts at
      head_r   spine head stand-off radius, as there
    """
    soma = _superellipse(soma_w, soma_h, soma_squareness, soma_pts, soma_wobble)
    a = np.deg2rad(float(branch_angle_deg))
    lens = ([float(branch_len)] * len(_INH_QUADRANTS) if branch_lens is None
            else [float(v) for v in branch_lens])

    branches = []
    for i, (sx, sy) in enumerate(_INH_QUADRANTS[:int(n_branches)]):
        d = np.array([sx * np.sin(a), sy * np.cos(a)])
        root_r = _superellipse_radius(d, soma_w, soma_h, soma_squareness)
        sgn = sx * sy
        b = _branch(d * root_r, d, lens[i % len(lens)], n_spines, spine_len,
                    first_t=first_t, last_t=last_t, first_side=first_side * sgn,
                    spine_angle_deg=spine_angle_deg, bend=bend * sgn,
                    wave_amp=wave_amp * sgn, wave_n=wave_n,
                    wave_phase=wave_phase, n_pts=n_pts)
        branches.append({'centre': b['centre'], 'spines': b['spines'],
                         'dir': d, 'quad': (sx, sy), 'root_r': float(root_r)})

    return {'soma': soma, 'branches': branches,
            'head_r': spine_len * _SPINE_HEAD_R}


def plot_inh_neuron(ax=None, n_branches=4, branch_len=1.05,
                    branch_angle_deg=45, branch_lens=None,
                    soma_w=0.42, soma_h=0.40, soma_squareness=3.4,
                    soma_wobble=0.015, soma_sink=0.75,
                    n_spines=0, spine_len=0.20,
                    tube_w=None, tube_taper=0.72, wall_lw=2.2,
                    fill_color=None, fill_alpha=None, edge_color=_INH_COLOR,
                    alpha=1.0, bg_color='white',
                    center=(0.0, 0.0), scale=1.0,
                    figsize=(4.0, 4.0), dpi=150,
                    geom_kw=None, pad=0.20, return_geom=False):
    """One cartoon inhibitory cell, drawn hollow in the style of
    `plot_ex_neuron`: a soma somewhere between a circle and a square with four
    smooth dendrites growing out of its corners, soma + dendrites forming one
    unbroken outline, and every arm ending *open* — two walls that run out and
    stop, no cap.

    The rendering is `plot_ex_neuron`'s exactly (`_tube_outline` +
    `_render_hollow`); only the body plan differs, and it differs on purpose.
    Where the pyramidal cell is polar — an apical up, two basals down off a
    triangle — this one has no top: four equal arms off a rounded-square body,
    symmetric about both axes. And it is *aspiny*, which beside the spiny
    excitatory cell is what the panel is there to say.

      n_branches       arms, taken from `_INH_QUADRANTS` (upper pair first):
                       4 for the X of the reference drawing, fewer to thin it.
      branch_len       arm length in local units, soma wall → tip. Not physical
                       µm; the axes are fitted to whatever is drawn.
      branch_angle_deg how far an arm leans off *vertical*. 45° puts it on the
                       corner diagonal of a square soma, which is where the
                       corners of a rounded square are; pushing it much either
                       way roots the arm on a flat side instead, and the cell
                       stops reading as four-cornered.
      branch_lens      per-arm lengths in the same order, overriding
                       `branch_len` — for the slightly uneven, hand-drawn look.

      soma_w, soma_h   soma semi-axes in local units (half its width / height).
      soma_squareness  2 = ellipse, → ∞ = rectangle. ~2.5-4 is the usable
                       window: below it the corners melt and the arms look
                       stuck onto a circle, above it the body reads as a box.
      soma_wobble      hand-drawn swell of the soma wall, as a fraction of its
                       radius. Small — this is life, not deformation.
      soma_sink        how far the arm's flat base is buried back inside the
                       soma, as a fraction of the wall distance it starts at, so
                       the base never surfaces through the wall. 0.75 leaves it
                       a quarter of the way out from the centre.

      n_spines         spines per arm. 0 (default) is the point — an inhibitory
                       dendrite is smooth. Non-zero draws `_SPINE_UNIT`
                       mushrooms exactly as on the excitatory cell.
      spine_len        their length in local units, when there are any.

      tube_w           full width of a dendrite at the soma, in local units.
                       `None` (default) ties it to the smaller soma semi-axis
                       via `_INH_TUBE_W_PER_SOMA`, so resizing the body rescales
                       the arms with it.
      tube_taper       width at the tip, × `tube_w`; 1.0 = parallel-sided.
      wall_lw          wall thickness in points. A stroke width, so it does not
                       scale with the axes — bump it for small figures.

      fill_color       the interior. `None` (default) washes it with the cell's
                       own ink at `fill_alpha`, exactly as on `plot_ex_neuron`;
                       `'white'` gives the hollow look.
      fill_alpha       how strongly that wash is inked, 0-1; `None` takes the
                       figure-wide `NEURON_FILL_ALPHA`.
      edge_color       the wall; defaults to the project's inhibitory blue.
      alpha, bg_color  as in `plot_ex_neuron` — faded by pre-blending onto the
                       page color, not by patch transparency (see `_blend`).

      geom_kw          passed through to `_inh_geometry` for the knobs this
                       signature does not expose (`bend`, `wave_*`, `first_t`,
                       `last_t`, `spine_angle_deg`, `soma_pts`, …).

    `ax` is optional — a new (fig, ax) is created if omitted. Axes are
    aspect-locked so the four arms stay at equal angles.

    Returns (fig, ax), or (fig, ax, geom) with the cell's world-space geometry
    when `return_geom=True`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    kw = dict(geom_kw or {})
    kw.setdefault('n_branches', int(n_branches))
    kw.setdefault('branch_len', float(branch_len))
    kw.setdefault('branch_angle_deg', float(branch_angle_deg))
    kw.setdefault('branch_lens', branch_lens)
    kw.setdefault('soma_w', float(soma_w))
    kw.setdefault('soma_h', float(soma_h))
    kw.setdefault('soma_squareness', float(soma_squareness))
    kw.setdefault('soma_wobble', float(soma_wobble))
    kw.setdefault('n_spines', int(n_spines))
    kw.setdefault('spine_len', float(spine_len))

    g = _inh_geometry(**kw)
    c = np.asarray(center, dtype=float)

    def W(p):                                   # local → world
        return c + np.asarray(p, dtype=float) * scale

    _tw = (_INH_TUBE_W_PER_SOMA * min(float(kw['soma_w']), float(kw['soma_h']))
           if tube_w is None else float(tube_w))
    hw  = 0.5 * _tw * scale

    # The soma ring and any spines are closed parts; the arms are the
    # `open_parts` of the group — walls that stop at the tip with no cap.
    parts = [W(g['soma'])]
    tubes = []
    for b in g['branches']:
        centre = W(b['centre'])
        tubes.append(_tube_outline(centre,
                                   np.linspace(hw, hw * float(tube_taper),
                                               len(centre)),
                                   base_ext=float(soma_sink) * b['root_r'] * scale,
                                   open_end=True))
        parts += [W(sp['outline']) for sp in b['spines']]

    _fill = _neuron_fill(fill_color, fill_alpha, edge_color, bg_color)
    _render_hollow(ax, parts, _blend(_fill, alpha, bg_color),
                   _blend(edge_color, alpha, bg_color), wall_lw,
                   open_parts=tubes)

    pts = np.vstack([np.asarray(p) for p in parts + tubes])
    (x0, y0), (x1, y1) = pts.min(axis=0) - pad * scale, pts.max(axis=0) + pad * scale
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')

    if return_geom:
        return fig, ax, {'center': c, 'scale': scale,
                         'soma': W(g['soma']),
                         # Drawn half-width of the arms at the soma and at the
                         # tip, so anything placed *beside* the cell clears the
                         # wall all the way along a tapered arm.
                         'hw': hw, 'hw_tip': hw * float(tube_taper),
                         'head_r': g['head_r'] * scale,
                         'branches': [{**b, 'centre': W(b['centre']),
                                       'spines': [{**sp, 'base': W(sp['base']),
                                                   'head': W(sp['head']),
                                                   'outline': W(sp['outline'])}
                                                  for sp in b['spines']]}
                                      for b in g['branches']]}
    return fig, ax


def _allocate(n, sizes):
    """Split `n` dots between branches holding `sizes` candidate sites each.

    Proportional to each branch's share of the sites, largest-remainder for the
    leftovers, and never more than a branch can hold. Doing it per branch is
    what keeps a thinned-out cell dotted everywhere: pooling the sites and
    subsampling the pool evenly aliases against the branch order, and every dot
    can land on the same neurite.
    """
    total = sum(sizes)
    n = min(int(n), total)
    if n <= 0:
        return [0] * len(sizes)
    quota = [n * s / total for s in sizes]
    out = [int(q) for q in quota]
    order = sorted(range(len(sizes)), key=lambda i: quota[i] - out[i],
                   reverse=True)
    k = 0
    while sum(out) < n:                       # hand out the remainder
        i = order[k % len(order)]
        if out[i] < sizes[i]:
            out[i] += 1
        k += 1
    return out


def _counts(n, sizes):
    """Per-group dot counts from `n`: a total to share out, or one count per
    group already — a sequence, apical first and then the basals left → right.
    Neither form ever asks a group for more sites than it has.

    `'all'` takes every candidate site. For spine dots that is the one setting
    that cannot leave an *orphan spine* — a drawn spine head with no synapse on
    it, which reads as a spine that receives nothing rather than as a spine the
    panel simply did not dot. A fixed count does that as soon as `n_spines`
    moves; `'all'` tracks whatever the cell was drawn with.
    """
    if isinstance(n, str):
        if n != 'all':
            raise ValueError(f"count must be a number, a sequence, or 'all'; got {n!r}")
        return list(sizes)
    if np.ndim(n) == 0:
        return _allocate(n, sizes)
    per = list(n) + [0] * (len(sizes) - len(n))
    return [min(int(a), s) for a, s in zip(per, sizes)]


def _take(groups, n):
    """`n` sites spread over `groups` of candidates, and evenly within each."""
    return [xy for g, m in zip(groups, _counts(n, [len(g) for g in groups]))
            for xy in _pick(g, m)]


def _take_by_wall(branches, n):
    """`n` dots over `branches`, each branch's share split by wall first.

    Each entry of `branches` is that branch's candidate sites already grouped
    into its two walls.

    Spines alternate sides as they go up a branch, so an evenly spaced
    subsample of them has a stride of 2 whenever it takes about half — and
    every dot lands on the *same* wall, leaving the other side of the dendrite
    conspicuously bare. Allocating per wall before spacing them out is what
    breaks that aliasing. The inhibitory cell's bare walls (`_wall_dot_sites`)
    need the same split for the same reason.
    """
    sizes = [sum(len(w) for w in b) for b in branches]
    return [xy for b, m in zip(branches, _counts(n, sizes)) for xy in _take(b, m)]


def _pick(items, n):
    """`n` items spread evenly over `items` (all of them if `n` is larger).

    Bin centres rather than `linspace` endpoints, so n=1 takes the middle item
    instead of the first and the chosen indices never repeat.
    """
    n = int(n)
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    idx = np.floor((np.arange(n) + 0.5) / n * len(items)).astype(int)
    return [items[i] for i in idx]


def _pt_to_data(ax, pts):
    """`pts` points as a length in data units.

    The axes these panels draw on are aspect-locked, so one number works in
    both directions. Needs the limits to be final — every caller sets them
    before it places anything quoted in points.
    """
    inv = ax.transData.inverted()
    px = float(pts) * ax.figure.dpi / 72.0
    (x0, _), (x1, _) = inv.transform((0.0, 0.0)), inv.transform((px, 0.0))
    return abs(x1 - x0)


def _dot_radius_data(ax, dot_size):
    """Radius of an `s=dot_size` scatter marker, in data units.

    `s` is the marker's *area* in points², so the drawn diameter is √s points.
    Converting it through the axes lets the dot stand-offs be quoted as the
    clearance the eye actually sees, instead of a centre offset that has to be
    re-tuned by hand every time the dot size changes.
    """
    return _pt_to_data(ax, 0.5 * np.sqrt(float(dot_size)))


def _spine_dot_sites(spines, head_r, gap):
    """One point just off the tip of each spine, along the spine's own axis."""
    return [sp['head'] + _unit(sp['dir']) * (head_r + gap) for sp in spines]


def _spine_walls(spines, head_r, gap):
    """One branch's spine sites, split into the two walls they grow off."""
    return [_spine_dot_sites([sp for sp in spines if sp['side'] == side],
                             head_r, gap) for side in (-1, 1)]


def _shaft_dot_sites(centre, half_w, spines, gap, ts, avoid_spines=0.06):
    """Points sitting just off the wall of one neurite, at the usable `t` in `ts`.

    A dot beside a spine head reads as a spine contact, which is the one thing
    this panel must not say about a shaft synapse, so two rules keep them apart:
    any `t` within `avoid_spines` of a spine's own `t` is dropped — which is
    what pushes the dots onto the bare proximal stretch of a densely spined
    branch — and whatever survives goes on the wall *opposite* the nearest
    spine. `_branch` puts a `side=-1` spine on the +normal side, hence the
    multiplier being the side itself rather than its negation.

    A branch whose every candidate is crowded out keeps the roomiest one, so it
    can still take a dot rather than silently dropping out of the allocation.
    """
    c = np.asarray(centre, dtype=float)
    d = np.gradient(c, axis=0)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    nrm = np.column_stack([-d[:, 1], d[:, 0]])

    near = [min(spines, key=lambda sp: abs(sp['t'] - t)) if spines else None
            for t in ts]
    room = [abs(sp['t'] - t) if sp else np.inf for t, sp in zip(ts, near)]
    keep = [i for i, r in enumerate(room) if r >= avoid_spines] \
        or [int(np.argmax(room))]

    sites = []
    for i in keep:
        j = int(round(float(ts[i]) * (len(c) - 1)))
        side = float(near[i]['side']) if near[i] else 1.0
        sites.append(c[j] + nrm[j] * side * (half_w + gap))
    return sites


def _soma_dot_sites(soma, gap, fracs):
    """Points just outside the soma's two slanted edges, at `fracs` along each.

    The bottom edge (lowest midpoint) is skipped — the basal roots come through
    its corners and it is bowed — and so is any edge much shorter than the rest,
    which is the flat top the apical grows out of when `soma_apex_w` cut it.
    """
    v = np.asarray(soma, dtype=float)
    ctr = v.mean(axis=0)
    edges = [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))]
    lens = np.array([np.linalg.norm(b - a) for a, b in edges])
    mids = np.array([0.5 * (a + b) for a, b in edges])
    skip = {int(np.argmin(mids[:, 1]))} | {i for i in range(len(edges))
                                           if lens[i] < 0.5 * lens.mean()}

    sites = []
    for i, (a, b) in enumerate(edges):
        if i in skip:
            continue
        e = (b - a) / lens[i]
        n = np.array([e[1], -e[0]])
        if np.dot(mids[i] - ctr, n) < 0:                # point it outward
            n = -n
        sites += [a + (b - a) * f + n * gap for f in fracs]
    return sites


def plot_synapse_dots(ax, geom, n_spine=0, n_shaft=0, n_soma=0,
                      spiny_color=_SPINY_COLOR, aspiny_color=_SHAFT_COLOR,
                      dot_size=38, dot_alpha=1.0,
                      dot_edge_color='none', dot_edge_lw=0.0,
                      spine_gap=0.01, shaft_gap=0.01, soma_gap=0.02,
                      shaft_t_range=(0.08, 0.92), n_shaft_cand=9,
                      avoid_spines=0.06, soma_fracs=(0.35, 0.60), zorder=5):
    """Synapses on a cell drawn by `plot_ex_neuron`, as plain dots.

    `geom` is that call's `return_geom=True` dict. The dot's *color is its
    target*: `n_spine` of them sit at the tip of a spine head in `spiny_color`,
    while `n_shaft` on the bare dendrite wall and `n_soma` on the soma wall are
    both `aspiny_color` — the shaft and the cell body being one class here.

    Counts are what the caller varies between a spiny and an aspiny cell; the
    sites themselves are laid out deterministically. Every candidate site is
    enumerated first (spine tips, `n_shaft_cand` points along each neurite,
    `soma_fracs` down each slanted soma edge), then each branch gets its
    proportional share and spreads it evenly (`_take`) — and a branch's spine
    dots are split between its two walls before that (`_take_spines`), or they
    all land on one side. So asking for fewer dots thins them out over the whole
    cell instead of emptying one end of it, one branch, or one wall. Asking for
    more spine dots than the cell has spines just puts one on every spine.

    `n_spine` and `n_shaft` also take **one count per branch** instead of a
    total — `(apical, basal_left, basal_right)` — for when a particular branch
    is the point of the panel and the proportional split does not give it
    enough, or `'all'` for every candidate site. `n_spine='all'` is the one
    value that guarantees no *orphan spine* — a drawn spine head carrying no
    dot — however many spines the cell ends up with.

    All three gaps are the *visible* clearance between the dot and the wall (or
    spine head) it belongs to, in the same local units as `spine_len` and scaled
    by the cell's own `scale`. The dot's own radius is added on top of them
    (`_dot_radius_data`), so changing `dot_size` never buries a dot in the wall
    and the gaps keep meaning what they say.

    Returns dict(spine=(n,2) array, aspiny=(m,2) array) of the drawn positions.
    """
    s = float(geom['scale'])
    r = _dot_radius_data(ax, dot_size)
    spine_gap, shaft_gap, soma_gap = (spine_gap * s + r, shaft_gap * s + r,
                                      soma_gap * s + r)

    spine_groups = ([_spine_walls(geom['spines'], geom['head_r'], spine_gap)]
                    + [_spine_walls(b['spines'], geom['head_r'], spine_gap)
                       for b in geom['basal']])

    # A basal's centreline starts inside the soma (`t0`), so its share of the
    # range is re-mapped onto the visible part — `shaft_t_range` then means the
    # same fraction of the *drawn* branch on every neurite.
    ts = np.linspace(*shaft_t_range, int(n_shaft_cand))
    shaft_groups = ([_shaft_dot_sites(geom['trunk'], geom['hw'], geom['spines'],
                                      shaft_gap, ts, avoid_spines)]
                    + [_shaft_dot_sites(b['centre'], geom['basal_hw'],
                                        b['spines'], shaft_gap,
                                        b['t0'] + (1 - b['t0']) * ts,
                                        avoid_spines)
                       for b in geom['basal']])

    soma_sites = _soma_dot_sites(geom['soma'], soma_gap, soma_fracs)

    spine_xy = np.array(_take_by_wall(spine_groups, n_spine),
                        dtype=float).reshape(-1, 2)
    aspiny_xy = np.array(_take(shaft_groups, n_shaft) + _pick(soma_sites, n_soma),
                         dtype=float).reshape(-1, 2)

    for xy, color in ((spine_xy, spiny_color), (aspiny_xy, aspiny_color)):
        if len(xy):
            ax.scatter(xy[:, 0], xy[:, 1], s=dot_size, color=color,
                       alpha=dot_alpha, edgecolors=dot_edge_color,
                       linewidths=dot_edge_lw, zorder=zorder)

    return {'spine': spine_xy, 'aspiny': aspiny_xy}


def plot_spiny_aspiny_pair(
        ax=None, spacing=2.30, scale=1.0, with_synapses=False,
        neuron_kw=None, spiny_kw=None, aspiny_kw=None,
        dot_kw=None, spiny_dots=None, aspiny_dots=None,
        figsize=(6.6, 4.4), dpi=150, pad=0.20):
    """Panel A: a spine-rich cell on the left, a spine-poor one on the right.

    Both are `plot_ex_neuron` cells sharing `neuron_kw`; `spiny_kw` /
    `aspiny_kw` are the per-cell overrides on top of it — normally just
    `n_spines` / `basal_spines`, high on the left and low on the right, on the
    apical and the basals alike.

    With `with_synapses`, each cell also gets its inputs as dots
    (`plot_synapse_dots`): `dot_kw` is the style both share, `spiny_dots` /
    `aspiny_dots` the per-cell counts. The left cell carries more of them, and
    more of them are purple, because it has the spines to put them on.

    Both cells go on one axes, so the limits are the union of what each
    `plot_ex_neuron` call asked for. They are set *before* any dot is drawn,
    because the dot stand-offs are quoted in points and need the final
    data→display scale; the union is then widened to hold the dots, which moves
    that scale by a hair and is why the pad is generous. `spacing` is the gap
    between the two cell centres in local units (× `scale`) — too small and the
    left cell's basal leg runs into the right cell's.

    Returns (fig, ax, [spiny_geom, aspiny_geom]).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    base = dict(neuron_kw or {})
    dots = dict(dot_kw or {})
    dx = 0.5 * float(spacing) * float(scale)
    cells = ((spiny_kw, spiny_dots, -dx), (aspiny_kw, aspiny_dots, +dx))

    geoms, boxes = [], []
    for cell_kw, _, x in cells:
        _, _, g = plot_ex_neuron(ax=ax, center=(x, 0.0), scale=scale,
                                 pad=pad, return_geom=True,
                                 **{**base, **dict(cell_kw or {})})
        # Each call fits the axes to its own cell; keep what it asked for and
        # take the union once both are down.
        boxes.append((ax.get_xlim(), ax.get_ylim()))
        geoms.append(g)

    x0, x1 = min(b[0][0] for b in boxes), max(b[0][1] for b in boxes)
    y0, y1 = min(b[1][0] for b in boxes), max(b[1][1] for b in boxes)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')

    if with_synapses:
        extra = []
        for g, (_, cell_dots, _) in zip(geoms, cells):
            xy = plot_synapse_dots(ax, g, **{**dots, **dict(cell_dots or {})})
            extra += [v for v in xy.values() if len(v)]
        if extra:
            pts = np.vstack(extra)
            lo, hi = pts.min(axis=0) - pad * scale, pts.max(axis=0) + pad * scale
            ax.set_xlim(min(x0, lo[0]), max(x1, hi[0]))
            ax.set_ylim(min(y0, lo[1]), max(y1, hi[1]))

    return fig, ax, geoms


def _wall_dot_sites(centre, half_w, gap, ts, stagger=0.5):
    """Points just off *both* walls of one smooth neurite, at each `t` in `ts`.

    The excitatory version (`_shaft_dot_sites`) has to dodge the spine heads —
    a dot beside one reads as a spine contact — so it can only ever use the wall
    opposite the nearest spine. An aspiny dendrite has no such constraint and
    the whole tube is free, which is what lets the inhibitory cell carry a much
    denser dusting than the pyramidal cell's bare shaft can.

    `half_w` is a scalar or one half-width per centreline point, so a dot on a
    tapered arm keeps its clearance all the way to the tip. The two walls come
    back separately, for `_take_by_wall` to split a branch's share between them.

    `stagger` offsets the second wall by that fraction of the `ts` spacing:
    facing pairs of dots at identical `t` turn a densely dotted dendrite into a
    ladder, and half a step off is enough to break the rungs.
    """
    c = np.asarray(centre, dtype=float)
    w = np.broadcast_to(np.asarray(half_w, dtype=float), (len(c),))
    d = np.gradient(c, axis=0)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    nrm = np.column_stack([-d[:, 1], d[:, 0]])

    lo, hi = float(ts[0]), float(ts[-1])
    step = (hi - lo) / max(len(ts) - 1, 1)
    walls = []
    for k, side in enumerate((-1.0, 1.0)):
        # The staggered wall starts half a step later and still ends at `hi`, so
        # it stays inside the asked-for range instead of running off the tip.
        tt = np.linspace(lo + k * float(stagger) * step, hi, len(ts))
        idx = [int(round(t * (len(c) - 1))) for t in tt]
        walls.append([c[j] + nrm[j] * side * (w[j] + gap) for j in idx])
    return walls


def _ring_dot_sites(ring, gap, skip_dirs=(), skip_deg=32.0):
    """Points just outside a smooth soma ring, grouped into the free arcs.

    `_soma_dot_sites` walks a polygon's *edges*, which is meaningless for a
    240-vertex superellipse — every edge is a hair long, so the whole ring reads
    as one edge per dot. This walks the ring itself instead, offsetting each
    vertex along the local outward normal.

    `skip_dirs` are the directions the dendrites leave in (`geom['branches']`'s
    `dir`): the wall within `skip_deg` of one is where the arm merges into the
    body, and it is not soma any more. Dropping those sectors cuts the ring into
    arcs, which come back separately so each gets its own share of the dots —
    otherwise an even spread over the concatenation crowds whichever arc happens
    to be listed first.
    """
    v = np.asarray(ring, dtype=float)
    ctr = v.mean(axis=0)
    d = np.gradient(v, axis=0)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    nrm = np.column_stack([d[:, 1], -d[:, 0]])
    if np.dot(v[0] - ctr, nrm[0]) < 0:                  # point them outward
        nrm = -nrm

    ang = np.degrees(np.arctan2(v[:, 1] - ctr[1], v[:, 0] - ctr[0]))
    free = np.ones(len(v), dtype=bool)
    for dr in skip_dirs:
        a0 = np.degrees(np.arctan2(dr[1], dr[0]))
        free &= np.abs((ang - a0 + 180) % 360 - 180) > float(skip_deg)

    if free.all():
        arcs = [list(range(len(v)))]
    elif not free.any():
        return []
    else:
        # Start walking just after a blocked vertex, so the arc straddling
        # index 0 comes out as one run rather than two.
        arcs, cur = [], []
        for i in np.roll(np.arange(len(v)), -int(np.flatnonzero(~free)[0])):
            if free[i]:
                cur.append(int(i))
            elif cur:
                arcs.append(cur)
                cur = []
        if cur:
            arcs.append(cur)

    return [[v[i] + nrm[i] * gap for i in arc] for arc in arcs]


def plot_inh_synapse_dots(ax, geom, n_shaft=0, n_soma=0, color=_SHAFT_COLOR,
                          dot_size=38, dot_alpha=1.0,
                          dot_edge_color='none', dot_edge_lw=0.0,
                          shaft_gap=0.01, soma_gap=0.02,
                          shaft_t_range=(0.14, 0.94), n_shaft_cand=7,
                          shaft_stagger=0.5, soma_skip_deg=32.0, zorder=5):
    """Synapses on a cell drawn by `plot_inh_neuron`, as plain dots.

    `geom` is that call's `return_geom=True` dict. Every dot is one `color` —
    the aspiny green — because on this cell there is nothing else for a synapse
    to land on: no spines, so shaft and soma are the only targets and they are
    one class. That is the whole comparison with the pyramidal cell, whose dots
    split purple/green.

    `n_shaft` is shared out over the arms (a total, or one count per arm) and
    each arm's share is split between its two walls before being spaced out
    (`_take_by_wall`), so thinning the count empties neither an arm nor a wall.
    The two walls are offset by `shaft_stagger` of a step so a densely dotted
    arm does not read as a ladder of facing pairs.
    `n_soma` is spread over the free arcs of the soma ring, skipping the sectors
    where the arms leave (`soma_skip_deg`).

    Both gaps are the *visible* clearance between dot and wall, in the cell's
    local units and scaled by its `scale`; the dot's own radius is added on top
    (`_dot_radius_data`), so `dot_size` never buries a dot in the wall. Needs
    the axes limits to be final before it is called.

    Returns dict(shaft=(n,2), soma=(m,2)) of the drawn positions.
    """
    s = float(geom['scale'])
    r = _dot_radius_data(ax, dot_size)
    shaft_gap, soma_gap = shaft_gap * s + r, soma_gap * s + r

    ts = np.linspace(*shaft_t_range, int(n_shaft_cand))
    hw, hw_tip = geom['hw'], geom.get('hw_tip', geom['hw'])
    branches = [_wall_dot_sites(b['centre'],
                                np.linspace(hw, hw_tip, len(b['centre'])),
                                shaft_gap, ts, shaft_stagger)
                for b in geom['branches']]
    arcs = _ring_dot_sites(geom['soma'], soma_gap,
                           [b['dir'] for b in geom['branches']], soma_skip_deg)

    shaft_xy = np.array(_take_by_wall(branches, n_shaft),
                        dtype=float).reshape(-1, 2)
    soma_xy = np.array(_take(arcs, n_soma), dtype=float).reshape(-1, 2)

    for xy in (shaft_xy, soma_xy):
        if len(xy):
            ax.scatter(xy[:, 0], xy[:, 1], s=dot_size, color=color,
                       alpha=dot_alpha, edgecolors=dot_edge_color,
                       linewidths=dot_edge_lw, zorder=zorder)

    return {'shaft': shaft_xy, 'soma': soma_xy}


def plot_ex_inh_pair(ax=None, spacing=3.7, scale=1.0, inh_scale=1.0, inh_dy=0.0,
                     with_synapses=False, ex_kw=None, inh_kw=None,
                     dot_kw=None, ex_dots=None, inh_dot_kw=None, inh_dots=None,
                     cell_labels=None, label_xys=((0.02, 0.94), (0.56, 0.94)),
                     label_colors=None, label_fontsize=15,
                     label_family='Arial',
                     figsize=(6.6, 4.4), dpi=150, pad=0.20):
    """Panel B: the spiny pyramidal cell on the left, an inhibitory cell on the
    right — and the two ways a synapse can land on each.

    The sibling of `plot_spiny_aspiny_pair`, and the same layout logic, but the
    two cells are different *functions* here rather than one function with
    different spine counts, so each takes its own full kwargs (`ex_kw` →
    `plot_ex_neuron`, `inh_kw` → `plot_inh_neuron`) instead of a shared base
    plus overrides. Pass the panel-A cell's kwargs as `ex_kw` and the same cell
    comes out, dots and all.

    The dots are the point of the pairing: the pyramidal cell's split
    purple/green by what they land on, while every one of the inhibitory cell's
    is green — an aspiny dendrite has no spines to offer, so shaft and soma are
    all there is, and they come thicker and denser to say so.

    The inhibitory cell is centred on the *drawn height* of the excitatory one
    rather than on y=0: the pyramidal cell hangs almost entirely above its soma,
    so aligning the two somas would leave the four-armed cell sitting at the
    pyramidal cell's ankles. `inh_dy` nudges it from there and `inh_scale`
    resizes it (× `scale`), since a cell that spans four ways needs less length
    per arm than the apical to read as the same size.

    `cell_labels` names the two cells — `('Ext.', 'Inh.')` — at `label_xys`,
    one `(x, y)` per label in **axes coordinates** ((0, 0) bottom left, (1, 1)
    top right), which is where they are nudged from. They are drawn *after* the
    limits are settled, so a label never drags the axes out around itself.
    `label_colors` defaults to each cell's own `edge_color`, so the label is
    always the colour of the cell it names.

    Returns (fig, ax, [ex_geom, inh_geom]).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    dx = 0.5 * float(spacing) * float(scale)

    _, _, ex_geom = plot_ex_neuron(ax=ax, center=(-dx, 0.0), scale=scale,
                                   pad=pad, return_geom=True,
                                   **dict(ex_kw or {}))
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()

    _, _, inh_geom = plot_inh_neuron(
        ax=ax, center=(dx, 0.5 * (y0 + y1) + float(inh_dy) * scale),
        scale=scale * float(inh_scale), pad=pad, return_geom=True,
        **dict(inh_kw or {}))
    (ix0, ix1), (iy0, iy1) = ax.get_xlim(), ax.get_ylim()

    x0, x1 = min(x0, ix0), max(x1, ix1)
    y0, y1 = min(y0, iy0), max(y1, iy1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')

    if with_synapses:
        xy = plot_synapse_dots(ax, ex_geom,
                               **{**dict(dot_kw or {}), **dict(ex_dots or {})})
        xy.update(plot_inh_synapse_dots(ax, inh_geom,
                                        **{**dict(inh_dot_kw or {}),
                                           **dict(inh_dots or {})}))
        extra = [v for v in xy.values() if len(v)]
        if extra:
            pts = np.vstack(extra)
            lo, hi = pts.min(axis=0) - pad * scale, pts.max(axis=0) + pad * scale
            ax.set_xlim(min(x0, lo[0]), max(x1, hi[0]))
            ax.set_ylim(min(y0, lo[1]), max(y1, hi[1]))

    if cell_labels:
        cols = label_colors or (dict(ex_kw or {}).get('edge_color', _EX_COLOR),
                                dict(inh_kw or {}).get('edge_color',
                                                       _INH_COLOR))
        for text, xy, color in zip(cell_labels, label_xys, cols):
            if not text:
                continue
            ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha='left',
                    va='top', color=color, family=label_family,
                    size=label_fontsize, clip_on=False, zorder=6)

    return fig, ax, [ex_geom, inh_geom]


# ---------------------------------------------------------------------------
# Panel C: a wired-up row of excitatory cells, and one inhibitory cell over it
# ---------------------------------------------------------------------------

# The geometry keywords `plot_inh_neuron` forwards to `_inh_geometry`. Panel C
# needs the inhibitory cell's *extent* before it can decide where to put it (see
# `plot_ex_circuit_panel`), so it builds the geometry once on its own first —
# which means separating the shape kwargs from the rendering ones by hand.
_INH_GEOM_ARGS = ('n_branches', 'branch_len', 'branch_angle_deg', 'branch_lens',
                  'soma_w', 'soma_h', 'soma_squareness', 'soma_wobble',
                  'n_spines', 'spine_len')


def _soma_centre(geom):
    """Centre of a drawn cell's soma, world space — where its axon starts.

    Works for either cell: `geom['soma']` is the pyramidal triangle's vertices
    or the inhibitory superellipse ring, and the mean of both is the body's
    middle. The axon is drawn *under* the cell, so starting it at the centre
    rather than at the wall costs nothing and needs no per-shape stand-off.
    """
    return np.asarray(geom['soma'], dtype=float).mean(axis=0)


def _all_spines(geom):
    """Every spine of one `plot_ex_neuron` cell — apical first, then the basals."""
    return list(geom['spines']) + [sp for b in geom['basal'] for sp in b['spines']]


def _branch_spines(geom, which=None):
    """The spines of one branch of a cell, for pinning an axon to it.

    `None` is every spine on the cell, `'apical'` the trunk above the soma,
    `'basal'` the legs below it, and an integer one particular basal (left → 0).
    Which compartment a contact lands on is a statement in this figure, so it
    has to be sayable per connection rather than left to whichever spine happens
    to be nearest.
    """
    if which is None:
        return _all_spines(geom)
    if which == 'apical':
        return list(geom['spines'])
    if which == 'basal':
        return [sp for b in geom['basal'] for sp in b['spines']]
    return list(geom['basal'][int(which)]['spines'])


def _arc_rad(src, dst, rad):
    """`arc3` curvature that bows the connector *upward* whichever way it runs.

    matplotlib puts the control point at `mid + rad·(dy, -dx)` — 90° clockwise
    off the A→B direction — so one fixed `rad` bows a rightward line down and a
    leftward line up, and a row of cells wired in both directions comes out with
    half its axons sagging into the basal thicket. Flipping the sign with the
    direction is what keeps them all arcing over the row, the way the reference
    sketch draws them: positive `rad` means "bow up" either way.
    """
    return float(rad) * (-1.0 if np.asarray(dst)[0] >= np.asarray(src)[0] else 1.0)


def _arc3_control(a, b, rad):
    """Where `arc3,rad=…` puts its quadratic control point between two points.

    Everything an axon is drawn on is built off this, so a stroke keeps bowing
    the way `rad` has always made it bow however the rest of it is assembled.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    return 0.5 * (a + b) + float(rad) * np.array([d[1], -d[0]])


def _axon_cubic(a, b, drop, rad, smooth):
    """The cubic one axon runs on: `(start, q1, q2, end)`, `start` being the
    foot of the straight descent out of the soma rather than the soma itself.

    Drawing the descent and the run as two pieces — a straight `plot` down and
    an `arc3` on from its end — makes them meet at whatever angle the target
    happens to lie at, and a schematic axon that turns a hard corner reads as a
    circuit diagram's wire rather than as a cell's process. Here the descent
    ends in a cubic whose first control point carries *straight on down*, so the
    curve leaves it tangentially and the join disappears; `smooth` is how far
    that control point reaches, as a fraction of the remaining run, and so how
    wide the turn comes out. The far control point is `arc3`'s own quadratic
    one raised to cubic, which leaves the arrival angle and the bow `rad` asks
    for exactly as they were — only the corner changes.

    With `drop` at 0 there is no corner to round, and the whole thing collapses
    back to the plain `arc3` between the two points.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a1 = a - [0.0, float(drop)]
    c = _arc3_control(a1, b, rad)
    q1 = (a1 - [0.0, float(smooth) * float(np.linalg.norm(b - a1))] if drop
          else a1 + (2.0 / 3.0) * (c - a1))
    return a1, q1, b + (2.0 / 3.0) * (c - b), b


def _axon_stroke(cubic, start=None):
    """One drawn stroke: a straight run from `start` into the cubic's first
    point (skipped when `start` is None), then the cubic itself."""
    from matplotlib.path import Path
    verts = ([np.asarray(start, dtype=float)] if start is not None else []) \
        + [np.asarray(p, dtype=float) for p in cubic]
    codes = ([Path.MOVETO, Path.LINETO] if start is not None
             else [Path.MOVETO]) + [Path.CURVE4] * 3
    return Path(np.asarray(verts, dtype=float), codes)


def _axon_path(a, b, drop, rad, smooth):
    """One axon, soma `a` → spine `b`, as a single stroke with no kink in it."""
    return _axon_stroke(_axon_cubic(a, b, drop, rad, smooth),
                        start=(a if drop else None))


def _bezier_split(cubic, t, fallback=(0.0, -1.0)):
    """de Casteljau: the cubic cut at `t` → (the piece before the cut, the unit
    tangent there). The piece after it is not wanted — what continues past a
    fork is a branch of its own, not the rest of this curve.

    A cubic collapsed to a point has no tangent to report, and `fallback` stands
    in: a stem whose aim lands exactly on the foot of its own descent is all
    descent, so what it hands the branches is the direction it fell in.
    """
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in cubic)
    t = float(t)
    u, v, w = p0 + (p1 - p0) * t, p1 + (p2 - p1) * t, p2 + (p3 - p2) * t
    x, y = u + (v - u) * t, v + (w - v) * t
    d = y - x
    return ((p0, u, x, x + d * t),
            _unit(d) if np.linalg.norm(d) > 1e-12 else _unit(fallback))


def _axon_tree(a, bs, drop, rads, smooth, fork, spread):
    """The axon of a cell that contacts several others, as one **branching**
    arbor: a shared stem out of the soma that forks into a branch per target.

    A cell with two outputs drawn as two separate strokes puts two lines out of
    one soma running side by side for as far as their targets agree, which reads
    as two axons — and on a staggered row they promptly cross each other, since
    the lower target's line starts above the higher one's. One axon that leaves
    and *branches* is both what the cell actually has and the shape `plot_axon_
    panel` draws next door, so panel C's wiring and panel D's arbor come out in
    the same hand.

    The stem is the ordinary `_axon_cubic` aimed at the targets' centroid (but
    never above the foot of the descent — see below), cut off at `fork` of the
    way along (`_bezier_split`), so it descends and turns exactly as a lone axon
    would, toward where its targets are as a group. Each
    branch then leaves the cut on a direction `spread` of the way from the
    stem's own tangent round to its target's bearing, and curves from there on
    the same `smooth` / `rad` terms as any other axon.

    `spread` is that turn, as a fraction of the angle between the stem and the
    branch's own bearing — 0 leaves tangent to the stem, 1 leaves pointing
    straight at the target — and it is what keeps the fork a bifurcation rather
    than a hairpin. Tangent is the smoothest possible crotch and fine while the
    targets lie on ahead, but this row is staggered and a cell often sends one
    branch off down the row and one back up it: a target *behind* the fork then
    has its branch turn through most of 180° inside the crotch. Turning it at
    the fork instead splits that between the fork and the run, which is also how
    `_axon_tree_geometry` sets panel D's arbor off its stem.

    Returns (stem Path, [branch Path per entry of `bs`]).
    """
    a = np.asarray(a, dtype=float)
    bs = [np.asarray(x, dtype=float) for x in bs]
    aim = np.mean(bs, axis=0)
    # ...but never at anything above the foot of the descent. A cell that sends
    # one branch off down the row and one back up it has its targets' centroid
    # level with its own soma, and a stem aimed *there* runs straight down for
    # `drop` and then climbs back up through itself: the fork lands above the
    # foot and what is drawn is a needle hanging below the crotch. Holding the
    # aim down at the foot makes the stem the descent and nothing more, and the
    # cell forks at the bottom of it — which is the shape that was wanted.
    aim[1] = min(aim[1], a[1] - float(drop))
    stem_cubic, tang = _bezier_split(
        _axon_cubic(a, aim, drop, float(np.mean(rads)), smooth), fork)
    f = stem_cubic[-1]
    branches = []
    for b, rad in zip(bs, rads):
        u = _unit(b - f)
        turn = np.degrees(np.arctan2(tang[0] * u[1] - tang[1] * u[0],
                                     tang @ u))          # stem → target, signed
        d = _rot(tang, turn * float(spread))
        c = _arc3_control(f, b, rad)
        q1 = f + d * (float(smooth) * float(np.linalg.norm(b - f)))
        branches.append(_axon_stroke((f, q1, b + (2.0 / 3.0) * (c - b), b)))
    return _axon_stroke(stem_cubic, start=(a if drop else None)), branches


def _trunk_contact(geom, t, src):
    """Where a connector aiming at this cell's apical *shaft* touches it, as
    `(point, outward normal)`: the trunk point at `t`, pushed sideways out to
    the cell's own **spine envelope** on the side the connector arrives from.

    Aiming at the centreline and standing the arrow off along its own path
    instead does not work, for two compounding reasons. The spines reach
    several times the tube's half-width out from the centreline, so a stand-off
    tuned to clear the tube leaves the head among the spines; and a `shrink`
    retreats *along the path*, which buys no sideways room at all when the
    connector arrives running parallel to the dendrite — exactly the case where
    a flat bar's overhang lands on it. Offsetting along the trunk's own normal
    is angle-independent and fixes both, and it is what the returned normal is
    for: everything the caller then stacks outside the cell (the synapse dot,
    the bar's clearance) goes along it.

    The envelope is measured off the drawn spine outlines rather than assumed
    from `spine_len`, so it stays right whatever the spines are set to.
    """
    trunk = np.asarray(geom['trunk'], dtype=float)
    j = int(np.clip(round(float(t) * (len(trunk) - 1)), 0, len(trunk) - 1))
    d = np.gradient(trunk, axis=0)[j]
    n = _unit([-d[1], d[0]])
    if np.dot(np.asarray(src, dtype=float) - trunk[j], n) < 0:   # face the source
        n = -n
    reach = float(geom['hw'])
    for sp in geom['spines']:
        reach = max(reach, float(((np.asarray(sp['outline'], dtype=float)
                                   - trunk[j]) @ n).max()))
    return trunk[j] + n * reach, n


def _spine_contact(geom, src, which='apical', rank=0):
    """Where a connector lands when it is meant to contact a **spine**, as
    `(point, outward normal)`: just off the head of one of `which`'s spines
    (`_branch_spines`), along that spine's own axis.

    The candidates are the spines pointing back toward the source — a contact
    on a head on the far side would have the line cross the dendrite to reach
    it — ordered **tip of the branch first**, so `rank=0` is the distal-most
    reachable spine and stepping `rank` walks back down toward the soma. Distal
    first because what this picks is the apical tuft: a connector coming down
    from a cell above the row arrives at the top of the tree, and the top of
    the tree is where the statement about it targeting a spine is legible.
    """
    spines = _branch_spines(geom, which)
    sx = 1.0 if np.asarray(src, dtype=float)[0] >= _soma_centre(geom)[0] else -1.0
    cand = [sp for sp in spines if np.sign(sp['dir'][0]) == sx] or list(spines)
    cand.sort(key=lambda sp: -float(sp['t']))
    sp = cand[int(rank) % len(cand)]
    n = _unit(sp['dir'])
    return np.asarray(sp['head'], dtype=float) + n * float(geom['head_r']), n


def _soma_contact(geom, t, src):
    """Where a connector lands when it is meant to contact the **soma**, as
    `(point, outward normal)`: `t` of the way down the slanted soma edge facing
    the source, `t=0` being its top end and `t=1` the bottom corner.

    Only the two slanted edges are contactable, the same two `_soma_dot_sites`
    puts dots on and for the same reasons — the bottom edge is bowed and has
    the basal roots coming through its corners, and the short top one is where
    the apical grows out. Of those, the edge whose outward normal points most
    nearly at the source is the one the connector reaches without crossing the
    cell, which for a cell sitting above the row means the near flank of
    whichever neighbour it is over.
    """
    v = np.asarray(geom['soma'], dtype=float)
    ctr = v.mean(axis=0)
    edges = [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))]
    lens = np.array([np.linalg.norm(b - a) for a, b in edges])
    mids = np.array([0.5 * (a + b) for a, b in edges])
    skip = {int(np.argmin(mids[:, 1]))} | {i for i in range(len(edges))
                                           if lens[i] < 0.5 * lens.mean()}
    keep = [i for i in range(len(edges)) if i not in skip] or list(range(len(edges)))

    u = _unit(np.asarray(src, dtype=float) - ctr)
    best, best_n, score = None, None, -np.inf
    for i in keep:
        a, b = edges[i]
        e = (b - a) / lens[i]
        n = np.array([e[1], -e[0]])
        if np.dot(mids[i] - ctr, n) < 0:                # point it outward
            n = -n
        if float(n @ u) > score:
            score, best_n = float(n @ u), n
            # top end first, so `t` runs apex → base corner on either flank
            best = (a, b) if a[1] >= b[1] else (b, a)
    a, b = best
    return a + (b - a) * float(np.clip(t, 0.0, 1.0)), best_n


def _per_target(value, cell, k, default=None):
    """One inhibitory connector's share of a panel-level setting.

    Three ways of saying it, so each knob can be written the way it reads best:
    a **dict** keyed by the target's own 1-based cell number (`{1: 'soma'}` —
    survives reordering `inh_targets`, and says which cell it means), a
    **sequence** running alongside `inh_targets` (`k` is the position in it), or
    a bare value every connector gets. Strings count as bare values, not as
    sequences of characters.
    """
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(int(cell) + 1, default)
    if isinstance(value, str) or np.ndim(value) == 0:
        return value
    return value[int(k) % len(value)]


def _axon_targets(geoms, conns, ranks=None, branches=None):
    """The spine each connection's axon ends on: {(src, dst) → spine dict}.

    An axon may only land on a spine that points back *toward* its source —
    anything else would have the line cross the target's own dendrite to reach
    a head on the far side, which reads as passing through the cell rather than
    contacting it. So the candidates for `(src, dst)` are `dst`'s spines whose
    direction has the same horizontal sign as the source's bearing (apical and
    basal alike), nearest head first.

    A target's incoming axons are handed out in that order and never share a
    spine, so two inputs onto the same cell land on two different heads instead
    of stacking two dots on one. `ranks[(src, dst)]` steps that choice further
    down the list when the nearest reachable spine is not the one wanted, and
    `branches[(src, dst)]` narrows it to one compartment first (`_branch_spines`)
    — nearest-first would otherwise always pick a basal, those being the part of
    the cell an axon coming along the row passes closest to.
    """
    ranks, branches = dict(ranks or {}), dict(branches or {})
    by_dst = {}
    for src, dst in conns:
        by_dst.setdefault(dst, []).append(src)

    out = {}
    for dst, srcs in by_dst.items():
        g, c = geoms[dst], _soma_centre(geoms[dst])
        used = []
        for src in srcs:
            s = _soma_centre(geoms[src])
            sx = 1.0 if s[0] >= c[0] else -1.0
            spines = _branch_spines(g, branches.get((src, dst)))
            cand = [sp for sp in spines
                    if np.sign(sp['dir'][0]) == sx] or list(spines)
            cand.sort(key=lambda sp: np.linalg.norm(np.asarray(sp['head']) - s))
            pool = [sp for sp in cand if not any(sp is u for u in used)] or cand
            sp = pool[int(ranks.get((src, dst), 0)) % len(pool)]
            used.append(sp)
            out[(src, dst)] = sp
    return out


def plot_ex_circuit_panel(
        ax=None, n_ex=4, spacing=3.50, scale=1.0, ex_kw=None,
        ex_dy=(0.0, -1.10, -0.55, -0.15), ex_xy=None,
        connections=((1, 2), (1, 3), (3, 2), (3, 4), (4, 3)),
        axon_color=None, axon_lw=1.0, axon_alpha=1.0, axon_drop=0.95,
        axon_rad=-0.05, axon_rads=None, axon_smooth=0.25,
        axon_fork=0.40, axon_forks=None, axon_spread=0.60, axon_zorder=1.5,
        spine_rank=None, spine_branch=None,
        syn_color=_SPINY_COLOR, syn_size=26, syn_alpha=1.0,
        syn_edge_color='none', syn_edge_lw=0.0, syn_gap=0.02, syn_zorder=6,
        with_inh=True, inh_kw=None, inh_scale=1.0, inh_dx=0.0, inh_dy=0.60,
        inh_targets=None, inh_target_t=0.55, inh_target_kind='trunk',
        inh_spine_branch='apical', inh_spine_rank=0,
        inh_line_color=None, inh_lw=1.2, inh_alpha=1.0,
        inh_rad=0.12, inh_rads=None, inh_src_offsets=None,
        inh_drop=0.0, inh_smooth=0.25, inh_fork=0.45, inh_spread=0.60,
        inh_bar_width=5.0, inh_bar_length=0.0, inh_bar_scale=3.0,
        inh_gap=12.0, inh_zorder=1.5,
        inh_syn_size=None, inh_syn_colors=None, inh_syn_gap=0.02,
        inh_syn_alpha=1.0, inh_syn_edge_color='none', inh_syn_edge_lw=0.0,
        inh_syn_zorder=6,
        figsize=(9.5, 5.0), dpi=150, pad=0.20):
    """Panel C: a row of excitatory cells wired to one another, under one
    inhibitory cell that contacts every one of them.

    `n_ex` `plot_ex_neuron` cells in a row sharing `ex_kw`, numbered 1…n left to right.
    `connections` is a list of `(pre, post)` pairs in those numbers, each drawn as an
    axon leaving the pre-synaptic soma and ending on a spine of the post-synaptic cell
    with a `syn_color` dot. `_axon_targets` picks the spine — always one facing the
    source; `spine_branch` and `spine_rank`, both `{(pre, post): …}`, override it.
    A cell with several outputs gets one axon that branches (`_axon_tree`) rather than
    several leaving side by side.

    Above the row, one `plot_inh_neuron` cell reaches every cell in `inh_targets`
    (default: all), likewise as one forking axon. Each branch ends in a flat bar
    (`arrowstyle='-['`) on a named compartment of its target, with a synapse dot in
    that compartment's colour:

      'soma'    off the slanted soma flank facing the cell, `inh_target_t` down it
      'spine'   off a spine head of `inh_spine_branch`, distal-most first
      'trunk'   off the bare apical shaft at `inh_target_t`

    Layout, in local units (× `scale`):
      spacing / ex_dy / ex_xy   even pitch with a per-cell vertical offset, or explicit
                       `(x, y)` centres overriding both. The row is deliberately not on
                       one baseline — `ex_dy` is what lets the axons run at readable
                       angles.
      inh_dx / inh_dy  where the inhibitory cell sits. `inh_dy` is the *gap* above the
                       drawn row, not a centre, so negative values are normal on a
                       staggered row.

    Axon shape (the excitatory knobs; `inh_*` are the same for the inhibitory cell):
      axon_drop        straight descent out of the soma before the turn; must clear the
                       soma's bottom edge. 0 disables it.
      axon_rad         bow off the straight line, as a fraction of length, positive is
                       upward. `axon_rads` takes `{(pre, post): rad}` — which is what
                       separates a reciprocal pair into two visible arcs.
      axon_smooth      width of the turn out of the descent, as a fraction of the run.
                       0 is a bare corner. Also sets how branches leave a fork.
      axon_fork        where a branching axon splits, as a fraction of the way to its
                       targets' centroid. `axon_forks` takes `{pre: t}`; 0 or None
                       turns branching off.
      axon_spread      how far a branch turns off the stem at the fork: 0 leaves
                       tangent to it, 1 pointing straight at the target.
      axon_zorder      below the cells' own zorder of 3, so only the free run between
                       cells is inked.

    Contacts and their ink:
      inh_target_kind / inh_target_t / inh_spine_branch / inh_spine_rank
                       what each connector contacts and where. Each may be written as
                       one value, `{cell number: value}`, or a sequence alongside
                       `inh_targets` (`_per_target`).
      syn_gap / inh_syn_gap    clearance of a dot off the cell, in local units
      inh_gap          clearance between bar and dot, in points
      inh_syn_size / inh_syn_colors   dot area in points² (None = `syn_size`, 0 = none)
                       and `{kind: colour}` for the three compartments
      inh_bar_width / inh_bar_length / inh_bar_scale
                       `widthB` / `lengthB` of the `-[` arrowstyle × `mutation_scale`

    `axon_color` defaults to the excitatory cells' `edge_color` and `inh_line_color` to
    the inhibitory cell's, so wiring is the colour of whatever sent it.

    Returns (fig, ax, dict(ex=[geom…], inh=geom or None, axons=[patch…],
    inh_arrows=[patch…], synapses=(n,2) array of the drawn dot positions,
    inh_synapses=(m,2) array of the inhibitory connectors' own dots)).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, PathPatch

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    base = dict(ex_kw or {})
    if axon_color is None:
        axon_color = base.get('edge_color', _EX_COLOR)

    n = int(n_ex)
    if ex_xy is None:
        dys = list(ex_dy or ())
        dys = dys[:n] + [0.0] * max(0, n - len(dys))
        x0 = -0.5 * (n - 1) * float(spacing)
        ex_xy = [(x0 + i * float(spacing), dys[i]) for i in range(n)]
    centres = [np.asarray(c, dtype=float) * float(scale) for c in ex_xy]

    # ---- part 1a: the cells ------------------------------------------------
    geoms, boxes = [], []
    for c in centres:
        _, _, g = plot_ex_neuron(ax=ax, center=c, scale=scale, pad=pad,
                                 return_geom=True, **base)
        # Each call fits the axes to its own cell; keep what every one asked
        # for and take the union once the whole row is down.
        boxes.append((ax.get_xlim(), ax.get_ylim()))
        geoms.append(g)

    bx0, bx1 = min(b[0][0] for b in boxes), max(b[0][1] for b in boxes)
    by0, by1 = min(b[1][0] for b in boxes), max(b[1][1] for b in boxes)

    # ---- part 2a: the inhibitory cell, placed clear of the row's top -------
    inh_geom = None
    if with_inh:
        ikw = dict(inh_kw or {})
        if inh_line_color is None:
            inh_line_color = ikw.get('edge_color', _INH_COLOR)
        probe = _inh_geometry(**{**dict(ikw.get('geom_kw') or {}),
                                 **{k: ikw[k] for k in _INH_GEOM_ARGS
                                    if k in ikw}})
        low = min([np.asarray(probe['soma'], dtype=float)[:, 1].min()]
                  + [np.asarray(b['centre'], dtype=float)[:, 1].min()
                     for b in probe['branches']])
        s_i = float(scale) * float(inh_scale)
        _, _, inh_geom = plot_inh_neuron(
            ax=ax, scale=s_i, pad=pad, return_geom=True,
            center=(float(np.mean([c[0] for c in centres])) + inh_dx * scale,
                    by1 + float(inh_dy) * scale - low * s_i),
            **ikw)
        (ix0, ix1), (iy0, iy1) = ax.get_xlim(), ax.get_ylim()
        bx0, bx1 = min(bx0, ix0), max(bx1, ix1)
        by0, by1 = min(by0, iy0), max(by1, iy1)

    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    ax.set_aspect('equal')
    ax.axis('off')

    # ---- part 1b: the axons, and a synapse dot at the end of each ----------
    # The dot stand-off is quoted as visible clearance, so it needs the final
    # data→display scale — hence everything below the limits above.
    gap = float(syn_gap) * float(scale) + _dot_radius_data(ax, syn_size)
    conns = [(int(a) - 1, int(b) - 1) for a, b in connections]
    targets = _axon_targets(
        geoms, conns,
        {(int(a) - 1, int(b) - 1): v
         for (a, b), v in dict(spine_rank or {}).items()},
        {(int(a) - 1, int(b) - 1): v
         for (a, b), v in dict(spine_branch or {}).items()})
    rads = {(int(a) - 1, int(b) - 1): float(v)
            for (a, b), v in dict(axon_rads or {}).items()}

    forks = {int(k) - 1: float(v) for k, v in dict(axon_forks or {}).items()}
    drop_u = float(axon_drop) * float(scale)

    # Where each axon ends, and which of them leave the same cell — a cell with
    # more than one output is drawn as one branching arbor, not as one line per
    # contact (`_axon_tree`).
    ends, by_src = {}, {}
    for src, dst in conns:
        sp = targets[(src, dst)]
        ends[(src, dst)] = (np.asarray(sp['head'], dtype=float)
                            + _unit(sp['dir']) * (geoms[dst]['head_r'] + gap))
        by_src.setdefault(src, []).append(dst)

    strokes = []
    for src, dsts in by_src.items():
        a = _soma_centre(geoms[src])
        rs = [rads.get((src, d), _arc_rad(a, ends[(src, d)], axon_rad))
              for d in dsts]
        fork = forks.get(src, axon_fork)
        if len(dsts) > 1 and fork:
            stem, branches = _axon_tree(a, [ends[(src, d)] for d in dsts],
                                        drop_u, rs, axon_smooth, fork,
                                        axon_spread)
            strokes += [stem] + branches
        else:
            strokes += [_axon_path(a, ends[(src, d)], drop_u, r, axon_smooth)
                        for d, r in zip(dsts, rs)]

    axons = []
    for path in strokes:
        p = PathPatch(path, fill=False, color=axon_color, linewidth=axon_lw,
                      alpha=axon_alpha, capstyle='round', joinstyle='round',
                      zorder=axon_zorder)
        ax.add_patch(p)
        axons.append(p)

    syn = np.array([ends[c] for c in conns], dtype=float).reshape(-1, 2)
    if len(syn):
        ax.scatter(syn[:, 0], syn[:, 1], s=syn_size, color=syn_color,
                   alpha=syn_alpha, edgecolors=syn_edge_color,
                   linewidths=syn_edge_lw, zorder=syn_zorder)

    # ---- part 2b: one flat-barred connector per target ---------------------
    inh_arrows, inh_syn = [], []
    if inh_geom is not None:
        tg = list(range(n)) if inh_targets is None else [int(t) - 1
                                                        for t in inh_targets]
        # The inhibitory synapse is the same dot as the excitatory ones unless
        # told otherwise, and its colour is its target compartment — spine
        # purple, soma and shaft green — exactly as the key reads it.
        i_syn_size = float(syn_size if inh_syn_size is None else inh_syn_size)
        i_syn_col = {'spine': syn_color, 'soma': _SHAFT_COLOR,
                     'trunk': _SHAFT_COLOR, **dict(inh_syn_colors or {})}
        r_syn = _dot_radius_data(ax, i_syn_size) if i_syn_size else 0.0
        irads = {int(k) - 1: float(v) for k, v in dict(inh_rads or {}).items()}
        # Per-edge source offsets — `inh_fork` off only, since with it on every
        # branch leaves the one stem. Each connector defaults to leaving from
        # the soma centre, but the arc's control point can pull the visible exit
        # over to a wall on cells whose target sits under a different quadrant
        # of the soma than its neighbours. Nudging just that one edge's source
        # in inh-cell-local units (× scale × inh_scale, i.e. the same units the
        # soma and arms are drawn in) puts its exit back next to its neighbours'
        # without disturbing anyone else's arc. Keep the nudged source *inside*
        # the soma: the connectors are drawn under the cell, so a source moved
        # out into the open is where the visible line starts, and it starts off
        # the wall with a gap instead of emerging from it.
        s_off = float(scale) * float(inh_scale)
        isrc = {int(k) - 1: (float(dx) * s_off, float(dy) * s_off)
                for k, (dx, dy) in dict(inh_src_offsets or {}).items()}
        src0 = _soma_centre(inh_geom)
        style = '-[,widthB=%g,lengthB=%g' % (inh_bar_width, inh_bar_length)
        # The bar is drawn *across* the path, so half of it sticks out sideways
        # from where the path stops, and a connector running alongside the
        # dendrite lands that overhang straight onto it. Folding the overhang
        # into the sideways clearance keeps the whole bar outside the envelope
        # at any arrival angle, and keeps it there when the bar is resized.
        clear = _pt_to_data(ax, float(inh_gap)
                            + 0.5 * float(inh_bar_width) * float(inh_bar_scale))
        # Every contact first, the lines that reach them after: with `inh_fork`
        # on they are one branching axon, and its stem cannot be aimed until
        # every branch's far end is known. Nothing here depends on how the line
        # gets there, so the endpoints are the same either way.
        contacts = []
        for k, j in enumerate(tg):
            dx, dy = isrc.get(j, (0.0, 0.0))
            src = src0 + np.array([dx, dy], dtype=float)
            kind = str(_per_target(inh_target_kind, j, k, 'trunk'))
            t = float(_per_target(inh_target_t, j, k, 0.55))
            if kind == 'spine':
                surf, nrm = _spine_contact(
                    geoms[j], src,
                    _per_target(inh_spine_branch, j, k, 'apical'),
                    int(_per_target(inh_spine_rank, j, k, 0)))
            elif kind == 'soma':
                surf, nrm = _soma_contact(geoms[j], t, src)
            else:
                surf, nrm = _trunk_contact(geoms[j], t, src)
            # dot first, then the bar outside it: both stack along the contact's
            # own outward normal, so the connector stops clear of the dot at
            # whatever angle it comes in at.
            dot = surf + nrm * (float(inh_syn_gap) * float(scale) + r_syn)
            b = dot + nrm * (r_syn + clear)
            if i_syn_size:
                inh_syn.append((dot, i_syn_col.get(kind, _SHAFT_COLOR)))
            contacts.append((j, k, src, b))

        common = dict(arrowstyle=style, mutation_scale=inh_bar_scale,
                      color=inh_line_color, linewidth=inh_lw,
                      alpha=inh_alpha, zorder=inh_zorder)
        crads = [irads.get(j, _arc_rad(src, b, inh_rad))
                 for j, _, src, b in contacts]

        if inh_fork and len(contacts) > 1:
            # One axon out of the bottom of the soma, forking into a branch per
            # target — `_axon_tree`, the same object the row's own axons are
            # drawn as, so the cell that inhibits the ensemble wires it in the
            # same hand as the cells that excite each other. The descent is
            # shared now, so `inh_drop` is one number: a per-connector one is
            # averaged into it rather than dropped. Per-edge `inh_src_offsets`
            # have nothing left to part and do not apply — there is one source.
            drop = float(np.mean([_per_target(inh_drop, j, k, 0.0)
                                  for j, k, _, _ in contacts])) * s_off
            stem, branches = _axon_tree(src0, [c[3] for c in contacts], drop,
                                        crads, inh_smooth, float(inh_fork),
                                        inh_spread)
            # the stem ends in the fork, not on a cell, so it carries no bar —
            # only what actually arrives somewhere is an arrow
            p = PathPatch(stem, fill=False, color=inh_line_color,
                          linewidth=inh_lw, alpha=inh_alpha, capstyle='round',
                          joinstyle='round', zorder=inh_zorder)
            ax.add_patch(p)
            inh_arrows.append(p)
            for branch in branches:
                p = FancyArrowPatch(path=branch, **common)
                ax.add_patch(p)
                inh_arrows.append(p)
        else:
            for (j, k, src, b), rad in zip(contacts, crads):
                drop = float(_per_target(inh_drop, j, k, 0.0)) * s_off
                # With a descent the connector is no longer an `arc3` between
                # two points but a stroke of its own, so it is handed over as a
                # path — `FancyArrowPatch` still puts its bar on the end, square
                # to whatever direction the path arrives in.
                p = FancyArrowPatch(
                    path=_axon_path(src, b, drop, rad, inh_smooth),
                    **common) if drop else FancyArrowPatch(
                    posA=src, posB=b, shrinkA=0, shrinkB=0,
                    connectionstyle='arc3,rad=%g' % rad, **common)
                ax.add_patch(p)
                inh_arrows.append(p)

        # One scatter per colour, so a two-colour set of contacts is still two
        # calls rather than one per connector.
        for col in dict.fromkeys(c for _, c in inh_syn):
            xy = np.array([d for d, c in inh_syn if c == col], dtype=float)
            ax.scatter(xy[:, 0], xy[:, 1], s=i_syn_size, color=col,
                       alpha=inh_syn_alpha, edgecolors=inh_syn_edge_color,
                       linewidths=inh_syn_edge_lw, zorder=inh_syn_zorder)

    inh_syn_xy = np.array([d for d, _ in inh_syn], dtype=float).reshape(-1, 2)
    dots = np.vstack([a for a in (syn, inh_syn_xy) if len(a)]) \
        if len(syn) or len(inh_syn_xy) else np.empty((0, 2))
    if len(dots):
        lo, hi = dots.min(axis=0) - pad * scale, dots.max(axis=0) + pad * scale
        ax.set_xlim(min(bx0, lo[0]), max(bx1, hi[0]))
        ax.set_ylim(min(by0, lo[1]), max(by1, hi[1]))

    return fig, ax, {'ex': geoms, 'inh': inh_geom, 'axons': axons,
                     'inh_arrows': inh_arrows, 'synapses': syn,
                     'inh_synapses': inh_syn_xy}


def plot_inh_target_panel(
        ax, n_spiny=3, spiny_pairs=None, spine_rank_by_target=(0, 1, 2),
        soma_color=None, dendrite_color=None, inh_color=_INH_COLOR,
        spiny_color=_SPINY_COLOR, shaft_color=_SHAFT_COLOR,
        title='', input_title='', label_fontsize=13, caption_y=0.02,
        title_dx=0, input_title_dx=0,
        inh_size=300, neuron_scale=0.60,
        inh_xy=None, ex_xy=None,
        spiny_lw=2.2, shaft_lw=1.0,
        spiny_alpha=1.0, shaft_alpha=0.5,
        spiny_head=14, shaft_head=9,
        n_soma_arrows=2, spine_gap=0.05, shaft_gap=0.06, soma_gap=0.03,
        dendrite_lw=2.4, spine_lw=2.6, geom_kw=None,
        xlim=(-0.75, 3.05), ylim=(-0.55, 4.95)):
    """3 inhibitory cells → 3 cartoon pyramidal cells, all-to-all (9 arrows).

    Arrow color encodes the post-synaptic *target compartment*, not the cell:
    `n_spiny` of the 9 land on an actual spine head (spine color), the rest on
    the apical shaft or the soma (shaft color). That is the whole point of the
    schematic, so the two arrow classes are styled independently — spine arrows
    thick and opaque, shaft arrows thin and semi-transparent (`shaft_alpha`), or
    9 same-weight arrows would read as an unresolvable thicket.

    Which pairs are spiny defaults to the diagonal, `[(i, i) for i in
    range(n_spiny)]` — pass `spiny_pairs` as explicit (inh_idx, ex_idx) tuples to
    override. Each spine arrow is routed to a *left-facing* spine so it never has
    to cross the dendrite to reach its head; `spine_rank_by_target[j]` picks
    which one for target `j`, indexing the left spines top-down, so the three
    contacts sit at different heights instead of all on the topmost spine.

    Shaft arrows land on the bare proximal trunk (`shaft_ts`, below the first
    spine) or, for the lowest `n_soma_arrows` per target, on the soma edge —
    never beside a spine head, where they would read as spine contacts.

    `title` is centred under the pyramidal column and `input_title` under the
    inhibitory one, both in data-x / axes-y, so the captions track their columns
    rather than the axes centre. `xlim` has to leave room for `input_title`, which
    is wider than the marker it labels.

    `spine_gap` / `shaft_gap` / `soma_gap` are the stand-off between arrow tip
    and target, in data units (the axes is aspect-locked, so one number works in
    both directions); `shrinkA` is instead in points, derived from the source
    marker's area the same way `plot_network_panel` does it.

    Returns dict(inh=inh coords, ex=[per-neuron geometry], arrows=[FancyArrowPatch]).
    """
    from matplotlib.patches import FancyArrowPatch

    if soma_color is None:
        soma_color = _EX_COLOR
    if dendrite_color is None:
        dendrite_color = _shade(soma_color, 0.55)

    # Vertical pitch between the pyramidal cells has to clear one whole cell —
    # soma bottom to apical tip is (soma_bot + soma_top + trunk_len) * neuron_scale
    # ≈ 1.50 at the default scale — or the trunk of one spears the soma of the one
    # above it. 1.65 leaves a visible gap.
    inh = np.array(inh_xy if inh_xy is not None
                   else [[0.0, 3.70], [0.0, 2.05], [0.0, 0.40]], dtype=float)
    ex_c = np.array(ex_xy if ex_xy is not None
                    else [[2.30, 3.45], [1.95, 1.80], [2.25, 0.15]], dtype=float)

    # Limits before the arrows: aspect + limits fix the data→display mapping the
    # stand-off geometry is computed in.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    neurons = [_draw_pyramidal(ax, c, neuron_scale, soma_color, dendrite_color,
                               dendrite_lw=dendrite_lw, spine_lw=spine_lw,
                               geom_kw=geom_kw)
               for c in ex_c]
    ax.scatter(inh[:, 0], inh[:, 1], marker='o', s=inh_size, color=inh_color,
               edgecolor='gray', linewidth=0.5, zorder=3)

    if spiny_pairs is None:
        spiny_pairs = [(i, i) for i in range(int(n_spiny))]
    spiny_pairs = [tuple(p) for p in spiny_pairs]
    spiny_set   = set(spiny_pairs)

    r_inh_pts = 0.5 * np.sqrt(inh_size)     # scatter marker radius, in points
    arrows    = []

    def _arrow(src, dst, color, lw, alpha, head, gap):
        u = np.asarray(dst) - np.asarray(src)
        n = np.linalg.norm(u)
        if n < 1e-9:
            return None
        u = u / n
        patch = FancyArrowPatch(
            posA=src, posB=np.asarray(dst) - u * gap,
            arrowstyle='-|>', mutation_scale=head,
            color=color, linewidth=lw, alpha=alpha,
            shrinkA=r_inh_pts + 1.5, shrinkB=0, zorder=1)
        ax.add_patch(patch)
        arrows.append(patch)
        return patch

    for j, geo in enumerate(neurons):
        # ── spine targets: left-facing heads, top-most first ──
        left_heads = sorted([s for s in geo['spines'] if s['side'] == -1],
                            key=lambda s: -s['t'])
        rank0 = spine_rank_by_target[j % len(spine_rank_by_target)]
        for k, i in enumerate(sorted(i for i, jj in spiny_pairs if jj == j)):
            head = left_heads[(rank0 + k) % len(left_heads)]
            _arrow(inh[i], head['head'], spiny_color, spiny_lw, spiny_alpha,
                   spiny_head, geo['head_r'] + spine_gap)

        # ── shaft / soma targets for everything else ──
        greens = sorted((i for i in range(len(inh)) if (i, j) not in spiny_set),
                        key=lambda i: -inh[i][1])          # top source first
        to_soma  = greens[len(greens) - min(n_soma_arrows, len(greens)):] \
                   if n_soma_arrows else []
        to_shaft = [i for i in greens if i not in to_soma]
        # Trunk anchors are the *lowest* stretch of the trunk (below every
        # spine); higher ones would put an arrowhead beside a spine head.
        anchors  = sorted(geo['shaft'], key=lambda s: -s['t'])   # highest first
        for k, i in enumerate(to_shaft):
            a = anchors[k % len(anchors)]
            _arrow(inh[i], a['xy'], shaft_color, shaft_lw, shaft_alpha,
                   shaft_head, shaft_gap)
        soma_anchors = geo['soma_side'][-1]                       # apex-side first
        for k, i in enumerate(to_soma):
            a = soma_anchors[k % len(soma_anchors)]
            _arrow(inh[i], a['xy'], shaft_color, shaft_lw, shaft_alpha,
                   shaft_head, soma_gap)

    ax.axis('off')
    # Column labels: x in *data* coords so each sits under the column it names
    # (the pyramidal cells are far off-centre), y in axes fraction so both panels
    # put their captions on the same baseline.
    # caption_y is that axes fraction — lower it to push both captions away from
    # the somas when the cells are drawn low in the box.
    tr = ax.get_xaxis_transform()
    # title_dx / input_title_dx: nudge the two column labels apart in POINTS
    # (1 pt = 1/72 inch). Points is DPI-independent — 'dots' units scale with
    # fig.dpi and diverge between AGG (inline) and PDF at high dpi.
    from matplotlib.transforms import offset_copy
    if title:
        tr_title = offset_copy(tr, fig=ax.figure, x=title_dx, y=0, units='points')
        ax.text(ex_c[:, 0].mean(), caption_y, title, transform=tr_title, ha='center',
                va='bottom', fontsize=label_fontsize, color='black')
    if input_title:
        tr_in = offset_copy(tr, fig=ax.figure, x=input_title_dx, y=0, units='points')
        ax.text(inh[:, 0].mean(), caption_y, input_title, transform=tr_in, ha='center',
                va='bottom', fontsize=label_fontsize, color='black')

    return {'inh': inh, 'ex': neurons, 'arrows': arrows}


# ---------------------------------------------------------------------------
# Panel D: two axons, and what each of them lands on
# ---------------------------------------------------------------------------
#
# Every other panel here is about the *post*-synaptic side — a dendrite and the
# inputs arriving on it. This one turns the cell around: the subject is the
# axon leaving one soma, and each of its output synapses is colored by what it
# lands on at the far end (a spine, or a shaft). The dendrite is still drawn,
# rotated onto its side and faded almost out, purely so the axon has a cell to
# come out of.


def _axon_geometry(start, direction=(1.0, 0.0), length=6.0, n_collaterals=10,
                   first_t=0.10, last_t=0.97, first_side=-1,
                   collateral_len=0.10, collateral_angle_deg=58,
                   len_jitter=0.35, angle_jitter=12.0, t_jitter=0.35,
                   bend=0.015, wave_amp=0.022, wave_n=2.6, wave_phase=0.20,
                   collateral_bend=0.10, collateral_wave_amp=0.06,
                   collateral_wave_n=0.8, collateral_wave_phase=0.35,
                   seed=0, n_pts=220, n_col_pts=28):
    """One projecting axon in world units: a long waving trunk leaving `start`
    along `direction`, with `n_collaterals` short branches off it.

    Built from the same `_branch` curve as every dendrite in this module, so the
    axon reads as drawn by the same hand — the difference is only in the
    proportions. Its `bend`, `wave_amp` and `collateral_len` are quoted as
    **fractions of `length`** rather than in absolute units (as `_branch` wants
    them), because the point of the panel is one long axon beside one short one:
    an absolute waver that reads as a hand-drawn line on a length of 8 reads as
    a coil on a length of 3.

    `seed` drives three jitters — where along the trunk a collateral leaves
    (`t_jitter`, as a fraction of the even spacing), how far it reaches
    (`len_jitter`) and at what angle (`angle_jitter`, in degrees) — so the
    branches do not come out as a comb. They still strictly alternate sides.

    Returns {'trunk': (n_pts,2), 'at': t → (xy, tangent), 'tip', 'tip_dir',
    'collaterals': [{'centre', 'tip', 'dir', 't', 'side'}]}.
    """
    L = float(length)
    rng = np.random.default_rng(int(seed))

    trunk = _branch(start, direction, L, 0, 0.0,
                    bend=float(bend) * L, wave_amp=float(wave_amp) * L,
                    wave_n=wave_n, wave_phase=wave_phase, n_pts=int(n_pts))

    n = max(int(n_collaterals), 0)
    ts = np.linspace(first_t, last_t, n) if n else np.zeros(0)
    if n > 1:
        step = (float(last_t) - float(first_t)) / (n - 1)
        ts = np.clip(ts + float(t_jitter) * step * (rng.random(n) - 0.5),
                     first_t, last_t)

    cols = []
    for k, tt in enumerate(ts):
        side = first_side * (-1) ** k
        base, tang = trunk['at'](float(tt))
        # +angle rotates a mostly-forward tangent to the left, hence -side —
        # the same convention the spines use, so a collateral leans toward the
        # tip rather than sticking straight out.
        ang = float(collateral_angle_deg) + float(angle_jitter) * (rng.random() - 0.5) * 2
        cl  = collateral_len * L * (1 + float(len_jitter) * (rng.random() - 0.5) * 2)
        b = _branch(base, _rot(tang, -side * ang), cl, 0, 0.0,
                    bend=float(collateral_bend) * cl * side,
                    wave_amp=float(collateral_wave_amp) * cl * side,
                    wave_n=collateral_wave_n, wave_phase=collateral_wave_phase,
                    n_pts=int(n_col_pts))
        c = b['centre']
        cols.append({'centre': c, 'tip': c[-1], 'dir': _unit(c[-1] - c[-2]),
                     't': float(tt), 'side': side})

    tr = trunk['centre']
    return {'trunk': tr, 'at': trunk['at'], 'collaterals': cols,
            'tip': tr[-1], 'tip_dir': _unit(tr[-1] - tr[-2])}


def _axon_tree_geometry(start, direction=(1.0, 0.0), length=6.0,
                        trunk_frac=0.24, stem_frac=1.0, depth=4, n_children=2,
                        first_children=None,
                        spread_deg=38.0, spread_decay=0.58, angle_jitter=10.0,
                        max_angle_deg=68.0,
                        child_frac=0.76, len_jitter=0.28,
                        bend=0.10, wave_amp=0.05, wave_n=1.1, wave_phase=0.30,
                        trunk_bend=0.0, trunk_wave_amp=0.02,
                        seed=0, n_pts=48):
    """One projecting axon grown as a **branching arbor** rather than a trunk
    with stubs: a short stem out of the soma that bifurcates `depth` times, so
    what reaches the far side of the panel is a tree of comparable branches.

    This is the alternative to `_axon_geometry`, and the difference is what the
    drawing is claiming. A trunk with collaterals says "one axon going
    somewhere, dropping the odd contact on the way"; a tree says "one axon
    arborising in a patch of cortex and contacting everything in it", which is
    what an intracortical projection actually looks like and what the panel is
    about. Both build every segment from the same `_branch` curve, so the two
    still read as drawn by the same hand.

    Each generation is a fixed fraction of its parent — `child_frac` in length
    and `spread_decay` in splitting angle — so the arbor gets finer and
    straighter outward instead of fanning ever wider. `length` is the scale for
    the whole thing (`trunk_frac` of it is the stem), not the reach: the reach
    is what the geometric series comes to, roughly
    `trunk_frac · length · (1 - child_frac**(depth+1)) / (1 - child_frac)`.

      spread_deg / spread_decay   half-angle of the first bifurcation, and what
                  each generation multiplies it by. The first split is the wide
                  one that makes the arbor a fan; the later ones only need to
                  keep sibling branches apart.
      stem_frac   shortens (or lengthens) the *drawn* stem alone, as a multiple
                  of `trunk_frac · length`. The children still scale off the
                  full `trunk_frac · length`, so this pulls the arbor in toward
                  the soma without shrinking it — which `trunk_frac` on its own
                  cannot do, being the scale of the whole tree.
      n_children / first_children   how many branches come off each node, and
                  how many off the stem's end if that one differs. Splitting
                  the stem three ways and everything after it two fills the
                  middle of the fan, which a pure binary tree leaves empty —
                  its two halves lean away from the axis, not along it.
      trunk_bend / trunk_wave_amp   the stem's own curvature, kept separate and
                  near zero: it is the one segment whose direction sets where
                  the whole arbor sits, so a waver on it swings everything.
      max_angle_deg   hard limit on how far any branch may point away from
                  `direction`. Angles compound down the generations, so without
                  it one unlucky run of same-side jitters sends a branch back
                  toward the soma and the fan stops reading as a projection.
      len_jitter / angle_jitter   per-branch draws off those two, so the arbor
                  is not a symmetric binary diagram; `seed` fixes them.
      bend / wave_amp   as in `_axon_geometry`, quoted as fractions of each
                  *segment's own* length, and signed by which side of its parent
                  the branch left on, so siblings curve apart rather than
                  parallel.

    Returns {'segments': [{'centre', 'gen', 'dist0', 'len', 'tip', 'dir',
    'terminal'}], 'tips': [terminal segments], 'length': length, 'trunk',
    'tip', 'tip_dir'} — `dist0` being the path distance from `start` to that
    segment's base, which is what the synapse colouring is graded on.
    """
    L = float(length)
    rng = np.random.default_rng(int(seed))
    base_dir = _unit(direction)
    segs = []

    def _grow(origin, dirn, seg_len, gen, dist0, spread, side, scale_len=None):
        # What the children are sized off, which is the segment's own length
        # everywhere but the stem — there `stem_frac` has already shortened
        # what gets drawn, and the arbor is not meant to shrink with it.
        sl = float(seg_len if scale_len is None else scale_len)
        bd = float(trunk_bend if gen == 0 else bend)
        wa = float(trunk_wave_amp if gen == 0 else wave_amp)
        b = _branch(origin, dirn, seg_len, 0, 0.0,
                    bend=bd * seg_len * side, wave_amp=wa * seg_len * side,
                    wave_n=wave_n, wave_phase=float(wave_phase) + 0.13 * gen,
                    n_pts=int(n_pts))
        c = b['centre']
        tip_dir = _unit(c[-1] - c[-2])
        segs.append({'centre': c, 'gen': gen, 'dist0': float(dist0),
                     'len': float(seg_len), 'tip': c[-1], 'dir': tip_dir,
                     'terminal': gen >= int(depth)})
        if gen >= int(depth):
            return

        m = max(int(n_children if gen or first_children is None
                    else first_children), 1)
        for k in range(m):
            off = (2.0 * k / (m - 1) - 1.0) if m > 1 else 0.0
            ang = off * spread + float(angle_jitter) * (rng.random() - 0.5) * 2
            d_new = _rot(tip_dir, ang)
            # Angles compound, so clamp each child against the *original*
            # direction rather than its parent's.
            a_abs = np.degrees(np.arctan2(
                base_dir[0] * d_new[1] - base_dir[1] * d_new[0],
                float(np.dot(base_dir, d_new))))
            if abs(a_abs) > float(max_angle_deg):
                d_new = _rot(base_dir, np.sign(a_abs) * float(max_angle_deg))
            cl = sl * float(child_frac) * (
                1 + float(len_jitter) * (rng.random() - 0.5) * 2)
            _grow(c[-1], d_new, cl, gen + 1, dist0 + seg_len,
                  spread * float(spread_decay),
                  1.0 if off >= 0 else -1.0)

    stem = float(trunk_frac) * L
    _grow(np.asarray(start, dtype=float), base_dir, float(stem_frac) * stem,
          0, 0.0, float(spread_deg), 1.0, scale_len=stem)

    tr = segs[0]['centre']
    return {'segments': segs, 'tips': [s for s in segs if s['terminal']],
            'length': L, 'trunk': tr, 'tip': tr[-1], 'tip_dir': segs[0]['dir']}


def _arc(c):
    """Cumulative arc length along a polyline, same length as it."""
    return np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(np.asarray(c, dtype=float), axis=0), axis=1))])


def _at_arc(c, d, a):
    """(point, unit tangent) at arc length `a` along polyline `c` (arcs `d`)."""
    i = int(np.clip(np.searchsorted(d, a), 1, len(d) - 1))
    f = (a - d[i - 1]) / max(d[i] - d[i - 1], 1e-12)
    return c[i - 1] + f * (c[i] - c[i - 1]), _unit(c[i] - c[i - 1])


def _tree_syn_sites(geom, n_syn, first_dist=1.15):
    """`n_syn` synapse sites spread over a whole axonal arbor.

    Evenly spaced *in arc length across the arbor as a whole*, not per branch:
    the walk carries its leftover from one segment into the next, so a stubby
    terminal branch gets its fair share of one dot rather than one each like
    every other branch. That is what makes the density read as uniform — per
    branch spacing piles dots up wherever the tree is finest, which is exactly
    where it is already busiest with lines.

    `first_dist` is the path distance from the soma at which sites start, as a
    **multiple of the trunk length**: the proximal axon is the trunk before
    any real arborisation and carrying boutons there reads as the cell
    synapsing on its own neighbourhood, so 1.0 lands the first site right at
    the first bifurcation, > 1 pushes it into the fan. Fraction of the trunk
    (not of `geom['length']`) so shortening the trunk does not push every dot
    off the tips.

    Returns (sites (n,2), dirs (n,2), dists (n,) path distance from the soma).
    """
    n = max(int(n_syn), 0)
    fd = float(first_dist) * float(geom['segments'][0]['len'])
    entries = [(s, s['centre'], _arc(s['centre'])) for s in geom['segments']]
    # How much of each segment is beyond `first_dist`, and the total of it.
    starts = [min(max(0.0, fd - s['dist0']), d[-1]) for s, _, d in entries]
    total = float(sum(d[-1] - a0 for (_, _, d), a0 in zip(entries, starts)))
    if not n or total <= 0:
        return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))

    ds = total / n
    nxt, acc = 0.5 * ds, 0.0
    sites, dirs, dists = [], [], []
    for (s, c, d), a0 in zip(entries, starts):
        span = d[-1] - a0
        while nxt <= acc + span + 1e-12 and len(sites) < n:
            xy, tang = _at_arc(c, d, a0 + (nxt - acc))
            sites.append(xy)
            dirs.append(tang)
            dists.append(s['dist0'] + a0 + (nxt - acc))
            nxt += ds
        acc += span

    return (np.asarray(sites, dtype=float).reshape(-1, 2),
            np.asarray(dirs, dtype=float).reshape(-1, 2),
            np.asarray(dists, dtype=float))


def _spiny_mask(ts, frac, bias=0.0, seed=0):
    """Which of the synapses at `ts` along an axon land on a spine.

    `frac` of them do, rounded to a whole synapse. Which ones is a draw, so that
    the colors read as a mixture rather than as a pattern — `bias` then tilts
    that draw toward the distal end: 0 is a pure shuffle, and each unit of bias
    is worth as much as the whole random range, so by ~2 the distal ones win
    almost every tie while a couple of proximal ones still slip through. That
    "sorted, but shuffled" look is what the reference sketch has.
    """
    t = np.asarray(ts, dtype=float)
    n = len(t)
    k = int(round(float(frac) * n))
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    span = t.max() - t.min()
    u = (t - t.min()) / span if span else np.zeros(n)
    score = np.random.default_rng(int(seed)).random(n) + float(bias) * u
    m = np.zeros(n, dtype=bool)
    m[np.argsort(score)[-k:]] = True
    return m


def _elbow(start, direction, run, turn_deg, radius, n_pts=32):
    """The kink between a soma and the axon proper: a straight `run` out of the
    cell along `direction`, then a fillet of `radius` turning through
    `turn_deg`, and the axon starts where that arc leaves off.

    A real axon does not set off toward its targets the moment it leaves the
    soma — it drops clear of the cell's own dendrites first and turns once it is
    out. Rounding that turn rather than mitring it is what keeps it reading as
    an axon and not as a circuit diagram's right angle: `radius` is the whole
    difference between the two, and at 0 the corner is a hard one.

    `turn_deg` is signed counter-clockwise, so with the cell drawn upright and
    `direction` pointing down out of it, +90 sends the axon off to the right.

    Returns (polyline (n,2), the xy the axon starts at, the unit direction it
    starts in).
    """
    d0 = _unit(direction)
    s = np.asarray(start, dtype=float)
    p = s + d0 * float(run)
    a, r = float(turn_deg), float(radius)
    if r <= 0 or a == 0:
        return np.vstack([s, p]), p, _rot(d0, a)

    # Centre of the fillet, `r` off to whichever side the axon turns toward.
    # The arc starts where the radius points back at the straight run, so the
    # two meet tangentially and the corner has no visible join.
    n = _rot(d0, np.sign(a) * 90.0)
    c = p + r * n
    t = (np.arctan2(-n[1], -n[0])
         + np.deg2rad(a) * np.linspace(0.0, 1.0, int(n_pts)))
    arc = c + r * np.stack([np.cos(t), np.sin(t)], axis=1)
    return np.vstack([[s], arc]), arc[-1], _rot(d0, a)


def plot_ex_axon(ax, start, direction=(1.0, 0.0), length=6.0, scale=1.0,
                 elbow_len=0.0, elbow_turn_deg=90.0, elbow_r=0.0, elbow_pts=32,
                 shape='collaterals', n_syn=12, tip_syn=True, syn_first_dist=1.15,
                 spiny_frac=0.80, spiny_bias=0.0, seed=0,
                 color=_EX_COLOR, lw=1.2, alpha=1.0, collateral_lw=0.8,
                 lw_decay=0.92, zorder=1.5,
                 spiny_color=_SPINY_COLOR, aspiny_color=_SHAFT_COLOR,
                 dot_size=26, dot_alpha=1.0, dot_edge_color='none',
                 dot_edge_lw=0.0, dot_gap=0.0, dot_zorder=6, geom_kw=None):
    """One cell's axon and its output synapses, drawn as dots along it.

    Two shapes, and `shape` picks which:

      'collaterals'  `_axon_geometry` — a long projecting trunk leaving `start`
                  along `direction` with short stubs off it, one synapse at the
                  tip of each and — with `tip_syn` — one more at the end of the
                  trunk, so `n_syn` is the count the caller cares about and the
                  branching follows from it.
      'tree'      `_axon_tree_geometry` — a bifurcating arbor, with the `n_syn`
                  synapses spread evenly along *every* branch of it rather than
                  sitting at tips (`_tree_syn_sites`, starting `syn_first_dist`
                  trunk-lengths out from the soma — 1.0 lands the first site
                  at the first bifurcation, > 1 into the fan). `tip_syn` does
                  not apply.

    Every dot is colored by **what it lands on at the far end**, which is the
    inverse of `plot_synapse_dots`: there the cell was the target and its own
    spines decided the color, here the targets are off-panel and the split is
    simply asserted — `spiny_frac` of them purple (a spine), the rest green (a
    shaft). `spiny_bias` tilts which ones toward the distal end (see
    `_spiny_mask`, graded on path distance from the soma in 'tree' mode and on
    position along the trunk in 'collaterals'); `seed` fixes both that draw and
    the branch jitters.

      length      the axon's scale in local units, × `scale`. Everything else
                  about the shape is a fraction of it, so one number sets how
                  far the cell projects and the drawing stays in proportion.
      elbow_len   how far the axon runs along `direction` before turning, in
                  world units × `scale` — **not** a fraction of `length`, since
                  what it has to clear is the cell it leaves, whose size has
                  nothing to do with how far the axon then projects. 0 (the
                  default) is no elbow at all: the axon sets off along
                  `direction` straight out of the soma.
      elbow_turn_deg / elbow_r   how far the elbow turns, counter-clockwise,
                  and the radius of the fillet it turns through (same units as
                  `elbow_len`). See `_elbow` — with the cell upright and
                  `direction` pointing down, +90 sends the axon off right.
      lw / collateral_lw / lw_decay   trunk linewidth in points, then the
                  collaterals' as a multiple of it ('collaterals'), or each
                  generation's as a multiple of its parent's ('tree') — either
                  way the branches are thinner children of the same axon.
      zorder      below the cells' own 3 by default, so the stretch inside the
                  soma is hidden by its opaque fill and the axon reads as
                  emerging from the cell body rather than starting beside it.
      dot_gap     push each dot off the branch it sits on, as a fraction of
                  `length`. 0 (default) centres it on the line — a bouton.
      geom_kw     the rest of whichever geometry `shape` selected.

    Returns dict(geom=…, sites=(n,2), spiny=(n,) bool, elbow=(k,2) or None,
    points=(m,2) every drawn vertex, for fitting the axes around it).
    """
    L = float(length) * float(scale)

    # The elbow out of the soma, if there is one — after it, `start` and
    # `direction` are where the axon proper begins, and everything below is
    # written as though the cell were sitting there pointing that way.
    elbow = None
    if float(elbow_len) > 0 or float(elbow_r) > 0:
        elbow, start, direction = _elbow(
            start, direction, float(elbow_len) * float(scale), elbow_turn_deg,
            float(elbow_r) * float(scale), elbow_pts)
        ax.plot(elbow[:, 0], elbow[:, 1], color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=zorder)

    if str(shape).startswith('tree'):
        g = _axon_tree_geometry(start, direction, L, seed=seed,
                                **dict(geom_kw or {}))
        for s in g['segments']:
            c = s['centre']
            ax.plot(c[:, 0], c[:, 1], color=color,
                    lw=lw * float(lw_decay) ** s['gen'], alpha=alpha,
                    solid_capstyle='round', zorder=zorder)

        sites, dirs, dists = _tree_syn_sites(g, n_syn, syn_first_dist)
        sites = sites + dirs * float(dot_gap) * L
        spiny = _spiny_mask(dists, spiny_frac, spiny_bias, seed)
        drawn = [s['centre'] for s in g['segments']]
    else:
        n_col = max(int(n_syn) - (1 if tip_syn else 0), 0)
        g = _axon_geometry(start, direction, L, n_col, seed=seed,
                           **dict(geom_kw or {}))

        ax.plot(g['trunk'][:, 0], g['trunk'][:, 1], color=color, lw=lw,
                alpha=alpha, solid_capstyle='round', zorder=zorder)
        for b in g['collaterals']:
            ax.plot(b['centre'][:, 0], b['centre'][:, 1], color=color,
                    lw=lw * float(collateral_lw), alpha=alpha,
                    solid_capstyle='round', zorder=zorder)

        ends = [(b['tip'], b['dir'], b['t']) for b in g['collaterals']]
        if tip_syn:
            ends.append((g['tip'], g['tip_dir'], 1.0))
        sites = np.array([xy + d * float(dot_gap) * L for xy, d, _ in ends],
                         dtype=float).reshape(-1, 2)
        spiny = _spiny_mask([t for _, _, t in ends], spiny_frac, spiny_bias, seed)
        drawn = [g['trunk']] + [b['centre'] for b in g['collaterals']]

    for m, c in ((spiny, spiny_color), (~spiny, aspiny_color)):
        if len(sites) and m.any():
            ax.scatter(sites[m, 0], sites[m, 1], s=dot_size, color=c,
                       alpha=dot_alpha, edgecolors=dot_edge_color,
                       linewidths=dot_edge_lw, zorder=dot_zorder)

    if elbow is not None:
        drawn = [elbow] + drawn
    points = np.vstack(drawn + ([sites] if len(sites) else []))
    return {'geom': g, 'sites': sites, 'spiny': spiny, 'elbow': elbow,
            'points': points}


# Panel D's default elbow: the drop out of the soma and the radius of the turn
# that follows it, both in the *cell's own* local units and so multiplied by
# `cell_scale` before use. What the drop has to do is clear the cell's basal
# dendrites, which is a fact about the cell and not about how far the axon then
# projects — the whole cell is about 2.6 local units tall, so together these
# come to a little over its full height and the hook reads at any `cell_scale`.
#
# The two are what set how high the cell rides in the panel, and that is the
# other thing they are tuned against: the axes are fitted around the arbor,
# which dwarfs the cell, so the soma sits `len + r` above the axon's horizontal
# run and nothing else moves it. On the assembled figures the key floats over
# panel d's top-left corner, and much more than this puts the cell's apical tip
# through it.
_AXON_ELBOW_LEN = 2.2
_AXON_ELBOW_R = 2.1


def plot_axon_panel(ax=None, cell_kw=None, cell_scale=0.55, cell_rotate_deg=0,
                    cell_xy=((0.0, 0.0), (0.0, -2.80)),
                    axon_kw=None, axon_kws=None, render_bottom=False,
                    figsize=(9.5, 4.4), dpi=150, pad=0.25):
    """Panel D: two excitatory cells one above the other, each sending an axon
    off to the right, with its output synapses dotted along it.

    The comparison is the whole panel: the top cell projects far and most of
    what it lands on is a spine, the bottom one projects a short way and lands
    on spines about half the time. Nothing about the two cells themselves
    differs — the statement is entirely in the axons, which is why they get all
    the ink and the cells get almost none.

    Each cell is a `plot_ex_neuron` drawn small, thin-walled and faded (whatever
    `cell_kw` says — normally the notebook's standard cell with a low `alpha`
    and a hairline `wall_lw`), and left **upright** (`cell_rotate_deg=0`),
    apical up. Its axon therefore leaves *downward*, the way a real one does,
    and gets across the page by dropping clear of the basal dendrites and
    turning a rounded 90° out to the right — `plot_ex_axon`'s elbow, defaulted
    on here (nothing else in the module needs one) at a size taken from
    `cell_scale`, and overridable through `axon_kw` like any other axon knob:

        elbow_len       how far it drops before turning, world units
        elbow_r         radius of the turn — 0 mitres it into a hard corner
        elbow_turn_deg  how far it turns, counter-clockwise; 0 drops it
                        straight down with no turn at all

    The axon's *starting* direction is the soma's own "down" taken through
    `cell_rotate_deg`, so laying a cell on its side (90 = apical left, axon
    leaving right) still turns its axon with it — the elbow then turns off
    whatever that gives.

      cell_xy     the two soma centres, in the same units as an axon's `length`
                  (**not** scaled by `cell_scale` — the cells shrink, the layout
                  does not).
      axon_kw     `plot_ex_axon` style shared by both axons; `axon_kws` is the
                  per-cell overrides on top of it, top cell first — normally
                  `length`, `n_syn`, `spiny_frac`, `spiny_bias` and `seed`.
      render_bottom  draw the second cell + axon at all. Off (the default)
                  drops it entirely and the axes are fitted to the top pair
                  alone — no reserved space for the missing half, no empty
                  band at the bottom of the panel.

    The axons are drawn after the cells and the limits fitted around everything
    at the end, so an axon may freely run behind a cell (it is drawn under them
    — see `plot_ex_axon`'s `zorder`).

    Returns (fig, ax, dict(cells=[geom…], axons=[dict…])).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # The cell is drawn apical-up with its axon leaving downward, so the axon's
    # direction is just that "down" taken through the same rotation.
    d = _rot((0.0, -1.0), float(cell_rotate_deg))

    # `render_bottom=False` drops the second cell + axon entirely; the axes
    # fit only what is drawn, so the panel shrinks to the top pair alone.
    xy = list(cell_xy) if render_bottom else list(cell_xy)[:1]

    geoms, boxes = [], []
    for c in xy:
        _, _, g = plot_ex_neuron(ax=ax, center=np.asarray(c, dtype=float),
                                 scale=cell_scale, rotate_deg=cell_rotate_deg,
                                 pad=pad, return_geom=True,
                                 **dict(cell_kw or {}))
        boxes.append((ax.get_xlim(), ax.get_ylim()))
        geoms.append(g)

    # The cells stand upright, so every axon leaves downward and needs the
    # elbow to get out across the page. Defaulted here rather than in
    # `plot_ex_axon` (where it is off) because this panel is the only caller
    # that wants one, and sized off `cell_scale` rather than off either axon's
    # `length`, so both drop the same amount and a smaller cell does not leave
    # a long bare stem hanging under it.
    base = dict(axon_kw or {})
    base.setdefault('elbow_len', _AXON_ELBOW_LEN * float(cell_scale))
    base.setdefault('elbow_r', _AXON_ELBOW_R * float(cell_scale))
    per = list(axon_kws or ())
    per += [{}] * (len(geoms) - len(per))

    axons = [plot_ex_axon(ax, _soma_centre(g), direction=d,
                          **{**base, **dict(extra or {})})
             for g, extra in zip(geoms, per[:len(geoms)])]

    pts = np.vstack([a['points'] for a in axons])
    x0 = min([b[0][0] for b in boxes] + [pts[:, 0].min() - pad])
    x1 = max([b[0][1] for b in boxes] + [pts[:, 0].max() + pad])
    y0 = min([b[1][0] for b in boxes] + [pts[:, 1].min() - pad])
    y1 = max([b[1][1] for b in boxes] + [pts[:, 1].max() + pad])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax, {'cells': geoms, 'axons': axons}


# ---------------------------------------------------------------------------
# The legend, and the assembled figure
# ---------------------------------------------------------------------------


def _legend_item(text, kind, colors, fontsize, family, text_color, dot_size,
                 dot_edge_color, dot_edge_lw, label_gap, dot_side='right'):
    """One legend entry: a label, and — if `kind` is given — its dot.

    The dot is a `DrawingArea` sized in *points*, `dot_size` being an area the
    way matplotlib's `scatter` reads it, so the key's dots come out the size the
    panels' dots are asked for.

    `dot_side` puts the dot after the label (`'right'`) or before it
    (`'left'`). Before is what a key stacked one entry per line wants: the dots
    then line up down the left edge, and the eye runs down them rather than
    down a ragged column of word-ends.
    """
    from matplotlib.offsetbox import DrawingArea, HPacker, TextArea
    from matplotlib.patches import Circle

    label = TextArea(text, textprops=dict(family=family, size=fontsize,
                                          color=text_color))
    if kind is None:
        return label

    r = float(np.sqrt(float(dot_size) / np.pi))
    dot = DrawingArea(2.0 * r, 2.0 * r, 0.0, 0.0)
    dot.add_artist(Circle((r, r), r, facecolor=colors.get(kind, kind),
                          edgecolor=dot_edge_color, linewidth=dot_edge_lw))
    children = [dot, label] if str(dot_side) == 'left' else [label, dot]
    return HPacker(pad=0.0, sep=float(label_gap), align='center',
                   children=children)


def plot_figure7_legend(
        ax=None,
        labels=(('Synapse onto',),
                (('spine', 'spiny'), ('shaft/soma', 'aspiny'))),
        fontsize=13, text_family='Arial', text_color='black',
        spiny_color=_SPINY_COLOR, aspiny_color=_SHAFT_COLOR,
        dot_size=110, dot_edge_color='none', dot_edge_lw=0.0,
        dot_side='right',
        label_gap=6.0, item_gap=14.0, row_gap=6.0, align='left',
        frame=True, frame_lw=0.8, frame_color='0.8', frame_facecolor='white',
        frame_alpha=1.0, frame_edge_alpha=None, frame_boxstyle=None,
        pad=6.0, loc='upper left', xy=(0.0, 1.0),
        figsize=(3.4, 0.9), dpi=150):
    """The figure's key: the synapse colors as labelled dots, in a box.

    `labels` is one tuple per row, and an entry in a row is either a bare
    string (a label on its own — the lead-in "Synapse onto") or
    `(label, kind)`, which puts a dot after the label. `kind` is `'spiny'` /
    `'aspiny'` for the two synapse colors, or any matplotlib color to use it
    directly. So the default reads

        Synapse onto
        spine ●   shaft/soma ●

    Everything is laid out by `offsetbox` packers — a row is an `HPacker` of
    entries, the rows a `VPacker`, and the whole thing an `AnchoredOffsetbox`
    against `ax`. That is what keeps a dot centred on the text beside it and
    the rows flush at their left edge whatever the wording, and it means
    **every gap here is in points**: the box hugs its contents at the size the
    font asks for, and neither the axes' data limits nor its aspect come into
    it. `ax` is only a place to hang the box — it is turned off, and its size
    sets where the key sits, not how big it is.

    A row per entry — `(('Synapse onto',), (('spine', 'spiny'),),
    (('shaft/soma', 'aspiny'),))` — with `dot_side='left'` is the other
    reading: a short stack with the dots down its left edge, which sits in a
    corner of a panel rather than across the top of one.

      label_gap   label → its own dot; `item_gap` entry → the next entry, and
                  `row_gap` the space between the rows.
      dot_side    `'right'` puts each dot after its label, `'left'` before it.
      pad         clearance between the contents and the frame.
      frame_alpha the fill's opacity — 1.0 a solid block, low a wash. A key
                  laid over a drawing wants to be read *and* to let the ink
                  under it show, which is a wash rather than a solid white box.
      frame_edge_alpha  the outline's, when it should differ from the fill's.
                  `None` gives it the fill's. The pair is what lets one grey do
                  both jobs: the outline near-solid so the key has an edge to
                  be read against, the fill faint so the drawing shows through.
                  They are set as color alphas rather than on the patch,
                  because a patch's own `alpha` would override both at once.
      frame_boxstyle  a `FancyBboxPatch` style for the frame, e.g.
                  `'round,pad=0.02,rounding_size=0.3'`; its `pad` is in font
                  units and stacks on `pad`, so keep it near zero and let `pad`
                  do the padding. `None` leaves the square default.
      loc / xy    which corner of the box lands where in `ax`'s axes
                  coordinates — `'upper left'` at `(0, 1)` puts the box's top
                  left corner in the axes' top left.

    Returns (fig, ax, box) — `box` being the `AnchoredOffsetbox`, whose
    `.get_window_extent(renderer)` is where the key actually ended up.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    colors = {'spiny': spiny_color, 'aspiny': aspiny_color}

    rows = []
    for row in labels:
        entries = [(e, None) if isinstance(e, str) else tuple(e) for e in row]
        rows.append(HPacker(pad=0.0, sep=float(item_gap), align='center',
                            children=[
            _legend_item(t, k, colors, fontsize, text_family, text_color,
                         dot_size, dot_edge_color, dot_edge_lw, label_gap,
                         dot_side)
            for t, k in entries]))

    box = AnchoredOffsetbox(
        loc=loc, child=VPacker(pad=0.0, sep=float(row_gap), align=align,
                               children=rows),
        pad=float(pad) / float(fontsize),   # AnchoredOffsetbox pads in font
        borderpad=0.0, frameon=bool(frame), #   units, not points
        prop=dict(size=fontsize),
        bbox_to_anchor=tuple(xy), bbox_transform=ax.transAxes)
    if frame:
        # The alphas go into the *colors*, not onto the patch: a patch's own
        # `alpha` overrides both of its colors at once, and the fill and the
        # outline here are the same grey told apart by nothing else — the
        # outline holding the shape while the fill lets the drawing through.
        from matplotlib.colors import to_rgba
        box.patch.set(
            facecolor=to_rgba(frame_facecolor, frame_alpha),
            edgecolor=to_rgba(frame_color,
                              frame_alpha if frame_edge_alpha is None
                              else frame_edge_alpha),
            linewidth=frame_lw)
        if frame_boxstyle:
            box.patch.set_boxstyle(frame_boxstyle)
    ax.add_artist(box)
    ax.axis('off')

    return fig, ax, box


# `figsize` / `dpi` belong to a figure of its own; on a shared one the figure is
# already made. Dropped rather than rejected so the notebook can hand the same
# config dict to a panel on its own and to `plot_figure7`.
_FIG_ONLY_KW = ('figsize', 'dpi')


def _panel_kw(kw):
    return {k: v for k, v in dict(kw or {}).items() if k not in _FIG_ONLY_KW}

# The mark that colours words inside a title. `*spines*` comes out purple in an
# otherwise black sentence — the sketch's own convention, kept as markup so the
# titles stay plain strings the notebook can retype at will.
_TITLE_MARK = '*'


def _title_segments(line, mark=_TITLE_MARK):
    """One title line → `[(text, is_highlighted), …]`.

    Splitting on the mark alternates outside/inside, and the spaces stay inside
    whichever segment they were typed in, so the pieces butt straight together
    and need no gap of their own (matplotlib does measure a trailing space).
    """
    return [(p, i % 2 == 1)
            for i, p in enumerate(str(line).split(mark)) if p != '']


# The bubble a heading's label sits in. A `Text`'s own `bbox`, so it is drawn
# and mutated with the label and needs no artist of its own — the cost being
# that it is *not* measured: `TextArea` sizes itself on the glyphs alone, so
# the bubble overhangs its box and `title_label_sep` is what keeps that
# overhang off the sentence underneath.
#
# One grey for every panel, on purpose. The reference this is drawn from tints
# each label its panel's own colour, which reads as five claims about five
# unrelated things; one grey reads as five parts of one page.
_FIG7_TITLE_BUBBLE = dict(boxstyle='round,pad=0.45', facecolor='0.85',
                          edgecolor='none', alpha=0.85)


def _split_title(spec):
    """A heading spec → `(label, description)`.

    A heading is either a plain **string** — one block of sentence, the way it
    has always been, with no bubble — or a `(label, description)` **pair**: a
    short capitalised label in a grey bubble over a smaller sentence saying
    what the panel claims. Either half of the pair may be `None`, which drops
    it: `('CONNECTIVITY', None)` is a bubble alone, and `(None, 'More
    *spines*…')` a sentence alone, which is how a panel sits under the bubble
    of the neighbour it shares a heading with.
    """
    if isinstance(spec, (tuple, list)):
        return ((spec[0] if len(spec) > 0 else None),
                (spec[1] if len(spec) > 1 else None))
    return None, spec


def _panel_title(ax, spec, xy, ha, va, fontsize, family, color,
                 highlight_color, line_sep, weight='normal',
                 label_fontsize=None, label_weight='bold', label_color='black',
                 label_family=None, label_sep=8.0, bubble=None):
    """One panel's title, drawn on its own axes in that axes' coordinates.

    `spec` is a string, or a `(label, description)` pair — see `_split_title`.
    In the description, `\\n` breaks it into lines and `*…*` colours words
    (`_title_segments`). A single `Text` cannot change colour mid-string, so
    each line is an `HPacker` of `TextArea`s and the lines a `VPacker` — which
    is what lets "prefer *spines*" be one sentence in two inks and still
    measure, wrap and align as one block.

    The label is one `TextArea` and so one ink: it is a name, not a sentence,
    and `*…*` in it is left as typed. It carries the bubble (`bubble`, a
    `Text`-`bbox` dict; `None` or `{}` draws none) and sits `label_sep` points
    above the description.

    `xy` is the anchor in axes coordinates; `ha` is both the alignment of the
    lines against one another and which end of the block lands on `xy`, and
    `va` which edge of it does. `('center', 'top')` at `(0.5, 1.0)` hangs the
    heading from the top of its band — the one anchor two bands of different
    depths agree on, and so the one that lines two headings up across the page.
    """
    from matplotlib.offsetbox import (AnchoredOffsetbox, HPacker, TextArea,
                                      VPacker)

    label, desc = _split_title(spec)
    if not label and not desc:
        return None

    children = []
    if label:
        children.append(TextArea(str(label), textprops=dict(
            family=label_family or family,
            size=float(label_fontsize or fontsize), weight=label_weight,
            color=label_color,
            bbox=(dict(bubble) if bubble else None))))

    if desc:
        props = dict(family=family, size=fontsize, weight=weight)
        rows = []
        for line in str(desc).split('\n'):
            segs = _title_segments(line) or [(' ', False)]
            rows.append(HPacker(pad=0.0, sep=0.0, align='baseline', children=[
                TextArea(t, textprops=dict(
                    props, color=highlight_color if hot else color))
                for t, hot in segs]))
        children.append(VPacker(pad=0.0, sep=float(line_sep), align=ha,
                                children=rows))

    box = (children[0] if len(children) == 1 else
           VPacker(pad=0.0, sep=float(label_sep), align=ha, children=children))
    loc = {('bottom', 'left'): 'lower left',
           ('bottom', 'center'): 'lower center',
           ('bottom', 'right'): 'lower right',
           ('center', 'left'): 'center left',
           ('center', 'center'): 'center',
           ('center', 'right'): 'center right',
           ('top', 'left'): 'upper left',
           ('top', 'center'): 'upper center',
           ('top', 'right'): 'upper right'}[(va, ha)]
    anchored = AnchoredOffsetbox(loc=loc, child=box, pad=0.0, borderpad=0.0,
                                 frameon=False, bbox_to_anchor=tuple(xy),
                                 bbox_transform=ax.transAxes)
    anchored.set_clip_on(False)
    ax.add_artist(anchored)
    return anchored


# The four panels' headings. Keys are `plot_figure7`'s own axes keys, so a
# panel is called the same thing here, in `axes`, in `out` and in its `*_kw`.
_FIG7_TITLES = {
    'a': 'Inh. neurons receive denser and\nmore synapses than Exc. neurons',
    'b': 'More *spines* more\npresynaptic partners',
    'c': ('Functional ensemble: shared presynaptic inhibition and high\n'
          '*intra-spine* connectivity'),
    'd': 'Long projecting Exc. axons\nprefer *spines*',
}

# The two left-hand titles are flush with the left edge of their column; the
# other two are centred over their panel.
_FIG7_TITLE_HAS = {'a': 'left', 'b': 'left'}
_FIG7_TITLE_XYS = {'a': (0.0, 1.0), 'b': (0.0, 1.0)}

# Share of each cell's height given to its title band. The two-line headings
# are the same height everywhere, so a taller cell needs a smaller share.
_FIG7_TITLE_RATIOS = {'a': 0.20, 'b': 0.20, 'c': 0.14, 'd': 0.12}


# The figure names its panels one of two ways. Either every panel carries a
# *heading* saying what it claims — the graphical-abstract reading, where the
# page is meant to stand without a caption — or every panel carries a bare
# *letter* for a caption to pick up, which is what the paper wants. One switch
# picks, for the whole page: half a figure lettered and half headed would be
# the worst of both. `PANEL_LETTERS` is that switch, and every function below
# reads it at call time, so a notebook can flip it with
# `figure7_utils.PANEL_LETTERS = True` and re-render.
PANEL_LETTERS = False

# Which letter each panel gets. The order is the *reader's*, not the code's:
# down the data half first (A the column, E its activity), then the schematic's
# four in the order they are meant to be read — B, C, D across the top band and
# F the ensemble across the bottom. Keys are the same panel keys used
# everywhere else, so nothing but this dict knows about the order.
_FIG7_LETTERS = {'column': 'A', 'a': 'B', 'b': 'C', 'd': 'D',
                 'traces': 'E', 'c': 'F'}

# How a letter is set, as against a heading. A letter is a label and not a
# sentence: bold, a size up, and in the corner of its panel rather than centred
# over the drawing. It ignores the headings' `title_xy` / `title_ha` /
# `title_va` (and their per-panel `*s` variants) — a one-word block has nothing
# to align, and the nudges those were given are about the sentences.
#
# Where a letter goes, and how to move one:
#
#   xys   `{key: (x, y)}` in **figure** coordinates — (0, 0) the bottom left of
#         the page, (1, 1) the top right, which is the grid
#         `figures_utils.debug_letter_placement(fig)` draws and the same
#         coordinates `figures_utils.add_panel_label` takes on the other
#         figures. A key left out (or set to `None`) is placed automatically,
#         at the top-left corner of that panel's own cell. Note these are
#         coordinates of a *page*, not of a panel: the full-page figure 7 and
#         the wide one put the same panel in different places, so a number
#         tuned on one is not true of the other — give each its own dict and
#         leave the other on the automatic corners.
#   dxy   `(dx, dy)` in figure coordinates, added to the *automatic* corners
#         only. The nudge for "all of them, a hair in off the edge"; an
#         explicit `xys` entry is taken exactly as written.
#   ha / va, has / vas   which corner of the letter lands on the point, for
#         all of them and then per panel. `('left', 'top')` means the point is
#         the letter's own top-left.
PANEL_LETTER_KW = dict(fontsize=22, weight='bold', color='black', family=None,
                       xys=None, dxy=(0.0, 0.0), ha='left', has=None,
                       va='top', vas=None)


def _by_key(value, key, default=None):
    """`value` as a per-panel setting: a dict is looked up, anything else is
    the one value every panel gets."""
    if isinstance(value, dict):
        return value.get(key, default)
    return value if value is not None else default


def _draw_letters(fig, axes, keys, letters, kw, family):
    """A bare letter per panel, on the figure itself.

    On the figure rather than on the panel's own axes because a letter is a
    label on the *page*: it is read against the other letters and the page's
    edges, not against the drawing under it, and half these panels are
    aspect-locked — their axes box moves as the drawing's aspect settles, which
    would carry a letter pinned to it along.

    Placement falls back to the top-left corner of the panel's heading band
    (its cell, since the band is a plain axes filling one), so the letters
    start where the headings started and only the ones that need it are given
    coordinates of their own.
    """
    for key in keys:
        text = letters.get(key)
        if not text:
            continue
        xy = _by_key(kw['xys'], key)
        if xy is None:
            ax = axes.get(f'title_{key}') or axes.get(key)
            if ax is None:
                continue
            box = ax.get_position()
            dx, dy = kw['dxy']
            xy = (box.x0 + dx, box.y1 + dy)
        fig.text(xy[0], xy[1], str(text),
                 ha=_by_key(kw['has'], key, kw['ha']) or kw['ha'],
                 va=_by_key(kw['vas'], key, kw['va']) or kw['va'],
                 family=kw['family'] or family, size=kw['fontsize'],
                 weight=kw['weight'], color=kw['color'], zorder=20)


def _lettering(panel_letters):
    """Whether this call letters its panels; `None` defers to the switch."""
    return bool(PANEL_LETTERS if panel_letters is None else panel_letters)


# A lettered page wants a *tighter grid* than a headed one, and the reason is
# the heading bands. A band is sized for what hangs in it — a bubble over two
# lines of claim — and a letter is one glyph: on the lettered page the rest of
# that band is white, and it is white in the worst place, between two panels
# that then read as further apart than they are. So every layout number a
# lettered page wants different from the headed one is a `letter_*` argument
# beside the one it overrides, applied only when the letters are on, and
# `None` — the default everywhere — means "whatever the headings use", so
# nothing about the headed page can move by adding one.
def _letter_ratio(ratio, letter_ratios, key, lettering):
    """One cell's title-band share, on the page this call is drawing.

    `letter_ratios` is `{key: share}` (or one share for every panel); a key it
    leaves out keeps the heading's `ratio`, which is what lets a page tighten
    the one band that needs it and say nothing about the rest.
    """
    return _by_key(letter_ratios, key, ratio) if lettering else ratio


def _draw_panel_names(fig, axes, keys, titles, panel_letters, letters,
                      letter_kw, style):
    """One half's panel names, drawn whichever way the switch says.

    `keys` is the panels this half owns and `style` its own `title_*` settings
    under short names. `panel_letters=None` defers to the module switch.
    """
    if not _lettering(panel_letters):
        _draw_titles(axes, titles, **style)
        return
    _draw_letters(fig, axes, keys,
                  _FIG7_LETTERS if letters is None else letters,
                  dict(PANEL_LETTER_KW, **dict(letter_kw or {})),
                  style['family'])


def _draw_titles(axes, titles, fontsize, family, color, highlight_color,
                 line_sep, xy, xys, ha, has, va, vas, weight='normal',
                 label_fontsize=None, label_weight='bold', label_color=None,
                 label_family=None, label_sep=8.0, bubble=None):
    """Every heading on its own band's axes; a panel given no band is skipped."""
    for key, spec in dict(titles or {}).items():
        t_ax = axes.get(f'title_{key}')
        if t_ax is None:
            continue
        _panel_title(t_ax, spec,
                     _by_key(xys, key, xy) or xy,
                     _by_key(has, key, ha) or ha,
                     _by_key(vas, key, va) or va,
                     fontsize, family, color, highlight_color, line_sep,
                     weight,
                     label_fontsize=label_fontsize, label_weight=label_weight,
                     label_color=(color if label_color is None
                                  else label_color),
                     label_family=label_family, label_sep=label_sep,
                     bubble=bubble)


def _titled_cell(fig, spec, ratio, hspace):
    """Split one grid cell into a title band over a panel.

    The title gets a band of its own rather than being hung off the panel's
    axes, because every panel here is aspect-locked: its axes box shrinks to
    the drawing and moves as the drawing's aspect changes, which would drag the
    heading with it. A band is a fixed share of the cell and stays put.

    Returns `(title_ax, panel_ax)`; `title_ax` is `None` when `ratio <= 0`.
    """
    r = float(ratio or 0.0)
    if r <= 0.0:
        return None, fig.add_subplot(spec)
    sub = spec.subgridspec(2, 1, height_ratios=(r, 1.0 - r), hspace=hspace)
    t_ax = fig.add_subplot(sub[0])
    t_ax.axis('off')
    return t_ax, fig.add_subplot(sub[1])


def plot_figure7(figsize=(11.0, 12.5), dpi=600,
                 height_ratios=(1.15, 1.0), width_ratios=(1.0, 1.20),
                 left_ratios=(1.15, 1.0), side_pad=(0.03, 0.03),
                 wspace=0.03, hspace=0.05, left_hspace=0.05,
                 title_ratios=_FIG7_TITLE_RATIOS, title_hspace=0.0,
                 margins=None, link_limits=(('a', 'b'),),
                 legend_kw=None, legend_cell='d',
                 panel_a_kw=None, panel_b_kw=None,
                 panel_c_kw=None, panel_d_kw=None,
                 titles=_FIG7_TITLES, title_fontsize=15, title_family='Arial',
                 title_color='black', title_highlight_color=_SPINY_COLOR,
                 title_line_sep=3.0, title_weight='normal',
                 title_label_fontsize=None, title_label_weight='bold',
                 title_label_color=None, title_label_family=None,
                 title_label_sep=8.0, title_bubble=None,
                 title_xy=(0.5, 1.0), title_xys=_FIG7_TITLE_XYS,
                 title_ha='center', title_has=_FIG7_TITLE_HAS,
                 title_va='top', title_vas=None,
                 panel_letters=None, letters=None, letter_kw=None,
                 letter_title_ratios=None):
    """The whole of figure 7 on one canvas.

    Two bands. The top one is a narrow left column — panel **a**
    (`plot_ex_inh_pair`) over panel **b** (`plot_spiny_aspiny_pair`) — beside a wider
    right column holding panel **d** (`plot_axon_panel`) at the full height of both,
    with the legend floating over its top-left corner. The bottom band is panel **c**
    (`plot_ex_circuit_panel`) alone, nearly the full width.

    Nothing is drawn here. Each `*_kw` is the full kwargs dict of the panel's own
    function, so every knob stays where it is documented and this function only places
    the axes. A panel's own `figsize` / `dpi` are dropped (`_FIG_ONLY_KW`), which lets
    the notebook pass the very same dicts it uses to draw that panel on its own. The
    keys 'a'/'b'/'c'/'d' name the panels throughout.

    Every panel is aspect-locked, so the ratios below set the space a panel is
    *offered*, not the size it takes: a panel that comes out too small is one whose
    cell is the wrong shape, and the fix is the ratios, never a scale inside the panel.

      height_ratios    top band : bottom band
      width_ratios     left column : right column, within the top band
      left_ratios      panel a : panel b, down the left column
      side_pad         `(left, right)` air beside panel c, as a fraction of figure width
      wspace / hspace / left_hspace   gaps between cells, as a fraction of cell size
      margins          `dict(left=…, right=…, top=…, bottom=…)` for the outer grid
      link_limits      groups of panels forced onto one common pair of data limits, and
                       so one common scale. Panels a and b are drawn to the same cartoon
                       cell and must come out the same size, but each fits its axes to
                       its own contents — and a's inhibitory cell is wider than b's
                       second pyramidal one, so left alone they settle at different
                       scales.
      legend_cell      which panel's cell the key is laid over; `legend_kw`'s own `xy`
                       (in that cell's coordinates) positions it

    Headings. A heading is either a plain sentence or a label over a description — a
    short capitalised name in a grey bubble with the claim under it — which is what a
    `(label, description)` pair in `titles` asks for (`_split_title`; either half may be
    None). `title_*` settings are the description's, `title_label_*` the bubble's:

      title_ratios     share of a cell's height given to its title band, scalar or
                       `{key: share}`; 0 drops the band. The band is what a heading
                       hangs from, not a box it is clipped to — one too shallow reaches
                       down over its own drawing.
      title_weight     the description's weight
      title_label_fontsize / title_label_weight / title_label_color / title_label_family
                       the bubble's lettering; each falls back to the description's
      title_label_sep  points between bubble and description. The bubble is a `Text`
                       `bbox`, drawn but not measured, so this is the clearance.
      title_bubble     the bubble, a `Text`-`bbox` dict (`_FIG7_TITLE_BUBBLE`); `None`
                       or `{}` sets the label with no bubble
      title_xy / title_xys    where a heading sits in its band, in band axes coords.
                       The default `(x, 1.0)` with `title_va='top'` hangs every heading
                       from the top of its band, which is what lines a's up with d's.
      title_ha / title_has, title_va / title_vas   alignment of the block and which of
                       its edges lands on `title_xy`
      title_highlight_color   ink for words wrapped in `*…*`

    Or letters instead of headings:

      panel_letters    True puts a bare letter in each heading band — the paper's figure
                       rather than the graphical abstract. `None` defers to the module
                       switch `PANEL_LETTERS`, which letters this figure and the wide
                       one together.
      letters          `{key: letter}`, default `_FIG7_LETTERS`. These four are B, C, F
                       and D, since A and E belong to the data half of
                       `plot_figure7_with_data` and the letters run in reading order
                       across both.
      letter_kw        overrides on `PANEL_LETTER_KW`: `fontsize`, `weight`, `color`,
                       and placement — `xys` = `{key: (x, y)}` in *figure* coordinates,
                       `dxy` to nudge the automatic ones, `ha`/`va` (and `has`/`vas`).
                       Left alone a letter takes its cell's top-left corner.
      letter_title_ratios   `title_ratios` for the lettered page, read only when the
                       letters are on: a band left at its heading depth is a stripe of
                       white above a letter. Panels a and b must stay equal to each
                       other whatever this says, or the linked pair comes out at two
                       different sizes.

    Returns (fig, axes, out) — `axes` is `dict(legend, a, b, c, d, title_a, …)` and
    `out` the panels' own return values under the panel keys.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes, out = _figure7_body(
        fig, None,
        height_ratios=height_ratios, width_ratios=width_ratios,
        left_ratios=left_ratios, side_pad=side_pad, wspace=wspace,
        hspace=hspace, left_hspace=left_hspace, title_ratios=title_ratios,
        title_hspace=title_hspace, margins=margins, link_limits=link_limits,
        legend_kw=legend_kw, legend_cell=legend_cell,
        panel_a_kw=panel_a_kw, panel_b_kw=panel_b_kw,
        panel_c_kw=panel_c_kw, panel_d_kw=panel_d_kw,
        titles=titles, title_fontsize=title_fontsize,
        title_family=title_family, title_color=title_color,
        title_highlight_color=title_highlight_color,
        title_line_sep=title_line_sep, title_weight=title_weight,
        title_label_fontsize=title_label_fontsize,
        title_label_weight=title_label_weight,
        title_label_color=title_label_color,
        title_label_family=title_label_family,
        title_label_sep=title_label_sep, title_bubble=title_bubble,
        title_xy=title_xy, title_xys=title_xys,
        title_ha=title_ha, title_has=title_has,
        title_va=title_va, title_vas=title_vas,
        panel_letters=panel_letters, letters=letters, letter_kw=letter_kw,
        letter_title_ratios=letter_title_ratios)
    return fig, axes, out


def _figure7_body(fig, spec,
                  height_ratios=(1.15, 1.0), width_ratios=(1.0, 1.20),
                  left_ratios=(1.15, 1.0), side_pad=(0.03, 0.03),
                  wspace=0.03, hspace=0.05, left_hspace=0.05,
                  title_ratios=_FIG7_TITLE_RATIOS, title_hspace=0.0,
                  margins=None, link_limits=(('a', 'b'),),
                  legend_kw=None, legend_cell='d',
                  panel_a_kw=None, panel_b_kw=None,
                  panel_c_kw=None, panel_d_kw=None,
                  titles=_FIG7_TITLES, title_fontsize=15, title_family='Arial',
                  title_color='black', title_highlight_color=_SPINY_COLOR,
                  title_line_sep=3.0, title_weight='normal',
                  title_label_fontsize=None, title_label_weight='bold',
                  title_label_color=None, title_label_family=None,
                  title_label_sep=8.0, title_bubble=None,
                  title_xy=(0.5, 1.0), title_xys=_FIG7_TITLE_XYS,
                  title_ha='center', title_has=_FIG7_TITLE_HAS,
                  title_va='top', title_vas=None,
                  panel_letters=None, letters=None, letter_kw=None,
                  letter_title_ratios=None):
    """Figure 7's four panels + key, drawn into `spec` on an existing figure.

    The whole of `plot_figure7` except making the canvas, so the same block of
    panels can be the whole page (`spec=None`, which is `plot_figure7`) or one
    cell of a bigger grid (`plot_figure7_with_data`, where it is the right-hand
    half). Every argument means exactly what it does on `plot_figure7` — see
    there — save that `margins` only applies when `spec is None`: a nested grid
    has no figure-coordinate margins of its own, its parent's cell *is* its
    margin.

    Returns `(axes, out)`; the caller owns the figure.
    """
    if spec is None:
        gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=hspace,
                              **dict(margins or {}))
    else:
        gs = spec.subgridspec(2, 1, height_ratios=height_ratios, hspace=hspace)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=width_ratios, wspace=wspace)
    gs_left = gs_top[0, 0].subgridspec(2, 1, height_ratios=left_ratios,
                                       hspace=left_hspace)

    # Panel c's side air. A zero-width column is not worth asking matplotlib
    # for, so the band is only split when there is something to split off.
    lpad, rpad = (float(v) for v in side_pad)
    if lpad > 0.0 or rpad > 0.0:
        gs_bot = gs[1].subgridspec(1, 3, wspace=0.0,
                                   width_ratios=(max(lpad, 1e-6),
                                                 1.0 - lpad - rpad,
                                                 max(rpad, 1e-6)))
        bot_cell = gs_bot[0, 1]
    else:
        bot_cell = gs[1]

    cells = {'a': gs_left[0], 'b': gs_left[1], 'c': bot_cell,
             'd': gs_top[0, 1]}

    lettering = _lettering(panel_letters)

    axes = {}
    for key in ('a', 'b', 'c', 'd'):
        t_ax, p_ax = _titled_cell(fig, cells[key],
                                  _letter_ratio(_by_key(title_ratios, key, 0.0),
                                                letter_title_ratios, key,
                                                lettering),
                                  title_hspace)
        axes[key] = p_ax
        if t_ax is not None:
            axes[f'title_{key}'] = t_ax

    # The key gets an axes over the whole of its host cell and places itself
    # inside that in points (`plot_figure7_legend`), so `legend_kw['xy']` reads
    # as a plain corner-relative coordinate of the cell it floats over.
    box = cells[legend_cell].get_position(fig)
    axes['legend'] = fig.add_axes((box.x0, box.y0, box.width, box.height),
                                  zorder=10)
    axes['legend'].patch.set_visible(False)

    out = {}
    _, _, out['a'] = plot_ex_inh_pair(ax=axes['a'], **_panel_kw(panel_a_kw))
    _, _, out['b'] = plot_spiny_aspiny_pair(ax=axes['b'],
                                            **_panel_kw(panel_b_kw))
    _, _, out['c'] = plot_ex_circuit_panel(ax=axes['c'], **_panel_kw(panel_c_kw))
    _, _, out['d'] = plot_axon_panel(ax=axes['d'], **_panel_kw(panel_d_kw))
    _, _, out['legend'] = plot_figure7_legend(ax=axes['legend'],
                                              **_panel_kw(legend_kw))

    # Common limits last: each panel has to have finished fitting itself to its
    # own contents before there is anything to take the union of.
    for group in (link_limits or ()):
        group = [k for k in group if k in axes]
        if len(group) < 2:
            continue
        xs = [axes[k].get_xlim() for k in group]
        ys = [axes[k].get_ylim() for k in group]
        xlim = (min(v[0] for v in xs), max(v[1] for v in xs))
        ylim = (min(v[0] for v in ys), max(v[1] for v in ys))
        for k in group:
            axes[k].set_xlim(xlim)
            axes[k].set_ylim(ylim)

    _draw_panel_names(
        fig, axes, ('a', 'b', 'c', 'd'), titles, panel_letters, letters,
        letter_kw,
        dict(fontsize=title_fontsize, family=title_family, color=title_color,
             highlight_color=title_highlight_color, line_sep=title_line_sep,
             weight=title_weight, label_fontsize=title_label_fontsize,
             label_weight=title_label_weight, label_color=title_label_color,
             label_family=title_label_family, label_sep=title_label_sep,
             bubble=title_bubble,
             xy=title_xy, xys=title_xys, ha=title_ha, has=title_has,
             va=title_va, vas=title_vas))

    return axes, out


# ---------------------------------------------------------------------------
# The data panels, and figure 7 with them
# ---------------------------------------------------------------------------
#
# Everything above draws cartoons: shapes made up on the spot out of numbers in
# the call. Everything below draws *measurements* — the column's cell bodies,
# its reconstructed arbors, its calcium traces — so each of these takes the
# already-loaded data as an argument and does nothing but ink it. Loading stays
# in the notebook, where the paths, the caches and the slow calls live.


def plot_column_scatter(ax=None, neurons_df=None,
                        x='pt_position_xt', y='pt_position_yt',
                        hue='clf_type', palette=None, hue_order=('E', 'I'),
                        s=12, alpha=0.8, linewidths=0.0, edgecolors='none',
                        color=_EX_COLOR, ylim=(800.0, 0.0), xlim=None,
                        aspect=None, scalebar=True, scalebar_size=100.0,
                        scalebar_label='100 µm', scalebar_loc='lower center',
                        scalebar_color='#2B2B2B', scalebar_fontsize=10,
                        scalebar_pad=0.5, scalebar_borderpad=0.5,
                        scalebar_sep=3, scalebar_size_vertical=1,
                        figsize=(2.0, 8.0), dpi=150):
    """The column as dots: one cell body per neuron, in its place in the slab.

    The left-most thing on the figure and the widest statement it makes — this
    is the population everything to the right is about, and the only panel that
    shows all of it at once. Excitatory red, inhibitory blue (`palette`), drawn
    from `neurons_df`'s transformed soma coordinates, so `x`/`y` are already
    physical µm and a `scalebar_size` of 100 is 100 µm on the page.

      neurons_df   any frame with the two position columns and, if `hue` is
                   given, the column to color by.
      hue          the column split into series, one `scatter` each, in
                   `hue_order` (levels not in the frame are skipped, levels not
                   in `hue_order` follow in the order they appear). `None`
                   draws every dot in `color`.
      ylim         `(800, 0)` — **descending on purpose**: pia at the top, as
                   every depth axis in the paper has it.
      aspect       `None` leaves matplotlib's `'auto'`, which stretches the
                   slab to fill its cell. `'equal'` makes the panel true to
                   scale in both directions, and then a narrow cell crops it.

    Returns (fig, ax, dict(groups={level: n}, scalebar=artist|None)).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if neurons_df is None or len(neurons_df) == 0:
        raise ValueError('plot_column_scatter needs a neurons frame')

    pal = dict(palette or {'E': _EX_COLOR, 'I': _INH_COLOR})

    groups = {}
    if hue is None:
        ax.scatter(neurons_df[x], neurons_df[y], s=s, alpha=alpha, color=color,
                   linewidths=linewidths, edgecolors=edgecolors)
        groups[None] = len(neurons_df)
    else:
        levels = [v for v in (hue_order or ()) if v in set(neurons_df[hue])]
        levels += [v for v in dict.fromkeys(neurons_df[hue]) if v not in levels]
        for level in levels:
            sub = neurons_df[neurons_df[hue] == level]
            ax.scatter(sub[x], sub[y], s=s, alpha=alpha,
                       color=pal.get(level, color), linewidths=linewidths,
                       edgecolors=edgecolors)
            groups[level] = len(sub)

    if aspect is not None:
        ax.set_aspect(aspect)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    bar = None
    if scalebar:
        import matplotlib.font_manager as fm
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        bar = AnchoredSizeBar(
            transform=ax.transData, size=float(scalebar_size),
            label=scalebar_label, sep=scalebar_sep, loc=scalebar_loc,
            pad=scalebar_pad, color=scalebar_color, frameon=False,
            size_vertical=scalebar_size_vertical, borderpad=scalebar_borderpad,
            fontproperties=fm.FontProperties(size=scalebar_fontsize))
        ax.add_artist(bar)

    ax.set_axis_off()
    return fig, ax, {'groups': groups, 'scalebar': bar}


def plot_column_cells(ax=None, skeletons=None, triangulations=None,
                      colors=None, mode='skeletons', color=_EX_COLOR,
                      lw=1.4, alpha=1.0, mesh_lw=0.005, mesh_alpha=0.75,
                      ignore_vertex_zero=False, ylim=(750.0, 0.0), xlim=None,
                      fit_x=True, x_margin=0.02, fit_y=True, y_margin=0.01,
                      aspect=None, figsize=(2.0, 20.0), dpi=300):
    """A handful of the column's cells, reconstructed, beside the dots.

    Same slab, one step in: where `plot_column_scatter` has every soma as a
    dot, this has a few whole cells with their arbors, in the same E/I ink. The
    two are meant to be read together and so want the same `ylim`.

      mode         `'skeletons'` draws `skeletons` as variable-width ribbons
                   (`plot_utils.plot_skeleton_continuous`, which uses each
                   node's own radius); `'meshes'` draws `triangulations` — the
                   segmented surface itself, as a hairline wireframe. The
                   skeletons are the fast, clean read and the default; the
                   meshes are the honest one, and cost a minute of loading and
                   a much heavier file.
      colors       one color per cell, in the order the cells are given. A
                   single color, or `None`, gives every cell `color`.
      lw / alpha   the skeleton ribbons; `mesh_lw` / `mesh_alpha` the
                   wireframe, which wants a hairline — a mesh has tens of
                   thousands of triangles and any real linewidth fills solid.
      fit_x        fit the x limits to the cells that are **visible** — the
                   part of each arbor inside `ylim`. Matplotlib autoscales x
                   over every vertex it was handed, including the ones the
                   depth crop then hides, so a single branch that dives past
                   the bottom of the panel otherwise buys itself a column of
                   white the whole height of the figure. `x_margin` is the air
                   left either side, as a fraction of that fitted width. An
                   explicit `xlim` wins over both.
      fit_y        the same downward: pull `ylim` in to the depth the cells
                   actually reach. `ylim` is a slab depth — the one the dots
                   beside them are drawn to — and a handful of cells rarely
                   fills it top to bottom, leaving a band of white under the
                   deepest arbor. This only ever *tightens* `ylim`, never
                   reaches past it, so the panel stays a crop of the same slab.
                   `y_margin` is its air, as a fraction of the fitted depth.
      aspect       as `plot_column_scatter`. Meshes are usually worth
                   `'equal'`: a stretched wireframe reads as a smear.
                   `plot_skeleton_continuous` locks it to `'equal'` on its own
                   — a skeleton is drawn at true scale whatever this says.

    Both inputs are whatever the notebook already built — `load_col_skeleton_only`
    skeletons, `matplotlib.tri.Triangulation`s of dendrite meshes — so nothing
    is loaded here.

    Returns (fig, ax, dict(mode=…, n=…)).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    items = list(skeletons or ()) if mode == 'skeletons' else list(triangulations or ())
    if not items:
        raise ValueError(f'plot_column_cells: nothing to draw in mode {mode!r}')

    if colors is None or isinstance(colors, str):
        cols = [colors or color] * len(items)
    else:
        cols = list(colors)
        cols += [color] * (len(items) - len(cols))

    if mode == 'skeletons':
        from plot_utils import plot_skeleton_continuous
        for sk, c in zip(items, cols):
            plot_skeleton_continuous(ax=ax, sk=sk, lw=lw, alpha=alpha, color=c,
                                     ignore_vertex_zero=ignore_vertex_zero)
    elif mode == 'meshes':
        for triang, c in zip(items, cols):
            ax.triplot(triang, color=c, alpha=mesh_alpha, lw=mesh_lw)
    else:
        raise ValueError(f"mode must be 'skeletons' or 'meshes', got {mode!r}")

    if aspect is not None:
        ax.set_aspect(aspect)

    # The drawn coordinates, per cell, whichever kind of thing was drawn.
    def _xy(item):
        if mode == 'skeletons':
            return item.vertices[:, 0], item.vertices[:, 1]
        return np.asarray(item.x), np.asarray(item.y)

    lo = hi = None
    if ylim is not None:
        lo, hi = sorted(float(v) for v in ylim)
        if fit_y:
            seen = [(vy[(vy >= lo) & (vy <= hi)].min(),
                     vy[(vy >= lo) & (vy <= hi)].max())
                    for vy in (_xy(item)[1] for item in items)
                    if ((vy >= lo) & (vy <= hi)).any()]
            if seen:
                y0 = min(b[0] for b in seen)
                y1 = max(b[1] for b in seen)
                pad = float(y_margin) * (y1 - y0)
                lo, hi = max(lo, y0 - pad), min(hi, y1 + pad)
        # `ylim` normally runs pia-down, so put it back the way it came
        ax.set_ylim(*((hi, lo) if float(ylim[0]) > float(ylim[1]) else (lo, hi)))

    if xlim is not None:
        ax.set_xlim(*xlim)
    elif fit_x and lo is not None:
        bounds = []
        for item in items:
            vx, vy = _xy(item)
            seen = (vy >= lo) & (vy <= hi)
            if seen.any():
                bounds.append((vx[seen].min(), vx[seen].max()))
        if bounds:
            x0 = min(b[0] for b in bounds)
            x1 = max(b[1] for b in bounds)
            pad = float(x_margin) * (x1 - x0)
            ax.set_xlim(x0 - pad, x1 + pad)

    ax.set_axis_off()

    return fig, ax, {'mode': mode, 'n': len(items)}


def plot_activity_traces(ax=None, traces=None, t=None, fps=None,
                         color=_EX_COLOR, lw=1.2, alpha=0.85, spacing=1.1,
                         normalize=True, reverse=True, zero_time=True,
                         xlim=None, figsize=(11.0, 4.0), dpi=250):
    """A few cells' activity, stacked — the figure's only time axis.

    Everything else on the page is anatomy; this is what the anatomy is for,
    and it sits under the column it was recorded in.

      traces     `(n, T)`, one row per cell — spike or calcium traces, already
                 windowed to the stretch worth showing.
      t          the row of time values. `None` builds one from `fps`, or falls
                 back to sample number; `zero_time` then slides it to start at
                 0 so the axis reads as seconds-into-the-window.
      normalize  scale each row by its own peak, so a quiet cell and a loud one
                 both fill their band. Off, the rows keep their relative
                 heights and `spacing` has to clear the tallest.
      spacing    the gap between rows, in normalized units — `1.1` leaves a
                 tenth of a row of air.
      reverse    stack the first row on top. The rows are drawn bottom-up, so
                 without this the frame's first cell ends up at the bottom.

    Returns (fig, ax, dict(t=…, offsets=…, scales=…)).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    mat = np.atleast_2d(np.asarray(traces, dtype=float))
    if mat.size == 0:
        raise ValueError('plot_activity_traces needs at least one trace')

    if t is None:
        t = np.arange(mat.shape[1]) / float(fps) if fps else np.arange(mat.shape[1])
    t = np.asarray(t, dtype=float)
    if zero_time:
        t = t - t[0]

    rows = mat[::-1] if reverse else mat
    offsets, scales = [], []
    for i, row in enumerate(rows):
        peak = row.max()
        scale = (1.0 / peak if peak > 0 else 1.0) if normalize else 1.0
        off = i * float(spacing)
        ax.plot(t, row * scale + off, color=color, lw=lw, alpha=alpha)
        offsets.append(off)
        scales.append(scale)

    ax.set_xlim(*(xlim if xlim is not None else (t[0], t[-1])))
    ax.axis('off')

    return fig, ax, {'t': t, 'offsets': offsets, 'scales': scales}


def _titled_band(fig, spec, ratio, hspace):
    """`_titled_cell` for a cell that is not a single panel.

    Same split — a title band over the rest — but the rest is handed back as a
    *spec*, not an axes, so the caller can subdivide it (the column band holds
    two panels side by side).
    """
    r = float(ratio or 0.0)
    if r <= 0.0:
        return None, spec
    sub = spec.subgridspec(2, 1, height_ratios=(r, 1.0 - r), hspace=hspace)
    t_ax = fig.add_subplot(sub[0])
    t_ax.axis('off')
    return t_ax, sub[1]


def _recentre_scalebar(fig, bar, axs):
    """Re-anchor `bar` on the union of `axs`' *drawn* boxes.

    An `AnchoredSizeBar` anchors to the axes it was added to, so the column
    band's bar centres under the dots alone — which is a third of the width the
    reader sees as "the column", the cells beside them being the rest of it.
    Handing it the pair's union instead moves it to the middle of the block.

    The union is taken over the boxes the two are *drawn* in, not the grid cells
    they were given: `plot_column_cells` locks its aspect, so its box shrinks
    inside its cell and hangs off one edge of it (`column_anchors`), and the
    dead air that leaves would otherwise drag the bar out past the ink.
    `apply_aspect` is what settles that box — matplotlib runs it at draw time,
    and running it here just brings the same answer forward.

    Only the anchor moves. The bar is still `scalebar_size` µm in the *scatter's*
    x scale, which is the one it was built in and the only one it is true to —
    an aspect-locked neighbour drawn at a different scale is not measured by it.
    """
    from matplotlib.transforms import Bbox

    for a in axs:
        a.apply_aspect()
    bar.set_bbox_to_anchor(Bbox.union([a.get_position() for a in axs]),
                           transform=fig.transFigure)


def _cell_box(axes, keys):
    """The union of the grid **cells** the named panels were given.

    Their cells, not the boxes they are drawn in: every panel here is
    aspect-locked, so its drawn box shrinks inside its cell and hangs off one
    edge of it, and a rule ruled on *that* would sit wherever the ink happened
    to settle rather than on the gap the layout actually left. `original=True`
    is the cell as the gridspec handed it out, before `apply_aspect` — which
    `_recentre_scalebar` has already run by the time these are drawn.

    A panel's heading band is a cell of its own, so both halves are unioned
    and a rule beside a panel runs past its heading too.
    """
    from matplotlib.transforms import Bbox

    boxes = [axes[name].get_position(original=True)
             for key in keys
             for name in (key, f'title_{key}') if name in axes]
    return Bbox.union(boxes) if boxes else None


# Where the page is ruled, when it carries headings rather than letters.
#
# Each entry is one hairline. `between` is two groups of panel keys and the
# rule goes down (`'v'`) or across (`'h'`) the gap between their cells,
# spanning the two groups together — so a rule follows the grid and stays
# true when the ratios are retuned. `at` / `span` (figure coordinates) override
# either half of that, `offset` nudges the computed `at`, and `shrink` is the
# fraction of the span pulled in at each end so a rule stops short of the
# page's edge instead of running into it.
#
# These four are the joins a reader would draw themselves: the measured half
# off the claimed one, the convergence pair off the axon, the top band off the
# ensemble, and the two convergence panels off each other. The ensemble gets a
# rule and *not* a panel of its own colour — a tinted box around one panel says
# it is a different kind of thing, and it is not.
_FIG7_SEPARATORS = (
    dict(orient='v', between=(('scatter', 'cells', 'traces'), ('a', 'b', 'd'))),
    dict(orient='v', between=(('a', 'b'), ('d',))),
    dict(orient='h', between=(('a', 'b', 'd'), ('c',))),
    dict(orient='h', between=(('a',), ('b',))),
)


def _draw_separators(fig, axes, specs, color='0.8', lw=0.8, alpha=1.0,
                     shrink=0.02, zorder=0):
    """The hairlines between the panels, on the figure itself.

    On the figure because a rule belongs to the *page*: it is a statement about
    where one block of it ends, and there is no one panel it is part of.

    Returns the `Line2D`s drawn, in the order asked for.
    """
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    drawn = []
    for spec in (specs or ()):
        s = dict(spec)
        vert = str(s.get('orient', 'v')).lower().startswith('v')
        at, span = s.get('at'), s.get('span')

        if at is None or span is None:
            groups = s.get('between') or ((), ())
            boxes = [_cell_box(axes, g) for g in groups[:2]]
            if len(boxes) < 2 or any(b is None for b in boxes):
                continue    # a panel this rule is about was never drawn
            first, second = boxes
            if at is None:
                # whichever group comes first along the axis owns the near edge
                if vert:
                    lo, hi = ((first, second) if first.x1 <= second.x0
                              else (second, first))
                    at = 0.5 * (lo.x1 + hi.x0)
                else:
                    lo, hi = ((first, second) if first.y1 <= second.y0
                              else (second, first))
                    at = 0.5 * (lo.y1 + hi.y0)
            if span is None:
                union = Bbox.union(boxes)
                span = (union.y0, union.y1) if vert else (union.x0, union.x1)

        s0, s1 = (float(v) for v in span)
        pull = float(s.get('shrink', shrink)) * (s1 - s0)
        s0, s1 = s0 + pull, s1 - pull
        at = float(at) + float(s.get('offset', 0.0))

        line = (Line2D([at, at], [s0, s1]) if vert else
                Line2D([s0, s1], [at, at]))
        line.set(transform=fig.transFigure, color=s.get('color', color),
                 linewidth=float(s.get('lw', lw)),
                 alpha=float(s.get('alpha', alpha)),
                 zorder=float(s.get('zorder', zorder)), solid_capstyle='round')
        fig.add_artist(line)
        drawn.append(line)
    return drawn


# The data half's two headings, over the block each names. A `(label, None)`
# pair is a bubble with no sentence under it: what these two blocks are is
# plain from the ink, and a description would only say it again.
_FIG7_DATA_TITLES = {'column': ('CONNECTIVITY', None),
                     'traces': ('ACTIVITY', None)}

# Share of each band's height given to its heading. Both are one line, so the
# tall column band needs far less of itself than the shallow trace band.
_FIG7_DATA_TITLE_RATIOS = {'column': 0.04, 'traces': 0.14}


def plot_figure7_with_data(
        figsize=(19.0, 12.5), dpi=300,
        width_ratios=(1.0, 1.55), wspace=0.04, margins=None,
        data_ratios=(0.8, 0.2), data_hspace=0.06,
        column_ratios=(0.4, 0.6), column_wspace=-0.05,
        column_anchors=('E', 'W'), column_scalebar='scatter',
        scatter_df=None, scatter_kw=None,
        cells_mode='skeletons', skeletons=None, triangulations=None,
        cell_colors=None, cells_kw=None,
        traces=None, trace_t=None, trace_fps=None, traces_kw=None,
        titles=_FIG7_DATA_TITLES, title_ratios=_FIG7_DATA_TITLE_RATIOS,
        title_hspace=0.0, title_fontsize=15, title_family='Arial',
        title_color='black', title_highlight_color=_SPINY_COLOR,
        title_line_sep=3.0, title_weight='normal',
        title_label_fontsize=None, title_label_weight='bold',
        title_label_color=None, title_label_family=None,
        title_label_sep=8.0, title_bubble=_FIG7_TITLE_BUBBLE,
        title_xy=(0.5, 1.0), title_xys=None,
        title_ha='center', title_has=None, title_va='top', title_vas=None,
        separators=_FIG7_SEPARATORS, separator_color='0.8',
        separator_lw=0.8, separator_alpha=1.0, separator_shrink=0.02,
        separator_zorder=0,
        panel_letters=None, letters=None, letter_kw=None,
        letter_title_ratios=None, letter_data_ratios=None,
        letter_data_hspace=None, schematic_kw=None):
    """Figure 7 with the data it is drawn from, side by side.

    The right half is `plot_figure7` unchanged — the four cartoon panels and their key.
    The left half is the measurements: the column as dots (`plot_column_scatter`) beside
    a few of its cells reconstructed (`plot_column_cells`), and under both a stack of
    activity traces (`plot_activity_traces`). Two headings label the left half, one per
    band; the right half keeps its own four. Everything on the left is measured and
    everything on the right is claimed.

      width_ratios      data half : schematic half
      data_ratios       column band : trace band, down the data half
      column_ratios     scatter : cells, across the column band
      column_anchors    which edge of its cell each keeps when it cannot fill it.
                        `plot_column_cells` locks its aspect, so its box shrinks inside
                        its cell and would otherwise centre there. `('E', 'W')` faces
                        the two at each other; a negative `column_wspace` then pushes
                        them until they touch.
      column_scalebar   what the dots' scale bar is centred under: `'scatter'` leaves it
                        in the middle of the dots' own cell, `'band'` re-anchors it on
                        dots and cells together (`_recentre_scalebar`). Its *length* is
                        the dots' x scale either way.
      wspace / data_hspace / column_wspace   gaps, as a fraction of cell size.
                        `column_wspace` may go negative, overlapping the two cells; the
                        arbors are drawn over the dots when it does.
      margins           `dict(left=…, right=…, top=…, bottom=…)` for the outer 1×2 grid

    The data:

      scatter_df        the neurons frame, straight to `plot_column_scatter`;
                        `scatter_kw` is the rest of that function's kwargs
      cells_mode        `'skeletons'` or `'meshes'` — which of `skeletons` /
                        `triangulations` `plot_column_cells` draws. `cell_colors` is one
                        color per cell, `cells_kw` the rest.
      traces            `(n, T)`; `trace_t` or `trace_fps` gives it a time axis,
                        `traces_kw` the rest

    Headings work as `plot_figure7`'s do, over the keys `'column'` and `'traces'`. Both
    default to a bubble with no description under it.

    The page is ruled: a hairline across each join between two blocks.

      separators        one dict per rule — `orient` 'v'/'h', `between` two groups of
                        panel keys, and `at` / `span` / `offset` / `shrink` / `color` /
                        `lw` / `alpha`. See `_FIG7_SEPARATORS`; `()` draws none.
      separator_color / separator_lw / separator_alpha / separator_shrink /
      separator_zorder  the default ink and reach for all of them

    Headings and rules both belong to the graphical-abstract reading of this page, and
    neither is drawn when `panel_letters` is on — a lettered page is the paper's figure,
    where the caption does the dividing.

      panel_letters     True letters the panels instead of heading them, here and in the
                        schematic half both: A (column), B, C, D (schematic top band),
                        E (traces), F (ensemble) — reading order across the whole page,
                        which is why it is one switch. `None` defers to `PANEL_LETTERS`.
      letter_title_ratios / letter_data_ratios / letter_data_hspace
                        the grid for a lettered page, over this half's keys. Read only
                        when the letters are on, and `None` means "whatever the headings
                        use", so none of them can move the headed page. The gap between
                        the two bands exists to keep the trace band's bubble off the
                        arbors above it; a letter needs no such clearance, so a lettered
                        page closes it and hands the height to the trace band.
      schematic_kw      the whole of what would be passed to `plot_figure7`. Its own
                        `figsize` / `dpi` are dropped (`_FIG_ONLY_KW`) and so is
                        `margins` — nested, its cell is its margin. Everything else is
                        honored, so the notebook can hand the same dict to
                        `plot_figure7` and to this and get the same right-hand half.

    Returns (fig, axes, out) — `axes` and `out` hold the schematic's keys (`a`–`d`,
    `legend`, `title_a`…) and the data half's (`scatter`, `cells`, `traces`,
    `title_column`, `title_traces`) in one flat dict each, with the rules under
    `out['separators']`.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=wspace,
                          **dict(margins or {}))

    # The lettered page's own grid, where it asked for one.
    lettering = _lettering(panel_letters)
    if lettering:
        if letter_data_ratios is not None:
            data_ratios = letter_data_ratios
        if letter_data_hspace is not None:
            data_hspace = letter_data_hspace

    gs_data = gs[0, 0].subgridspec(2, 1, height_ratios=data_ratios,
                                   hspace=data_hspace)

    axes, out = {}, {}

    # The column band: one heading over a scatter and a set of cells.
    t_col, col_spec = _titled_band(fig, gs_data[0],
                                   _letter_ratio(
                                       _by_key(title_ratios, 'column', 0.0),
                                       letter_title_ratios, 'column',
                                       lettering),
                                   title_hspace)
    gs_col = col_spec.subgridspec(1, 2, width_ratios=column_ratios,
                                  wspace=column_wspace)
    axes['scatter'] = fig.add_subplot(gs_col[0, 0])
    axes['cells'] = fig.add_subplot(gs_col[0, 1])
    # The two face each other (`column_anchors`) so that what air an
    # aspect-locked panel cannot fill falls on the *outside* of the pair, and
    # the cells are drawn over the dots — with a negative `column_wspace` the
    # boxes overlap, and an arbor reaching across should pass in front rather
    # than be cut off by the neighbouring axes' background.
    s_anchor, c_anchor = column_anchors
    axes['scatter'].set_anchor(s_anchor)
    axes['cells'].set_anchor(c_anchor)
    axes['cells'].set_zorder(axes['scatter'].get_zorder() + 1)
    axes['cells'].patch.set_visible(False)
    if t_col is not None:
        axes['title_column'] = t_col

    # The trace band, across the whole of the data half.
    t_tr, tr_spec = _titled_band(fig, gs_data[1],
                                 _letter_ratio(
                                     _by_key(title_ratios, 'traces', 0.0),
                                     letter_title_ratios, 'traces', lettering),
                                 title_hspace)
    axes['traces'] = fig.add_subplot(tr_spec)
    if t_tr is not None:
        axes['title_traces'] = t_tr

    _, _, out['scatter'] = plot_column_scatter(
        ax=axes['scatter'], neurons_df=scatter_df, **_panel_kw(scatter_kw))
    _, _, out['cells'] = plot_column_cells(
        ax=axes['cells'], skeletons=skeletons, triangulations=triangulations,
        colors=cell_colors, mode=cells_mode, **_panel_kw(cells_kw))
    _, _, out['traces'] = plot_activity_traces(
        ax=axes['traces'], traces=traces, t=trace_t, fps=trace_fps,
        **_panel_kw(traces_kw))

    if column_scalebar == 'band' and out['scatter'].get('scalebar') is not None:
        _recentre_scalebar(fig, out['scatter']['scalebar'],
                           (axes['scatter'], axes['cells']))
    elif column_scalebar not in ('scatter', 'band'):
        raise ValueError("column_scalebar must be 'scatter' or 'band', got "
                         f'{column_scalebar!r}')

    _draw_panel_names(
        fig, axes, ('column', 'traces'), titles, panel_letters, letters,
        letter_kw,
        dict(fontsize=title_fontsize, family=title_family, color=title_color,
             highlight_color=title_highlight_color, line_sep=title_line_sep,
             weight=title_weight, label_fontsize=title_label_fontsize,
             label_weight=title_label_weight, label_color=title_label_color,
             label_family=title_label_family, label_sep=title_label_sep,
             bubble=title_bubble,
             xy=title_xy, xys=title_xys, ha=title_ha, has=title_has,
             va=title_va, vas=title_vas))

    # The right half, from the very dict `plot_figure7` would have taken.
    schematic = {k: v for k, v in dict(schematic_kw or {}).items()
                 if k not in _FIG_ONLY_KW and k != 'margins'}
    # One switch letters the whole page: the schematic half is lettered exactly
    # when this one is, unless its own dict says otherwise.
    for k, v in (('panel_letters', panel_letters), ('letters', letters),
                 ('letter_kw', letter_kw)):
        schematic.setdefault(k, v)
    s_axes, s_out = _figure7_body(fig, gs[0, 1], **schematic)
    axes.update(s_axes)
    out.update(s_out)

    # Last, and only under headings: the rules need every cell on the page to
    # exist before there is a gap between two of them to find, and a lettered
    # page is the paper's figure, where the caption does the dividing.
    out['separators'] = ([] if lettering else
                         _draw_separators(fig, axes, separators,
                                          color=separator_color,
                                          lw=separator_lw,
                                          alpha=separator_alpha,
                                          shrink=separator_shrink,
                                          zorder=separator_zorder))

    return fig, axes, out
