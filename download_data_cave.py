"""
Rebuild `data/` from the MICrONS public release — one script, one run.

This is route 2 of the two ways to populate `data/`. `download_data_zenodo.py` is route
1 and takes minutes: it pulls the already-formatted tables from Zenodo. This script is
the slow path, for anyone who would rather rebuild those tables from the source than
take an archive on trust. Both routes are pinned to the same materialization and yield
the same files (three documented exceptions, at the bottom of this docstring).

It writes everything under `data/` except the one thing it cannot:

  data/activity/   the calcium recordings and the ensemble results, behind figures 6,
                   7, S14 and S15. Not on Zenodo either — the recordings are too large
                   to redistribute. Built by scripts/extract_calcium_data_via_docker.ipynb
                   and then scripts/ensemble_run.py — see the README.

Nothing here rebuilds the pre-rendered dendrite images behind figures 1, 2 and S1 —
they were traced and rendered by hand in VAST, not derived from anything CAVE serves.
Neither route can, which is why they ship in the repository itself, in `images/`.

Everything else comes from CAVEclient against `minnie65_public` at materialization
v1718 (`MATERIALIZATION_VERSION`, utils/connectome_types.py). Pinning that version is
what makes the run reproducible: root ids are not stable across materializations.

    python download_data_cave.py                    # everything, in order
    python download_data_cave.py --steps raw,column # only these

Every step skips what is already on disk, so an interrupted run resumes where it
stopped. Budget roughly a day of wall clock and ~25 GB. The two per-neuron CAVE loops
(synapses, then spine tags) dominate — one query per neuron, no bulk endpoint. Peak RAM
is in `connectome_outgoing_synapses.csv`, which holds every synapse of every neuron in
the network at once: ~8 GB for the column, ~16 GB for the proofread-axon network.

Steps, in dependency order:

  raw       data/raw_tables/                            minutes
  column    data/micro_column_network/                  ~a day
              neurons/        1,351 .pkl — every synapse in the volume
              skeletons/      1,351 .swc
              spine_table.csv, spine_table_outgoing.csv
              em_neurons/     1,351 .pkl — within-column synapses only, plus
                              per-synapse distance to soma and cable lengths
              connectome_neurons.csv, connectome_synapses.csv
              connectome_outgoing_synapses.csv
              connectivity_matrix/
  subnets   data/micro_column_network/subnetworks/      minutes, derived, offline
  meshes    data/micro_column_network/meshes/           ~1.4 GB — figures 1, 2, 7, S1
  axon_pr   data/all_axon_pr_network/                   ~a day — figure S3 only

`axon_pr` is the same pipeline over a different cell list (every cell with a proofread
axon) and is needed by one supplementary figure. Skip it with `--steps` if S3 is not
what you are after.

Ported from `connectome_download.py` in the research repo, trimmed to the chain that
produces the shipped files: no MorphoPy or NeuroM feature tables, no v661 s3 skeletons,
no auto-coregistration network, no synapse-compartment or tree-depth passes (those
columns ship as -1 and nothing reads them).

Where a rebuild will not match the Zenodo snapshot
--------------------------------------------------
Checked column by column against the shipped files: `connectome_neurons.csv` (all 46
columns, 1,351 rows), `connectome_synapses.csv` (all 17 columns, 146,544 rows), the
connectivity matrix and its mapping, and every subnetwork synapse, spine and
connectivity table all come out identical. Three known exceptions, none of which changes
a figure:

1. One cell — 864691136620192653, an inhibitory MC — carries `axon_length` and
   `dendrite_length` of -1 in the published tables: its skeleton had not been fetched
   yet when those tables were built. A rebuild downloads it like any other and gets real
   lengths (19,831 µm axon, 3,919 µm dendrite), so the cell now clears
   `filter_valid_neuron_w_spines`, which keeps 1,140 cells rather than the published
   1,139. Nothing else about it differs.
2. The per-subnetwork `connectome_neurons.csv` is the one table a rebuild does not
   reproduce, in two ways. It gains a column, `ds_outgoing_synapses_density`, which
   `calc_basic_degrree_features` started emitting after the shipped subnetworks were
   cut and every reader recomputes on load. And its cable lengths move: the shipped
   subnetworks were sliced from an older column table, so their `axon_length` /
   `dendrite_length` — and the seven density columns derived from them — sit a few
   percent below the values in the shipped `micro_column_network/connectome_neurons.csv`
   (medians ~2.6% on the axon, ~4.4% on the dendrite, on nearly every row). `subnets`
   slices from that shipped column table, so a rebuild is the self-consistent one.
   Nothing reads those columns: figures 5 and S13 name none of them, and the only place
   a cable length is touched is the `> 0` test in `filter_valid_neuron_w_spines`, which
   selects exactly the same cells in all four subnetworks either way.
3. Row order within `spine_table*.csv` follows directory listing order, not the
   snapshot's. Contents are identical.
"""

import argparse
import os
import pickle
import sys
import warnings

# The progress lines below use arrows and box-drawing characters; a Windows console on a
# non-UTF-8 code page raises UnicodeEncodeError on them mid-run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from caveclient import CAVEclient
from meshparty import trimesh_io

from connectivity_matrix_utils import save_connectivity
from connectome_types import ClfType, DATA_BASE_PATH, MATERIALIZATION_VERSION
from neuron import Neuron, read_local_swc
from sk_load_utils import filter_skeleton
from synapse import Synapse
from utils import (load_neurons_table, load_synapses_position_transformed,
                   process_str_position, transform_sk)

RAW_TABLES_DIR = os.path.join(DATA_BASE_PATH, 'raw_tables')

# Stands in for a synapse partner outside the network — the post-synaptic side of most
# outgoing synapses. Its fields are what `connectome_outgoing_synapses.csv` carries for
# those rows: clf_type 'both', cell_type 'Null', mtype 0.
MOCK_NEURON = Neuron(root_id=-1, clf_type=ClfType.both, cell_type='Null', mtype=0,
                     position=np.ones(0), volume=0, pre_synapses=[], post_synapses=[])


class Network:
    """One network folder under `data/`, and which type tables name its cells.

    `column_manual_ct` True  — the micro-column's hand-checked cell types.
                       False — the volume-wide model types, with the column's cells
                               overwritten by the hand-checked ones.
    """

    def __init__(self, name: str, column_manual_ct: bool):
        self.name = name
        self.column_manual_ct = column_manual_ct
        self.root = os.path.join(DATA_BASE_PATH, name)
        self.neurons = os.path.join(self.root, 'neurons')
        self.em_neurons = os.path.join(self.root, 'em_neurons')
        self.skeletons = os.path.join(self.root, 'skeletons')
        self.meshes = os.path.join(self.root, 'meshes')
        self.connectivity = os.path.join(self.root, 'connectivity_matrix')
        self.neuron_table = os.path.join(self.root, 'connectome_neurons.csv')
        self.syn_table = os.path.join(self.root, 'connectome_synapses.csv')
        self.out_syn_table = os.path.join(self.root, 'connectome_outgoing_synapses.csv')
        self.spine_table = os.path.join(self.root, 'spine_table.csv')
        self.spine_table_out = os.path.join(self.root, 'spine_table_outgoing.csv')


MICRO_COLUMN = Network('micro_column_network', column_manual_ct=True)
ALL_AXON_PR = Network('all_axon_pr_network', column_manual_ct=False)

# The cells whose meshes the figures draw. Figures 1, 2 and S1 render a dendrite branch
# off the full mesh; figure 7 panel A draws its six cells as meshes when CELLS_MODE says
# so. Root ids as listed in the paper's "Neuron identification in figures".
FIGURE_MESHES = [
    864691135277186789,   # figure 1B  L2/3 pyramidal   — also figure 7
    864691135875777166,   # figure 1B  L5 basket        — also figure 7
    864691136618553563,   # figure 2A  L4 pyramidal
    864691135214643256,   # figure 2A  L6 IT pyramidal
    864691135571356038,   # figure S1A L2/3 Martinotti
    864691135875414414,   # figure S1A L5 basket
    864691135503344477,   # figure 7   L4
    864691135324963484,   # figure 7   L5 PT
    864691135106189901,   # figure 7   basket, bottom left
    864691135503293250,   # figure 7   L6
]


# ─────────────────────────────────────────────────────────────────────────────
# raw tables
# ─────────────────────────────────────────────────────────────────────────────

RAW_TABLES = {
    'allen_v1_column_types_slanted_ref': 'the 1,351 column cells and their manual types — every figure',
    'allen_column_mtypes_v2':            'the column cells m-types',
    'aibs_metamodel_celltypes_v661':     'volume-wide cell types — all_axon_pr_network',
    'aibs_metamodel_mtypes_v661_v2':     'volume-wide m-types — all_axon_pr_network',
    'nucleus_detection_v0':              'root_id to nucleus_id',
    'proofreading_status_and_strategy':  'figure 4, and the proofread-axon cell list',
    'coregistration_manual_v4':          'input to scripts/extract_calcium_data_via_docker.ipynb',
}

NUCLEUS_TABLE = os.path.join(RAW_TABLES_DIR, 'nucleus_detection_v0.csv')
COLUMN_TYPES_TABLE = os.path.join(RAW_TABLES_DIR, 'allen_v1_column_types_slanted_ref.csv')
COLUMN_MTYPES_TABLE = os.path.join(RAW_TABLES_DIR, 'allen_column_mtypes_v2.csv')
PROOFREADING_TABLE = os.path.join(RAW_TABLES_DIR, 'proofreading_status_and_strategy.csv')


def cave_client() -> CAVEclient:
    client = CAVEclient('minnie65_public')
    client.materialize.version = MATERIALIZATION_VERSION
    print(f'CAVEclient minnie65_public, materialization v{client.materialize.version}')
    return client


def download_raw_tables(client):
    """The seven CAVE tables everything downstream is named against.

    All seven ship in the Zenodo snapshot too, so this step is only for the rebuild
    route. Two of them the figures read directly (the column types and the proofreading
    status) and `coregistration_manual_v4` is the input to
    scripts/extract_calcium_data_via_docker.ipynb; the rest are build inputs only.
    """
    os.makedirs(RAW_TABLES_DIR, exist_ok=True)
    for table in RAW_TABLES:
        path = os.path.join(RAW_TABLES_DIR, f'{table}.csv')
        if os.path.exists(path):
            continue
        print(f'  {table}')
        client.materialize.query_table(table).to_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# which cells, and what they are called
# ─────────────────────────────────────────────────────────────────────────────

def resolve_cell_types(root_ids, column_manual_ct: bool):
    """Return (ids we keep, cell-type table, m-type table).

    A cell survives only if it appears — unambiguously, i.e. exactly once — in both
    tables, and is neither a non-neuron nor coarse-unclear.
    """
    print(f'Resolving {len(root_ids)} cells:')
    if column_manual_ct:
        cell_types_df = pd.read_csv(COLUMN_TYPES_TABLE, index_col=0)
        mtypes_df = pd.read_csv(COLUMN_MTYPES_TABLE, index_col=0)
    else:
        cell_types_df = pd.read_csv(os.path.join(RAW_TABLES_DIR, 'aibs_metamodel_celltypes_v661.csv'), index_col=0)
        mtypes_df = pd.read_csv(os.path.join(RAW_TABLES_DIR, 'aibs_metamodel_mtypes_v661_v2.csv'), index_col=0)

        # The model labels the whole volume, but the column's cells were typed by hand —
        # prefer the hand labels wherever both tables have the cell.
        col_types = pd.read_csv(COLUMN_TYPES_TABLE, index_col=0).drop_duplicates('pt_root_id', keep='last')
        col_mtypes = pd.read_csv(COLUMN_MTYPES_TABLE, index_col=0).drop_duplicates('pt_root_id', keep='last')
        col_ids = set(col_types.pt_root_id) & set(col_mtypes.pt_root_id)
        col_types = col_types.set_index('pt_root_id')
        col_mtypes = col_mtypes.set_index('pt_root_id')

        mask = cell_types_df.pt_root_id.isin(col_ids)
        cell_types_df.loc[mask, 'cell_type'] = cell_types_df.loc[mask, 'pt_root_id'].map(col_types['cell_type'])
        cell_types_df.loc[mask, 'classification_system'] = \
            cell_types_df.loc[mask, 'pt_root_id'].map(col_types['classification_system'])

        mask = mtypes_df.pt_root_id.isin(col_ids)
        mtypes_df.loc[mask, 'cell_type'] = mtypes_df.loc[mask, 'pt_root_id'].map(col_mtypes['cell_type'])

    cell_types = cell_types_df[cell_types_df.classification_system != 'nonneuron'].copy()
    cell_types = cell_types[cell_types.classification_system != 'aibs_coarse_unclear'].copy()
    cell_types.drop_duplicates('pt_root_id', keep=False, inplace=True)
    cell_types.rename(columns={'target_id': 'nucleus_id'}, inplace=True)

    m_types = mtypes_df[mtypes_df.pt_root_id != 0].copy()
    m_types.rename(columns={'target_id': 'nucleus_id'}, inplace=True)

    ct_ids = set(cell_types[cell_types.pt_root_id.isin(root_ids)].pt_root_id)
    mt_ids = set(m_types[m_types.pt_root_id.isin(root_ids)].pt_root_id)
    combined = sorted(ct_ids & mt_ids)
    print(f'  cell types {len(ct_ids)} | m-types {len(mt_ids)} | keeping {len(combined)}')
    return combined, cell_types, m_types


def column_root_ids():
    return list(pd.read_csv(COLUMN_TYPES_TABLE, index_col=0).pt_root_id)


def proofread_axon_root_ids():
    df = pd.read_csv(PROOFREADING_TABLE, index_col=0)
    return list(df[df.status_axon == 't'].pt_root_id)


# ─────────────────────────────────────────────────────────────────────────────
# per-neuron downloads
# ─────────────────────────────────────────────────────────────────────────────

def _synapse_table_to_synapses(df: pd.DataFrame) -> list[Synapse]:
    ids = df['id'].to_numpy()
    pre_ids = df['pre_pt_root_id'].to_numpy()
    post_ids = df['post_pt_root_id'].to_numpy()
    sizes = df['size'].to_numpy()
    centers = df['ctr_pt_position'].apply(np.array).tolist()
    return [Synapse(id_=ids[i], pre_pt_root_id=pre_ids[i], post_pt_root_id=post_ids[i],
                    size=sizes[i], center_position=centers[i])
            for i in range(len(df))]


def download_neurons(net: Network, ids, cell_types, m_types, client):
    """One `Neuron` pickle per cell, carrying every synapse it has anywhere in the volume.

    Two CAVE queries per cell — incoming and outgoing — and by far the longest step.
    """
    os.makedirs(net.neurons, exist_ok=True)
    ct = cell_types.drop_duplicates('pt_root_id').set_index('pt_root_id')
    mt = m_types.drop_duplicates('pt_root_id').set_index('pt_root_id')

    todo = [i for i in ids if not os.path.exists(os.path.join(net.neurons, f'{i}.pkl'))]
    print(f'{net.name}: {len(ids) - len(todo)} neurons on disk, {len(todo)} to fetch')
    for cell_id in tqdm(todo, desc='neurons'):
        try:
            ct_row, mt_row = ct.loc[cell_id], mt.loc[cell_id]
            clf_type = (ClfType.excitatory if 'excitatory' in ct_row['classification_system']
                        else ClfType.inhibitory)
            neuron = Neuron(
                root_id=cell_id,
                clf_type=clf_type,
                cell_type=ct_row['cell_type'],
                mtype=mt_row['cell_type'],
                position=np.array(process_str_position(ct_row['pt_position'])),
                volume=ct_row['volume'],
                pre_synapses=_synapse_table_to_synapses(client.materialize.synapse_query(post_ids=cell_id)),
                post_synapses=_synapse_table_to_synapses(client.materialize.synapse_query(pre_ids=cell_id)),
            )
            with open(os.path.join(net.neurons, f'{cell_id}.pkl'), 'wb') as f:
                pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'  neuron {cell_id}: {e}')


def download_skeletons(net: Network, ids, client):
    """One SWC per cell, ten at a time — the bulk endpoint's limit."""
    os.makedirs(net.skeletons, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(net.skeletons, f'{i}.swc'))]
    print(f'{net.name}: {len(ids) - len(todo)} skeletons on disk, {len(todo)} to fetch')
    batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    for batch in tqdm(batches, desc='skeletons'):
        try:
            for root_id, swc_df in client.skeleton.get_bulk_skeletons(output_format='swc', root_ids=batch).items():
                swc_df.to_csv(os.path.join(net.skeletons, f'{root_id}.swc'),
                              sep=' ', header=False, index=False)
        except Exception as e:
            print(f'  skeleton batch {batch[0]}…: {e}')


def download_spines(net: Network, ids, client, incoming: bool):
    """Spine / shaft / soma tags per synapse, from `synapse_target_predictions_ssa_v2`.

    https://tutorial.microns-explorer.org/release_manifests/version-1718.html

    Queried one cell at a time into `neurons_spines_{incoming,outgoing}/`, then
    concatenated. Those per-cell CSVs are only a resume cache — delete them once the
    combined table exists.
    """
    raw_dir = os.path.join(net.root, f"neurons_spines_{'incoming' if incoming else 'outgoing'}")
    os.makedirs(raw_dir, exist_ok=True)

    todo = [i for i in ids if not os.path.exists(os.path.join(raw_dir, f'{i}.csv'))]
    print(f"{net.name}: {'incoming' if incoming else 'outgoing'} spines — "
          f'{len(ids) - len(todo)} on disk, {len(todo)} to fetch')
    for root_id in tqdm(todo, desc='spines'):
        try:
            table = client.materialize.tables.synapse_target_predictions_ssa_v2
            df = (table(post_pt_root_id=root_id) if incoming else table(pre_pt_root_id=root_id)).query()
            df[['target_id', 'tag', 'pre_pt_root_id', 'post_pt_root_id']].to_csv(
                os.path.join(raw_dir, f'{root_id}.csv'), index=False)
        except Exception as e:
            print(f'  spines {root_id}: {e}')

    print(f'  combining {len(ids)} cells')
    keep = set(ids)
    frames = []
    for fname in tqdm(os.listdir(raw_dir), desc='combine'):
        root_id = int(fname.replace('.csv', ''))
        if root_id not in keep:
            continue
        df = pd.read_csv(os.path.join(raw_dir, fname))
        # Stamped with the queried cell. For the incoming table that is the post-synaptic
        # cell and the column is unchanged; for the outgoing table it overwrites the real
        # post ids with the pre-synaptic one. The published tables carry it that way and
        # every reader of the outgoing table keys on `pre_pt_root_id`, so it is inert —
        # kept as-is so a rebuilt table matches the one on Zenodo.
        df['post_pt_root_id'] = root_id
        frames.append(df)

    out = net.spine_table if incoming else net.spine_table_out
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f'  wrote {out}')


def download_meshes(client):
    """The ten segmented meshes the figures draw, ~1.4 GB in total.

    Figures 1, 2 and S1 have theirs in the Zenodo snapshot. Figure 7 panel A renders its
    six cells as meshes only when `CELLS_MODE = 'meshes'`, and the snapshot carries only
    two of those six, which is why that notebook ships set to 'skeletons'. After this
    step all ten are present and the flag can be flipped.
    """
    os.makedirs(MICRO_COLUMN.meshes, exist_ok=True)
    mm = trimesh_io.MeshMeta(cv_path=client.info.segmentation_source(),
                             disk_cache_path=MICRO_COLUMN.meshes)
    for root_id in FIGURE_MESHES:
        if os.path.exists(os.path.join(MICRO_COLUMN.meshes, f'{root_id}.h5')):
            continue
        print(f'  mesh {root_id}')
        mm.mesh(seg_id=root_id)


# ─────────────────────────────────────────────────────────────────────────────
# derived: em_neurons
# ─────────────────────────────────────────────────────────────────────────────

def _load_skeleton(root_id, skeleton_dir):
    path = os.path.join(skeleton_dir, f'{root_id}.swc')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return read_local_swc(path)


def calculate_synapse_dist_to_soma(neuron: Neuron, skeleton_dir):
    """Path length along the skeleton from each synapse to the cell's own soma.

    Synapse positions are voxels (4×4×40 nm); the SWC is in µm, untransformed. Scaling
    the synapses by [4,4,40]/1000 puts both in the same space for the kd-tree query.
    """
    try:
        sk = _load_skeleton(neuron.root_id, skeleton_dir)
        if sk is None:
            return

        post_xyz = [syn.center_position * np.array([4, 4, 40]) / 1000 for syn in neuron.post_synapses]
        pre_xyz = [syn.center_position * np.array([4, 4, 40]) / 1000 for syn in neuron.pre_synapses]

        if post_xyz:
            _, nodes = sk.kdtree.query(post_xyz)
            for syn, node in zip(neuron.post_synapses, nodes):
                syn.dist_to_pre_syn_soma = sk.distance_to_root[node]

        if pre_xyz:
            _, nodes = sk.kdtree.query(pre_xyz)
            for syn, node in zip(neuron.pre_synapses, nodes):
                syn.dist_to_post_syn_soma = sk.distance_to_root[node]

    except Exception as e:
        print(f'  dist_to_soma failed for {neuron.root_id}: {e}')


def calculate_neuron_comp_lengths(neuron: Neuron, skeleton_dir):
    """Axonal and dendritic cable length, and how far the tallest dendrite reaches.

    Reads its own copy of the skeleton: this one is transformed in place, and the
    distance-to-soma pass above needs the untransformed one.
    """
    try:
        sk = _load_skeleton(neuron.root_id, skeleton_dir)
        if sk is None:
            return
        transform_sk(sk, s3=True)

        # Compartment 2 is axon, 3 and 4 dendrite, 1 soma — and the soma is deliberately in
        # neither. Including it (compartments [1,2] and [1,3,4]) adds the short edges from
        # the root out to each primary branch: ~0.2% on the axon, ~2.5% on the dendrite,
        # since a cell has one axon but many primary dendrites. The published tables were
        # built without it, and these two lines reproduce all 1,351 of their cable lengths
        # bit for bit, so leave them alone.
        neuron.axon_length = filter_skeleton(sk, target_compartments=[2]).path_length()
        neuron.dendrite_length = filter_skeleton(sk, target_compartments=[3, 4]).path_length()

        # y grows downward, so the tallest dendrite is the smallest y
        comp = np.asarray(sk.vertex_properties['compartment'])
        dendrite_idx = np.where(comp != 2)[0]
        dendrite_vertices = sk.vertices[dendrite_idx]
        top = np.argmin(dendrite_vertices[:, 1])

        neuron.max_dend_y_path_dist_to_soma = sk.distance_to_root[dendrite_idx[top]]
        neuron.max_dend_y_euclid_dist_to_soma = sk.vertices[sk.root][1] - dendrite_vertices[top][1]

    except Exception as e:
        print(f'  comp_lengths failed for {neuron.root_id}: {e}')


def build_em_neurons(net: Network):
    """`neurons/` → `em_neurons/`: the same cells, keeping only within-network synapses.

    The `ds_*` attributes are measured first, over the whole volume, because that is the
    only place the outside-the-network synapses are still there to count. Cells left
    with no internal synapse at all are dropped.
    """
    os.makedirs(net.em_neurons, exist_ok=True)
    network_ids = {int(f.split('.')[0]) for f in os.listdir(net.neurons)}
    nucleus_id = (pd.read_csv(NUCLEUS_TABLE, index_col=0)
                  .drop_duplicates('pt_root_id').set_index('pt_root_id')['id'])

    todo = [f for f in sorted(os.listdir(net.neurons))
            if not os.path.exists(os.path.join(net.em_neurons, f))]
    print(f'{net.name}: {len(network_ids) - len(todo)} em_neurons on disk, {len(todo)} to build')
    for filename in tqdm(todo, desc='em_neurons'):
        with open(os.path.join(net.neurons, filename), 'rb') as f:
            neuron: Neuron = pickle.load(f)

        pre_synapses = [syn for syn in neuron.pre_synapses if syn.pre_pt_root_id in network_ids]
        post_synapses = [syn for syn in neuron.post_synapses if syn.post_pt_root_id in network_ids]
        if not pre_synapses and not post_synapses:
            continue

        incoming = np.array([syn.size for syn in neuron.pre_synapses])
        outgoing = np.array([syn.size for syn in neuron.post_synapses])
        neuron.ds_num_of_incoming_synapses = len(neuron.pre_synapses)
        neuron.ds_num_of_outgoing_synapses = len(neuron.post_synapses)
        neuron.ds_incoming_syn_mean_weight = np.mean(incoming)
        neuron.ds_outgoing_syn_mean_weight = np.mean(outgoing)
        neuron.ds_incoming_syn_std_weight = np.std(incoming)
        neuron.ds_outgoing_syn_std_weight = np.std(outgoing)
        neuron.ds_incoming_syn_sum_weight = np.sum(incoming)
        neuron.ds_outgoing_syn_sum_weight = np.sum(outgoing)

        neuron.pre_synapses = pre_synapses
        neuron.post_synapses = post_synapses

        if neuron.root_id in nucleus_id.index:
            neuron.nucleus_id = int(nucleus_id[neuron.root_id])
        else:
            print(f'  missing nucleus_id for {neuron.root_id}')

        calculate_synapse_dist_to_soma(neuron, net.skeletons)
        calculate_neuron_comp_lengths(neuron, net.skeletons)

        with open(os.path.join(net.em_neurons, filename), 'wb') as f:
            pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_neurons_dict(neuron_path) -> dict[int, Neuron]:
    neurons = {}
    for filename in tqdm(os.listdir(neuron_path), desc=os.path.basename(neuron_path)):
        with open(os.path.join(neuron_path, filename), 'rb') as f:
            neuron: Neuron = pickle.load(f)
            neurons[neuron.root_id] = neuron
    return neurons


# ─────────────────────────────────────────────────────────────────────────────
# derived: the CSV tables
# ─────────────────────────────────────────────────────────────────────────────

def _weights(synapses) -> tuple:
    """(mean, sum, std) of synapse size. Empty gives (nan, 0.0, nan), which is what the
    published tables carry for cells with no inhibitory input, no excitatory output, …"""
    w = np.array([syn.size for syn in synapses])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # mean/std of an empty array
        return np.mean(w), np.sum(w), np.std(w)


def neuron_table(neurons_dict: dict[int, Neuron]) -> pd.DataFrame:
    """`connectome_neurons.csv` — one row per cell.

    `ds_` columns are the whole volume; the unprefixed ones count only synapses whose
    both endpoints are in this network.
    """
    neurons = list(neurons_dict.values())
    ex, inh = ClfType.excitatory, ClfType.inhibitory

    rows = []
    for n in tqdm(neurons, desc='neuron table'):
        ex_pre = [s for s in n.pre_synapses if neurons_dict[s.pre_pt_root_id].clf_type == ex]
        inh_pre = [s for s in n.pre_synapses if neurons_dict[s.pre_pt_root_id].clf_type == inh]
        ex_post = [s for s in n.post_synapses if neurons_dict[s.post_pt_root_id].clf_type == ex]
        inh_post = [s for s in n.post_synapses if neurons_dict[s.post_pt_root_id].clf_type == inh]

        pre_mean, pre_sum, pre_std = _weights(n.pre_synapses)
        post_mean, post_sum, post_std = _weights(n.post_synapses)
        ex_pre_mean, ex_pre_sum, ex_pre_std = _weights(ex_pre)
        ex_post_mean, ex_post_sum, ex_post_std = _weights(ex_post)
        inh_pre_mean, inh_pre_sum, inh_pre_std = _weights(inh_pre)
        inh_post_mean, inh_post_sum, inh_post_std = _weights(inh_post)

        rows.append({
            'root_id': n.root_id, 'nucleus_id': n.nucleus_id, 'volume': n.volume,
            'clf_type': n.clf_type, 'cell_type': n.cell_type, 'mtype': n.mtype,
            'axon_length': n.axon_length, 'dendrite_length': n.dendrite_length,
            'max_dend_y_path_dist_to_soma': n.max_dend_y_path_dist_to_soma,
            'max_dend_y_euclid_dist_to_soma': n.max_dend_y_euclid_dist_to_soma,
            'ds_num_of_incoming_synapses': n.ds_num_of_incoming_synapses,
            'ds_num_of_outgoing_synapses': n.ds_num_of_outgoing_synapses,
            'ds_incoming_syn_mean_weight': n.ds_incoming_syn_mean_weight,
            'ds_outgoing_syn_mean_weight': n.ds_outgoing_syn_mean_weight,
            'ds_incoming_syn_sum_weight': n.ds_incoming_syn_sum_weight,
            'ds_outgoing_syn_sum_weight': n.ds_outgoing_syn_sum_weight,
            'ds_incoming_syn_std_weight': n.ds_incoming_syn_std_weight,
            'ds_outgoing_syn_std_weight': n.ds_outgoing_syn_std_weight,
            'num_of_incoming_synapses': len(n.pre_synapses),
            'num_of_outgoing_synapses': len(n.post_synapses),
            'num_of_incoming_neurons': len({s.pre_pt_root_id for s in n.pre_synapses}),
            'num_of_outgoing_neurons': len({s.post_pt_root_id for s in n.post_synapses}),
            'num_of_ex_incoming_synapses': len(ex_pre),
            'num_of_ex_outgoing_synapses': len(ex_post),
            'num_of_inh_incoming_synapses': len(inh_pre),
            'num_of_inh_outgoing_synapses': len(inh_post),
            'num_of_ex_incoming_neurons': len({s.pre_pt_root_id for s in ex_pre}),
            'num_of_inh_incoming_neurons': len({s.pre_pt_root_id for s in inh_pre}),
            'incoming_syn_mean_weight': pre_mean,
            'incoming_syn_sum_weight': pre_sum,
            'incoming_syn_std_weight': pre_std,
            'outgoing_syn_mean_weight': post_mean,
            'outgoing_syn_sum_weight': post_sum,
            'outgoing_syn_std_weight': post_std,
            'ex_incoming_syn_mean_weight': ex_pre_mean,
            'ex_incoming_syn_sum_weight': ex_pre_sum,
            'ex_incoming_syn_std_weight': ex_pre_std,
            'ex_outgoing_syn_mean_weight': ex_post_mean,
            'ex_outgoing_syn_sum_weight': ex_post_sum,
            'ex_outgoing_syn_std_weight': ex_post_std,
            'inh_incoming_syn_mean_weight': inh_pre_mean,
            'inh_incoming_syn_sum_weight': inh_pre_sum,
            'inh_incoming_syn_std_weight': inh_pre_std,
            'inh_outgoing_syn_mean_weight': inh_post_mean,
            'inh_outgoing_syn_sum_weight': inh_post_sum,
            'inh_outgoing_syn_std_weight': inh_post_std,
        })
    return pd.DataFrame(rows)


def synapse_table(synapses: list[Synapse], neurons_dict: dict[int, Neuron]) -> pd.DataFrame:
    """`connectome_synapses.csv` / `connectome_outgoing_synapses.csv` — one row per synapse.

    Partners outside the network fall back to `MOCK_NEURON`. `synapses_depth_*` and
    `compartment_*` are -1 throughout: the passes that would fill them are not part of
    this pipeline and no figure reads them. The columns stay so the table matches the
    published one.
    """
    rows = []
    for syn in tqdm(synapses, desc='synapse table'):
        pre = neurons_dict.get(syn.pre_pt_root_id, MOCK_NEURON)
        post = neurons_dict.get(syn.post_pt_root_id, MOCK_NEURON)
        rows.append({
            'id_': syn.id_, 'pre_id': syn.pre_pt_root_id, 'post_id': syn.post_pt_root_id,
            'dist_to_post_syn_soma': syn.dist_to_post_syn_soma,
            'dist_to_pre_syn_soma': syn.dist_to_pre_syn_soma,
            'size': syn.size, 'center_position': syn.center_position,
            'synapses_depth_in_post': syn.depth_in_post_syn_tree,
            'synapses_depth_in_pre': syn.depth_in_pre_syn_tree,
            'compartment_in_post_syn_soma': syn.compartment_in_post_syn_soma,
            'compartment_in_pre_syn_soma': syn.compartment_in_pre_syn_soma,
            'pre_clf_type': pre.clf_type, 'pre_cell_type': pre.cell_type, 'pre_m_type': pre.mtype,
            'post_clf_type': post.clf_type, 'post_cell_type': post.cell_type,
            'post_mtype_type': post.mtype,
        })
    return pd.DataFrame(rows)


def build_connectome_tables(net: Network):
    """`connectome_neurons.csv` and `connectome_synapses.csv`, from `em_neurons/`."""
    neurons = load_neurons_dict(net.em_neurons)

    pre_synapses, post_synapses = [], []
    for n in tqdm(neurons.values(), desc='collect'):
        pre_synapses.extend(n.pre_synapses)
        post_synapses.extend(n.post_synapses)

    # Each internal synapse is stored twice — once on each endpoint — and each copy only
    # ever learned its distance to *that* cell's soma. Merge the pre-side measurement
    # into the post-side copy, which is the one that becomes a row.
    by_id = {s.id_: s for s in post_synapses}
    for syn in tqdm(pre_synapses, desc='merge'):
        twin = by_id[syn.id_]
        syn.dist_to_pre_syn_soma = twin.dist_to_pre_syn_soma
        syn.compartment_in_pre_syn_soma = twin.compartment_in_pre_syn_soma

    print(f'  {len(neurons)} neurons, {len(pre_synapses)} synapses')
    neuron_table(neurons).to_csv(net.neuron_table)
    synapse_table(pre_synapses, neurons).to_csv(net.syn_table)
    print(f'  wrote {net.neuron_table}\n  wrote {net.syn_table}')


def build_outgoing_synapse_table(net: Network):
    """`connectome_outgoing_synapses.csv` — every synapse these cells *make*, including
    the ones landing outside the network. The peak-memory step."""
    neurons = load_neurons_dict(net.neurons)     # full-volume scope, not em_neurons

    synapses = []
    for neuron in tqdm(neurons.values(), desc='dist to soma'):
        calculate_synapse_dist_to_soma(neuron, net.skeletons)
        synapses.extend(neuron.post_synapses)

    print(f'  {len(synapses)} outgoing synapses')
    synapse_table(synapses, neurons).to_csv(net.out_syn_table)
    print(f'  wrote {net.out_syn_table}')


def build_connectivity_matrix(net: Network):
    """`connectivity_matrix/` — synapse counts as a sparse matrix plus its row mapping.

    Rows are post-synaptic, columns pre-synaptic, and the row order is the row order of
    `connectome_neurons.csv`. `load_bin_mat` transposes on read, which is where the
    documented `[pre, post]` convention comes from.
    """
    os.makedirs(net.connectivity, exist_ok=True)
    order = list(pd.read_csv(net.neuron_table, index_col=0).root_id)
    index = {root_id: i for i, root_id in enumerate(order)}

    syn_df = pd.read_csv(net.syn_table, usecols=['pre_id', 'post_id'])
    matrix = np.zeros((len(order), len(order)), dtype=int)
    for pre_id, post_id in tqdm(zip(syn_df.pre_id, syn_df.post_id), total=len(syn_df), desc='matrix'):
        if pre_id in index and post_id in index:
            matrix[index[post_id], index[pre_id]] += 1

    mapping = {i: root_id for i, root_id in enumerate(order)}
    save_connectivity(sparse.csr_matrix(matrix), mapping, net.connectivity, 'network_synapses')
    print(f'  {len(order)} neurons, {matrix.sum()} synapses, '
          f'{np.count_nonzero(matrix) / matrix.size * 100:.2f}% of pairs connected')


# ─────────────────────────────────────────────────────────────────────────────
# derived: subnetworks (figure 5)
# ─────────────────────────────────────────────────────────────────────────────

def build_subnetworks():
    """Four nested slabs of the column, each half the width of the last.

    Figure 5 asks whether the spine-targeting effects survive when the network is
    cropped, so each subnetwork is the same tables restricted to the cells inside a
    narrowing x-band. No network access — it is all sliced from the column's own CSVs.

    The neuron table written here is the one `load_neurons_table` returns: the connectome
    columns plus transformed soma positions and the basic degree features. That is one
    column wider than the shipped snapshot (see the module docstring) and every reader
    recomputes those features anyway.
    """
    from connectome_types import (CONNECTOME_PRE_SYN_TABLE_PATH, NETWORK_NAME,
                                  SPINE_TABLE, SPINE_TABLE_OUTGOING)

    neurons_df = load_neurons_table(use_column_manual_ct=True)
    syn_df = load_synapses_position_transformed()
    spine_df = pd.read_csv(SPINE_TABLE)
    outgoing_syn_df = load_synapses_position_transformed(base_syn_table_path=CONNECTOME_PRE_SYN_TABLE_PATH)
    spine_df_outgoing = pd.read_csv(SPINE_TABLE_OUTGOING)

    x = neurons_df.pt_position_xt
    max_x, min_x = max(x), min(x)
    dx = max_x - min_x
    fraction = 1.0

    while dx > 10:
        max_x = max_x - dx // 4
        min_x = min_x + dx // 4
        dx = dx // 2
        fraction /= 2

        neurons = neurons_df[(neurons_df.pt_position_xt <= max_x) &
                             (neurons_df.pt_position_xt >= min_x)].copy()
        sub_syn = syn_df[syn_df.pre_id.isin(neurons.root_id) &
                         syn_df.post_id.isin(neurons.root_id)].copy()
        # every incoming spine, as in the full network — what calc_spines_features expects
        sub_spine = spine_df[spine_df.post_pt_root_id.isin(neurons.root_id)].copy()
        sub_syn_out = outgoing_syn_df[outgoing_syn_df.pre_id.isin(neurons.root_id)].copy()
        sub_spine_out = spine_df_outgoing[spine_df_outgoing.pre_pt_root_id.isin(neurons.root_id)].copy()

        folder = os.path.join(DATA_BASE_PATH, NETWORK_NAME, 'subnetworks', str(fraction))
        os.makedirs(folder, exist_ok=True)
        print(f'  {fraction}: {len(neurons)} neurons, {len(sub_syn)} synapses, '
              f'{len(sub_spine)} incoming spines')

        neurons.to_csv(os.path.join(folder, 'connectome_neurons.csv'))
        sub_syn.to_csv(os.path.join(folder, 'connectome_synapses.csv'))
        sub_spine.to_csv(os.path.join(folder, 'spine_table.csv'))
        sub_syn_out.to_csv(os.path.join(folder, 'connectome_outgoing_synapses.csv'))
        sub_spine_out.to_csv(os.path.join(folder, 'spine_table_outgoing.csv'))

        # Row order here is first appearance in the synapse table, not the neuron table —
        # a subnetwork's matrix only ever covers the cells that have an internal synapse.
        ordered = [int(r) for r in pd.unique(pd.concat([sub_syn.post_id, sub_syn.pre_id],
                                                       ignore_index=True)) if pd.notna(r)]
        index = {root_id: i for i, root_id in enumerate(ordered)}
        matrix = np.zeros((len(ordered), len(ordered)), dtype=int)
        for pre_id, post_id in zip(sub_syn.pre_id, sub_syn.post_id):
            matrix[index[int(post_id)], index[int(pre_id)]] += 1

        conn_dir = os.path.join(folder, 'connectivity_matrix')
        os.makedirs(conn_dir, exist_ok=True)
        save_connectivity(sparse.csr_matrix(matrix),
                          {i: root_id for i, root_id in enumerate(ordered)},
                          conn_dir, 'network_synapses')


# ─────────────────────────────────────────────────────────────────────────────
# steps
# ─────────────────────────────────────────────────────────────────────────────

def build_full_network(net: Network, root_ids, client):
    """Everything for one network folder: download, then derive."""
    ids, cell_types, m_types = resolve_cell_types(root_ids, net.column_manual_ct)
    download_neurons(net, ids, cell_types, m_types, client)
    download_skeletons(net, ids, client)
    download_spines(net, ids, client, incoming=True)
    download_spines(net, ids, client, incoming=False)
    build_em_neurons(net)
    build_connectome_tables(net)
    build_outgoing_synapse_table(net)
    build_connectivity_matrix(net)


def step_raw(client):
    download_raw_tables(client)


def step_column(client):
    build_full_network(MICRO_COLUMN, column_root_ids(), client)


def step_subnets(client):
    build_subnetworks()


def step_meshes(client):
    download_meshes(client)


def step_axon_pr(client):
    build_full_network(ALL_AXON_PR, proofread_axon_root_ids(), client)


STEPS = {
    'raw': step_raw,
    'column': step_column,
    'subnets': step_subnets,
    'meshes': step_meshes,
    'axon_pr': step_axon_pr,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default=','.join(STEPS),
                        help=f"comma-separated subset of: {', '.join(STEPS)} (default: all, in order)")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}. Pick from: {', '.join(STEPS)}")

    client = cave_client()
    for name in steps:
        print(f'\n=== {name} ===')
        STEPS[name](client)

    print('\nDone. data/activity/ is still yours to build — see the README.')


if __name__ == '__main__':
    main()
