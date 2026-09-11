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

Interruptions are expected — a day-long run meets dropped connections, Ctrl+C and the
occasional reboot. None of them costs more than the file being written at that moment:
every write lands in a `.part` sibling and is renamed into place only once it is
complete, so a file that exists is a file that is finished, and every step works out
what is left to do by looking at what is on disk. The per-cell loops fetch only the
cells they are missing; the derived tables rebuild only when something they read has
changed. Re-run the same command and it continues where it stopped. Transient CAVE
errors are retried before a cell is given up on, and a run that ends with cells still
missing says so and refuses to build the network-wide tables from a partial set
(`--allow-incomplete` overrides that).

Budget roughly a day of wall clock and ~25 GB. The two per-neuron CAVE loops
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
  meshes    data/micro_column_network/meshes/           ~figures 1, 2, 7, S1
  axon_pr   data/all_axon_pr_network/                   ~figure S3 only

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
import shutil
import sys
import time
import warnings
from contextlib import contextmanager

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


class BuildError(Exception):
    """Something on disk is not what the run needs. `main` prints it without a traceback."""


# ─────────────────────────────────────────────────────────────────────────────
# surviving an interrupted run
# ─────────────────────────────────────────────────────────────────────────────
#
# Every step below works out what is left to do by looking at what is on disk, which is
# only sound if a file on disk is a finished file. So nothing is written in place: each
# write goes to a `.part` sibling and is renamed over the target once it is whole, and a
# rename within a directory is atomic. Kill the run at any moment and what survives is
# exactly the set of files that were already complete.

PARTIAL_SUFFIX = '.part'

# Files an earlier run left unreadable, removed as they were met. Reported by `main`.
DISCARDED = []


@contextmanager
def atomic(path):
    """Yield a staging path to write to, then move it onto `path` in one step."""
    staging = path + PARTIAL_SUFFIX
    try:
        yield staging
        os.replace(staging, path)
    except BaseException:
        # BaseException, not Exception: a Ctrl+C must not leave the stub behind either.
        if os.path.exists(staging):
            os.remove(staging)
        raise


def complete(path) -> bool:
    """Whether `path` is a finished file. Empty means a write that never got going —
    only possible for files left by a run that predates the staging above."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def clear_partials(*paths):
    """Drop staging files a hard kill (a reboot, a power cut) left behind."""
    for directory in paths:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.endswith(PARTIAL_SUFFIX):
                continue
            leftover = os.path.join(directory, name)
            if os.path.isdir(leftover):
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                os.remove(leftover)


def listdir(directory, suffix) -> list[str]:
    """Sorted names ending in `suffix` — anything mid-write is filtered out by name."""
    if not os.path.isdir(directory):
        return []
    return sorted(name for name in os.listdir(directory) if name.endswith(suffix))


def retry(call, what: str, attempts: int = 4, wait: float = 5.0):
    """Run `call`, retrying the transient CAVE failures — a timeout, a 502, a reset.

    One query per neuron for a day is long enough to meet a few. Without this each one
    costs that cell an entire further pass over the list.
    """
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt == attempts:
                raise
            tqdm.write(f'  {what}: {type(e).__name__}: {e} — retrying in {wait:.0f}s '
                       f'({attempt}/{attempts - 1})')
            time.sleep(wait)
            wait *= 2


def discard_unreadable(path, error):
    """Delete a file that will not load, so a later run fetches or rebuilds it."""
    tqdm.write(f'  {os.path.basename(path)} will not load '
               f'({type(error).__name__}: {error}) — removing it')
    DISCARDED.append(path)
    try:
        os.remove(path)
    except OSError:
        pass


def load_pickle(path):
    """The unpickled object, or None if the file is damaged — in which case it is gone."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except MemoryError:
        raise                       # the machine ran out, not a bad file — do not delete
    except KeyboardInterrupt:
        raise
    except Exception as e:
        discard_unreadable(path, e)
        return None


def _newest_mtime(inputs) -> float:
    """The most recent mtime across `inputs`; a directory counts as the files in it.

    The directory's own mtime counts too, and is the only thing that moves when a file
    is *removed* from it — which is how a cell dropped for being damaged comes back
    around as a rebuild of everything that was derived from it.
    """
    newest = 0.0
    for path in inputs:
        if os.path.isdir(path):
            newest = max(newest, os.path.getmtime(path))
            for entry in os.scandir(path):
                if entry.is_file() and not entry.name.endswith(PARTIAL_SUFFIX):
                    newest = max(newest, entry.stat().st_mtime)
        elif os.path.exists(path):
            newest = max(newest, os.path.getmtime(path))
    return newest


def up_to_date(outputs, inputs) -> bool:
    """Whether every output exists and none of the inputs has moved since.

    Existence alone would be the wrong test: a run interrupted part-way through the
    per-cell downloads and then resumed has more cells than the tables a previous run
    wrote, and those tables have to be built again.
    """
    if not all(complete(path) for path in outputs):
        return False
    return min(os.path.getmtime(path) for path in outputs) >= _newest_mtime(inputs)


def already_built(label: str, outputs, inputs) -> bool:
    if not up_to_date(outputs, inputs):
        return False
    print(f'  {label}: already built, and newer than everything it reads — skipping')
    return True


def save_connectivity_atomic(matrix, mapping, directory):
    """`save_connectivity` writes two files; stage both so a kill leaves neither."""
    staging = os.path.join(directory, 'staging' + PARTIAL_SUFFIX)
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        save_connectivity(matrix, mapping, staging, 'network_synapses')
        for path in connectivity_files(staging):
            os.replace(path, os.path.join(directory, os.path.basename(path)))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def connectivity_files(directory) -> list[str]:
    """What `save_connectivity(..., 'network_synapses')` leaves in `directory`."""
    return [os.path.join(directory, 'network_synapses_matrix.npz'),
            os.path.join(directory, 'network_synapses_mapping.pkl')]


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
    clear_partials(RAW_TABLES_DIR)
    for table in RAW_TABLES:
        path = os.path.join(RAW_TABLES_DIR, f'{table}.csv')
        if complete(path):
            continue
        print(f'  {table}')
        df = retry(lambda: client.materialize.query_table(table), f'table {table}')
        with atomic(path) as staging:
            df.to_csv(staging)


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
    ids = list(pd.read_csv(COLUMN_TYPES_TABLE, index_col=0).pt_root_id)
    if not ids:
        raise BuildError(f'{COLUMN_TYPES_TABLE} holds no cells. Delete it and re-run '
                         f'`--steps raw` to fetch it again.')
    return ids


def _flag(series: pd.Series) -> pd.Series:
    """A CAVE boolean column as a mask.

    Postgres serves these as 't'/'f' and older CAVEclients hand them straight through,
    but a newer one casts to real booleans, which `to_csv` then writes as True/False.
    Testing `== 't'` matches nothing on a table saved by the second kind — silently, so
    the step it feeds resolves zero cells and everything after it is empty.
    """
    return series.astype(str).str.strip().str.lower().isin({'t', 'true', '1', 'yes'})


def proofread_axon_root_ids():
    df = pd.read_csv(PROOFREADING_TABLE, index_col=0)
    ids = list(df[_flag(df.status_axon)].pt_root_id)
    if not ids:
        values = sorted(set(df.status_axon.astype(str)))[:6]
        raise BuildError(
            f'No cell in {PROOFREADING_TABLE} has a proofread axon — status_axon holds '
            f'{values}, none of which reads as true. The axon_pr step has nothing to '
            f'build from; check that table, or skip the step with --steps.')
    return ids


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
    clear_partials(net.neurons)
    ct = cell_types.drop_duplicates('pt_root_id').set_index('pt_root_id')
    mt = m_types.drop_duplicates('pt_root_id').set_index('pt_root_id')

    todo = [i for i in ids if not complete(os.path.join(net.neurons, f'{i}.pkl'))]
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
                pre_synapses=_synapse_table_to_synapses(retry(
                    lambda: client.materialize.synapse_query(post_ids=cell_id),
                    f'neuron {cell_id} incoming')),
                post_synapses=_synapse_table_to_synapses(retry(
                    lambda: client.materialize.synapse_query(pre_ids=cell_id),
                    f'neuron {cell_id} outgoing')),
            )
            with atomic(os.path.join(net.neurons, f'{cell_id}.pkl')) as staging:
                with open(staging, 'wb') as f:
                    pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            tqdm.write(f'  neuron {cell_id}: {e}')


def download_skeletons(net: Network, ids, client):
    """One SWC per cell, ten at a time — the bulk endpoint's limit."""
    os.makedirs(net.skeletons, exist_ok=True)
    clear_partials(net.skeletons)
    todo = [i for i in ids if not complete(os.path.join(net.skeletons, f'{i}.swc'))]
    print(f'{net.name}: {len(ids) - len(todo)} skeletons on disk, {len(todo)} to fetch')
    batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    for batch in tqdm(batches, desc='skeletons'):
        try:
            skeletons = retry(
                lambda: client.skeleton.get_bulk_skeletons(output_format='swc', root_ids=batch),
                f'skeleton batch {batch[0]}…')
            for root_id, swc_df in skeletons.items():
                with atomic(os.path.join(net.skeletons, f'{root_id}.swc')) as staging:
                    swc_df.to_csv(staging, sep=' ', header=False, index=False)
        except Exception as e:
            tqdm.write(f'  skeleton batch {batch[0]}…: {e}')


def download_spines(net: Network, ids, client, incoming: bool, allow_incomplete=False):
    """Spine / shaft / soma tags per synapse, from `synapse_target_predictions_ssa_v2`.

    https://tutorial.microns-explorer.org/release_manifests/version-1718.html

    Queried one cell at a time into `neurons_spines_{incoming,outgoing}/`, then
    concatenated. Those per-cell CSVs are only a resume cache — delete them once the
    combined table exists.
    """
    direction = 'incoming' if incoming else 'outgoing'
    raw_dir = os.path.join(net.root, f'neurons_spines_{direction}')
    out = net.spine_table if incoming else net.spine_table_out

    # Before the loop, not after it: the per-cell cache is disposable and documented as
    # such, so the combined table standing newer than it means the step is done. Testing
    # this afterwards would re-query every cell to rebuild a table that is already there.
    if already_built(f'{direction} spine table', [out], [raw_dir]):
        return

    os.makedirs(raw_dir, exist_ok=True)
    clear_partials(raw_dir)

    todo = [i for i in ids if not complete(os.path.join(raw_dir, f'{i}.csv'))]
    print(f'{net.name}: {direction} spines — '
          f'{len(ids) - len(todo)} on disk, {len(todo)} to fetch')
    for root_id in tqdm(todo, desc='spines'):
        try:
            table = client.materialize.tables.synapse_target_predictions_ssa_v2
            df = retry(lambda: (table(post_pt_root_id=root_id) if incoming
                                else table(pre_pt_root_id=root_id)).query(),
                       f'spines {root_id}')
            with atomic(os.path.join(raw_dir, f'{root_id}.csv')) as staging:
                df[['target_id', 'tag', 'pre_pt_root_id', 'post_pt_root_id']].to_csv(
                    staging, index=False)
        except Exception as e:
            tqdm.write(f'  spines {root_id}: {e}')

    # The combined table is one table over all the cells, so a cell still missing its
    # tags is a hole in it — and the usual reason for one is a connection that dropped,
    # which the next run simply re-fetches. Leave it unwritten rather than wrong.
    missing = [i for i in ids if not complete(os.path.join(raw_dir, f'{i}.csv'))]
    if missing and not allow_incomplete:
        print(f'  {len(missing)} of {len(ids)} cells have no {direction} spine tags yet — '
              f'not writing {os.path.basename(out)} from a partial set. Re-run to fetch '
              f'them, or pass --allow-incomplete.')
        return

    print(f'  combining {len(ids)} cells')
    keep = set(ids)
    frames = []
    for fname in tqdm(listdir(raw_dir, '.csv'), desc='combine'):
        root_id = int(fname[:-len('.csv')])
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

    if not frames:
        print(f'  no {direction} spine tags on disk — nothing to combine, '
              f'leaving {os.path.basename(out)} alone')
        return

    with atomic(out) as staging:
        pd.concat(frames, ignore_index=True).to_csv(staging, index=False)
    print(f'  wrote {out}')


def download_meshes(client):
    """The ten segmented meshes the figures draw.

    Figures 1, 2 and S1 have theirs in the Zenodo snapshot. Figure 7 panel A renders its
    six cells as meshes only when `CELLS_MODE = 'meshes'`, and the snapshot carries only
    two of those six, which is why that notebook ships set to 'skeletons'. After this
    step all ten are present and the flag can be flipped.
    """
    os.makedirs(MICRO_COLUMN.meshes, exist_ok=True)
    clear_partials(MICRO_COLUMN.meshes)

    # meshparty writes into its own disk cache, so the cache is a staging directory and
    # the finished file is moved out of it. That also settles the name: meshparty 2.0.3
    # caches a mesh as `<id>_<lod>.h5` (`mesh()` defaults to lod=0), while `load_full_mesh`
    # — and the existence check below — want a plain `<id>.h5`, so whatever the fetch
    # leaves behind is renamed to that on the way out.
    staging = os.path.join(MICRO_COLUMN.meshes, 'staging' + PARTIAL_SUFFIX)
    os.makedirs(staging, exist_ok=True)
    mm = trimesh_io.MeshMeta(cv_path=client.info.segmentation_source(),
                             disk_cache_path=staging)
    try:
        for root_id in FIGURE_MESHES:
            final = os.path.join(MICRO_COLUMN.meshes, f'{root_id}.h5')
            if complete(final):
                continue
            print(f'  mesh {root_id}')
            retry(lambda: mm.mesh(seg_id=root_id), f'mesh {root_id}')

            fetched = [n for n in listdir(staging, '.h5') if n.startswith(str(root_id))]
            if not fetched:
                print(f'  mesh {root_id}: the fetch left no file behind — skipping it')
                continue
            os.replace(os.path.join(staging, fetched[0]), final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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


def build_em_neurons(net: Network) -> bool:
    """`neurons/` → `em_neurons/`: the same cells, keeping only within-network synapses.

    The `ds_*` attributes are measured first, over the whole volume, because that is the
    only place the outside-the-network synapses are still there to count. Cells left
    with no internal synapse at all are dropped.

    False if a cell was lost to a damaged file, which makes every table below it wrong.
    """
    damaged = len(DISCARDED)
    os.makedirs(net.em_neurons, exist_ok=True)
    clear_partials(net.em_neurons)
    network_ids = {int(f[:-len('.pkl')]) for f in listdir(net.neurons, '.pkl')}
    nucleus_id = (pd.read_csv(NUCLEUS_TABLE, index_col=0)
                  .drop_duplicates('pt_root_id').set_index('pt_root_id')['id'])

    # Each em_neuron keeps only the synapses whose partner is also in the network, so it
    # is derived from the whole of `neurons/`, not from its own cell. Once that set
    # changes — a cell refetched after a failure, a damaged one dropped — every em_neuron
    # is out of date, not just the new one, and the ones already rebuilt stay put.
    newest_neuron = _newest_mtime([net.neurons])

    def stale(filename):
        path = os.path.join(net.em_neurons, filename)
        return not complete(path) or os.path.getmtime(path) < newest_neuron

    todo = [f for f in listdir(net.neurons, '.pkl') if stale(f)]
    print(f'{net.name}: {len(network_ids) - len(todo)} em_neurons on disk, {len(todo)} to build')
    for filename in tqdm(todo, desc='em_neurons'):
        neuron: Neuron = load_pickle(os.path.join(net.neurons, filename))
        if neuron is None:
            continue

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
            tqdm.write(f'  missing nucleus_id for {neuron.root_id}')

        calculate_synapse_dist_to_soma(neuron, net.skeletons)
        calculate_neuron_comp_lengths(neuron, net.skeletons)

        with atomic(os.path.join(net.em_neurons, filename)) as staging:
            with open(staging, 'wb') as f:
                pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)

    if len(DISCARDED) > damaged:
        print(f'  {len(DISCARDED) - damaged} cell(s) were lost to a damaged file. Every '
              f'table below this reads all the cells at once, so none is written from '
              f'what is left — re-run to fetch those cells again.')
        return False
    return True


def load_neurons_dict(neuron_path) -> dict[int, Neuron]:
    """Every neuron in a folder. Returns None if any of them turned out to be damaged:
    the tables built from this are network-wide, and a dropped cell is a wrong table."""
    damaged = len(DISCARDED)
    neurons = {}
    for filename in tqdm(listdir(neuron_path, '.pkl'), desc=os.path.basename(neuron_path)):
        neuron: Neuron = load_pickle(os.path.join(neuron_path, filename))
        if neuron is not None:
            neurons[neuron.root_id] = neuron

    if len(DISCARDED) > damaged:
        print(f'  {len(DISCARDED) - damaged} neuron file(s) in {os.path.basename(neuron_path)} '
              f'were damaged and have been removed — re-run to fetch and rebuild them')
        return None
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
    if already_built('connectome tables', [net.neuron_table, net.syn_table], [net.em_neurons]):
        return

    neurons = load_neurons_dict(net.em_neurons)
    if neurons is None:
        return

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
    with atomic(net.neuron_table) as staging:
        neuron_table(neurons).to_csv(staging)
    with atomic(net.syn_table) as staging:
        synapse_table(pre_synapses, neurons).to_csv(staging)
    print(f'  wrote {net.neuron_table}\n  wrote {net.syn_table}')


def build_outgoing_synapse_table(net: Network):
    """`connectome_outgoing_synapses.csv` — every synapse these cells *make*, including
    the ones landing outside the network. The peak-memory step."""
    if already_built('outgoing synapse table', [net.out_syn_table],
                     [net.neurons, net.skeletons]):
        return

    neurons = load_neurons_dict(net.neurons)     # full-volume scope, not em_neurons
    if neurons is None:
        return

    synapses = []
    for neuron in tqdm(neurons.values(), desc='dist to soma'):
        calculate_synapse_dist_to_soma(neuron, net.skeletons)
        synapses.extend(neuron.post_synapses)

    print(f'  {len(synapses)} outgoing synapses')
    with atomic(net.out_syn_table) as staging:
        synapse_table(synapses, neurons).to_csv(staging)
    print(f'  wrote {net.out_syn_table}')


def build_connectivity_matrix(net: Network):
    """`connectivity_matrix/` — synapse counts as a sparse matrix plus its row mapping.

    Rows are post-synaptic, columns pre-synaptic, and the row order is the row order of
    `connectome_neurons.csv`. `load_bin_mat` transposes on read, which is where the
    documented `[pre, post]` convention comes from.
    """
    os.makedirs(net.connectivity, exist_ok=True)
    clear_partials(net.connectivity)
    if already_built('connectivity matrix', connectivity_files(net.connectivity),
                     [net.neuron_table, net.syn_table]):
        return

    order = list(pd.read_csv(net.neuron_table, index_col=0).root_id)
    index = {root_id: i for i, root_id in enumerate(order)}

    syn_df = pd.read_csv(net.syn_table, usecols=['pre_id', 'post_id'])
    matrix = np.zeros((len(order), len(order)), dtype=int)
    for pre_id, post_id in tqdm(zip(syn_df.pre_id, syn_df.post_id), total=len(syn_df), desc='matrix'):
        if pre_id in index and post_id in index:
            matrix[index[post_id], index[pre_id]] += 1

    mapping = {i: root_id for i, root_id in enumerate(order)}
    save_connectivity_atomic(sparse.csr_matrix(matrix), mapping, net.connectivity)
    print(f'  {len(order)} neurons, {matrix.sum()} synapses, '
          f'{np.count_nonzero(matrix) / matrix.size * 100:.2f}% of pairs connected')


# ─────────────────────────────────────────────────────────────────────────────
# derived: subnetworks (figure 5)
# ─────────────────────────────────────────────────────────────────────────────

SUBNETWORK_TABLES = ('connectome_neurons.csv', 'connectome_synapses.csv', 'spine_table.csv',
                     'connectome_outgoing_synapses.csv', 'spine_table_outgoing.csv')


def subnetwork_files(folder) -> list[str]:
    """Everything one subnetwork folder holds when it is finished."""
    return ([os.path.join(folder, name) for name in SUBNETWORK_TABLES] +
            connectivity_files(os.path.join(folder, 'connectivity_matrix')))


def subnetwork_bands(neurons_df) -> list[tuple]:
    """(fraction, min_x, max_x) per slab, each half the width of the one before it."""
    x = neurons_df.pt_position_xt
    max_x, min_x = max(x), min(x)
    dx = max_x - min_x
    fraction = 1.0

    bands = []
    while dx > 10:
        max_x = max_x - dx // 4
        min_x = min_x + dx // 4
        dx = dx // 2
        fraction /= 2
        bands.append((fraction, min_x, max_x))
    return bands


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
    sources = [MICRO_COLUMN.neuron_table, MICRO_COLUMN.syn_table, MICRO_COLUMN.out_syn_table,
               MICRO_COLUMN.spine_table, MICRO_COLUMN.spine_table_out]

    bands = subnetwork_bands(neurons_df)
    folders = {fraction: os.path.join(DATA_BASE_PATH, NETWORK_NAME, 'subnetworks', str(fraction))
               for fraction, _, _ in bands}
    todo = [band for band in bands
            if not up_to_date(subnetwork_files(folders[band[0]]), sources)]
    if not todo:
        print(f'  all {len(bands)} subnetworks are built and newer than the column '
              f'tables they are cut from — skipping')
        return

    # Only now the expensive reads: `connectome_outgoing_synapses.csv` is the peak-RAM
    # load of the whole script, and an interrupted run has no reason to pay it again to
    # rewrite subnetworks that are already there.
    syn_df = load_synapses_position_transformed()
    spine_df = pd.read_csv(SPINE_TABLE)
    outgoing_syn_df = load_synapses_position_transformed(base_syn_table_path=CONNECTOME_PRE_SYN_TABLE_PATH)
    spine_df_outgoing = pd.read_csv(SPINE_TABLE_OUTGOING)

    for fraction, min_x, max_x in todo:
        neurons = neurons_df[(neurons_df.pt_position_xt <= max_x) &
                             (neurons_df.pt_position_xt >= min_x)].copy()
        sub_syn = syn_df[syn_df.pre_id.isin(neurons.root_id) &
                         syn_df.post_id.isin(neurons.root_id)].copy()
        # every incoming spine, as in the full network — what calc_spines_features expects
        sub_spine = spine_df[spine_df.post_pt_root_id.isin(neurons.root_id)].copy()
        sub_syn_out = outgoing_syn_df[outgoing_syn_df.pre_id.isin(neurons.root_id)].copy()
        sub_spine_out = spine_df_outgoing[spine_df_outgoing.pre_pt_root_id.isin(neurons.root_id)].copy()

        folder = folders[fraction]
        os.makedirs(folder, exist_ok=True)
        clear_partials(folder)
        print(f'  {fraction}: {len(neurons)} neurons, {len(sub_syn)} synapses, '
              f'{len(sub_spine)} incoming spines')

        for df, name in ((neurons, 'connectome_neurons.csv'),
                         (sub_syn, 'connectome_synapses.csv'),
                         (sub_spine, 'spine_table.csv'),
                         (sub_syn_out, 'connectome_outgoing_synapses.csv'),
                         (sub_spine_out, 'spine_table_outgoing.csv')):
            with atomic(os.path.join(folder, name)) as staging:
                df.to_csv(staging)

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
        save_connectivity_atomic(sparse.csr_matrix(matrix),
                                 {i: root_id for i, root_id in enumerate(ordered)},
                                 conn_dir)


# ─────────────────────────────────────────────────────────────────────────────
# steps
# ─────────────────────────────────────────────────────────────────────────────

def neurons_complete(net: Network, ids, allow_incomplete: bool) -> bool:
    """Whether every cell's synapse pickle landed — the input all four tables below read.

    They are built from the whole set at once, so a cell missing here is a hole in every
    one of them. The usual cause is a connection that dropped somewhere in a day of
    queries, and the fix is another run, which re-fetches only the cells it has to. (The
    spine tables are built from none of this and report their own gaps, in
    `download_spines`.)
    """
    # A skeleton is the one thing the pipeline already tolerates losing: a cell without
    # one keeps the -1 cable lengths it was born with, which is how 864691136620192653
    # ships in the published tables. Worth saying, not worth stopping for.
    no_skeleton = [i for i in ids if not complete(os.path.join(net.skeletons, f'{i}.swc'))]
    if no_skeleton:
        print(f'  {len(no_skeleton)} of {len(ids)} cells have no skeleton — their cable '
              f'lengths and synapse-to-soma distances stay -1')

    missing = [i for i in ids if not complete(os.path.join(net.neurons, f'{i}.pkl'))]
    if not missing:
        return True

    print(f'  {len(missing)} of {len(ids)} cells never got their synapses: '
          f'{", ".join(str(i) for i in missing[:3])}{" …" if len(missing) > 3 else ""}')
    if allow_incomplete:
        print('  --allow-incomplete: building the tables from what is here anyway.')
        return True
    print('  Not building the network-wide tables from a partial set. Re-run the same '
          'command to fetch what is missing — it picks up exactly here.')
    return False


def build_full_network(net: Network, root_ids, client, allow_incomplete=False):
    """Everything for one network folder: download, then derive."""
    ids, cell_types, m_types = resolve_cell_types(root_ids, net.column_manual_ct)
    if not ids:
        raise BuildError(f'{net.name}: none of the {len(root_ids)} cells survived the cell-type '
                         f'tables in {RAW_TABLES_DIR}. Nothing to build — check those tables.')

    download_neurons(net, ids, cell_types, m_types, client)
    download_skeletons(net, ids, client)
    download_spines(net, ids, client, incoming=True, allow_incomplete=allow_incomplete)
    download_spines(net, ids, client, incoming=False, allow_incomplete=allow_incomplete)

    if not neurons_complete(net, ids, allow_incomplete):
        return

    if not build_em_neurons(net):
        return

    build_connectome_tables(net)
    build_outgoing_synapse_table(net)
    build_connectivity_matrix(net)


def step_raw(client, args):
    download_raw_tables(client)


def step_column(client, args):
    build_full_network(MICRO_COLUMN, column_root_ids(), client, args.allow_incomplete)


def step_subnets(client, args):
    build_subnetworks()


def step_meshes(client, args):
    download_meshes(client)


def step_axon_pr(client, args):
    build_full_network(ALL_AXON_PR, proofread_axon_root_ids(), client, args.allow_incomplete)


STEPS = {
    'raw': step_raw,
    'column': step_column,
    'subnets': step_subnets,
    'meshes': step_meshes,
    'axon_pr': step_axon_pr,
}

# `subnets` is cut from files the other steps already wrote, so on its own it needs
# neither a connection nor a CAVE account.
OFFLINE_STEPS = {'subnets'}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default=','.join(STEPS),
                        help=f"comma-separated subset of: {', '.join(STEPS)} (default: all, in order)")
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='build the network-wide tables even when some cells are still '
                             'missing their downloads (default: stop and say which)')
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}. Pick from: {', '.join(STEPS)}")

    client = None if set(steps) <= OFFLINE_STEPS else cave_client()
    try:
        for name in steps:
            print(f'\n=== {name} ===')
            STEPS[name](client, args)
    except KeyboardInterrupt:
        print('\nStopped. Nothing half-written was kept — re-run the same command to '
              'continue from here.')
        return 130
    except BuildError as e:
        print(f'\n{e}')
        return 1

    if DISCARDED:
        print(f'\n{len(DISCARDED)} damaged file(s) left by an earlier run were removed. '
              f'Re-run the same command to fetch and rebuild them.')
        return 1

    print('\nDone. data/activity/ is still yours to build — see the README.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
