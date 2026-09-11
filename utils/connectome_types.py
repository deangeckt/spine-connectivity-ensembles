import os
from enum import Enum

# Absolute path to data/micro_column_network — works wherever this repo is checked out
_MCN = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'micro_column_network'))
_DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
_ACTIVITY = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'activity'))

# The pre-rendered dendrite images and EM cut-outs behind figures 1, 2 and S1. These
# live in the repository, not in data/: the branch renders were made by hand in VAST and
# neither download route can rebuild them. See the README.
IMAGES_BASE_PATH            = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'images'))

DATA_BASE_PATH              = _DATA
NETWORK_NAME                = 'micro_column_network'
MATERIALIZATION_VERSION     = 1718

CONNECTOME_SYN_TABLE_PATH   = os.path.join(_MCN, 'connectome_synapses.csv')
CONNECTOME_PRE_SYN_TABLE_PATH = os.path.join(_MCN, 'connectome_outgoing_synapses.csv')
CONNECTOME_NEURON_TABLE_PATH  = os.path.join(_MCN, 'connectome_neurons.csv')
SKELETONS_DIR_PATH          = os.path.join(_MCN, 'skeletons')
NEURONS_PATH                = os.path.join(_MCN, 'neurons')
EM_NEURONS_PATH             = os.path.join(_MCN, 'em_neurons')
CONNECTIVITY_DIR            = os.path.join(_MCN, 'connectivity_matrix')
SPINE_TABLE                 = os.path.join(_MCN, 'spine_table.csv')
SPINE_TABLE_OUTGOING        = os.path.join(_MCN, 'spine_table_outgoing.csv')

# Activity / ensembles (figures 6, 7, S14, S15). NONE of this is in the Zenodo
# snapshot: the two H5 files are built with scripts/extract_calcium_data_via_docker.ipynb
# inside the MICrONS database container, and the ensemble results with
# scripts/ensemble_run.py, which reads them. See the README.
CALCIUM_H5_PATH             = os.path.join(_ACTIVITY, 'coreg_manual_v4_calcium_v2.h5')
STIMULI_H5_PATH             = os.path.join(_ACTIVITY, 'microns_per_scan_stimuli.h5')
ENSEMBLE_RESULTS_DIR        = os.path.join(_ACTIVITY, 'ensembles')

class ClfType(str, Enum):
    excitatory = 'E'
    inhibitory = 'I'
    both = 'both'


m_types = [
    'DTC', 'ITC', 'L2a', 'L2b', 'L2c', 'L3a', 'L3b',
    'L4a', 'L4b', 'L4c', 'L5ET', 'L5NP', 'L5a', 'L5b',
    'L6short-a', 'L6short-b', 'L6tall-a', 'L6tall-b', 'L6tall-c',
    'PTC', 'STC',
]

cell_types = [
    '23P', '4P', '5P-ET', '5P-IT', '5P-NP',
    '6P-CT', '6P-IT', 'BC', 'BPC', 'MC', 'NGC',
    'OPC', 'astrocyte', 'microglia', 'oligo', 'pericyte',
]

col_cell_types_ordered = [
    '23P', '4P', '5P-IT', '5P-NP', '5P-PT',
    '6P-CT', '6P-IT', '6P-U', 'WM-P',
    'Unsure E', 'BC', 'BPC', 'MC', 'NGC', 'Unsure I',
]
