import os
from enum import Enum

# Absolute path to standalone/data/micro_column_network — works wherever this folder is copied
_MCN = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'micro_column_network'))
_DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))

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

# Not included in standalone data — kept as None so existing utils import without errors
NEURON_IDS_TABLE                    = None
MOTIFS_DIR                          = None
ORACLE_SCORES_TABLE                 = None
ALLEN_COL_ALL_SYNAPSES_TABLE        = None
CONNECTOME_NEURON_MorphoPy_TABLE_PATH = None
CONNECTOME_NEURON_neuroM_TABLE_PATH   = None
CONNECTOME_NEURON_SHUFFLED_TABLE_PATH = None
SKELETONS_S3_DIR_PATH               = None
SKELETONS_S3_SWC_DIR_PATH           = None


class SynapseSide(str, Enum):
    post = 'post'
    pre = 'pre'


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
