import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from connectivity_matrix_utils import load_connectivity
from connectome_types import (
    CONNECTIVITY_DIR, CONNECTOME_SYN_TABLE_PATH,
    DATA_BASE_PATH, CONNECTOME_NEURON_TABLE_PATH,
)
from standard_transform import minnie_ds
from connectome_types import col_cell_types_ordered


process_str_position = lambda s: [float(x) if '.' in x else int(x) for x in re.findall(r'-?\d+(?:\.\d+)?', s)]


def transform_coord(df, s3=True, is_scale=True):
    if s3:
        df['pos'] = df['pos'].apply(lambda x: x * 1000)
    if is_scale:
        df['pos'] = df['pos'].apply(lambda x: x / np.array([4, 4, 40]))
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pos', df)
    X_transformed = np.array(X_transformed)
    df['pos'] = list(X_transformed)


def transform_sk(sk, s3=True):
    sk_pos_df = pd.DataFrame({'pos': list(sk.vertices)})
    transform_coord(sk_pos_df, s3=s3, is_scale=True)
    sk.vertices = list(sk_pos_df.pos)


def load_neuron_position_transformed(use_column_manual_ct=True, neuron_table=CONNECTOME_NEURON_TABLE_PATH):
    neurons_df = pd.read_csv(neuron_table, index_col=0)
    print('connectome neurons table: ', len(neurons_df))
    root_ids = list(neurons_df.root_id)

    if use_column_manual_ct:
        celltypes_df = pd.read_csv(f'{DATA_BASE_PATH}/raw_tables/allen_v1_column_types_slanted_ref.csv', index_col=0)
    else:
        celltypes_df = pd.read_csv(f'{DATA_BASE_PATH}/raw_tables/aibs_metamodel_celltypes_v661.csv', index_col=0)

    celltypes_df = celltypes_df[celltypes_df.classification_system != 'nonneuron'].copy()
    celltypes_df = celltypes_df[celltypes_df.classification_system != 'aibs_coarse_unclear'].copy()
    celltypes_df = celltypes_df[celltypes_df.pt_root_id != 0]
    celltypes_df.drop_duplicates("pt_root_id", keep=False, inplace=True)

    valid_neurons = celltypes_df[celltypes_df.pt_root_id.isin(root_ids)].copy()
    valid_neurons = valid_neurons.sort_values('cell_type')
    print('valid neurons w position:', len(valid_neurons))

    if 'pt_position_x' not in valid_neurons.columns:
        valid_neurons['pt_position_num'] = valid_neurons['pt_position'].apply(process_str_position)
        valid_neurons['pt_position_x'] = valid_neurons['pt_position_num'].apply(lambda x: x[0])
        valid_neurons['pt_position_y'] = valid_neurons['pt_position_num'].apply(lambda x: x[1])
        valid_neurons['pt_position_z'] = valid_neurons['pt_position_num'].apply(lambda x: x[2])

    merged_df = pd.merge(
        neurons_df,
        valid_neurons[['pt_root_id', 'pt_position_x', 'pt_position_y', 'pt_position_z']],
        left_on='root_id',
        right_on='pt_root_id',
        how='left'
    )
    merged_df = merged_df.drop('pt_root_id', axis=1)

    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', merged_df)
    X_transformed = np.array(X_transformed)
    merged_df['pt_position_xt'] = X_transformed[:, 0]
    merged_df['pt_position_yt'] = X_transformed[:, 1]
    merged_df['pt_position_zt'] = X_transformed[:, 2]

    return merged_df


def load_neurons_table(use_column_manual_ct=True, load_raw_table=False):
    if load_raw_table:
        return pd.read_csv(CONNECTOME_NEURON_TABLE_PATH, index_col=0)

    from neuron_custom_features import calc_basic_degrree_features

    df = load_neuron_position_transformed(use_column_manual_ct=use_column_manual_ct)
    calc_basic_degrree_features(df)
    df_ = df.copy()
    if use_column_manual_ct:
        df_ = df_.astype({'cell_type': pd.CategoricalDtype(categories=col_cell_types_ordered, ordered=True)}).sort_values('cell_type')
    return df_


def load_synapses_position_transformed(base_syn_table_path=CONNECTOME_SYN_TABLE_PATH):
    synapse_df = pd.read_csv(base_syn_table_path, index_col=0)
    synapse_df['center_position'] = synapse_df['center_position'].apply(process_str_position)
    synapse_df['pos'] = synapse_df['center_position']
    transform_coord(synapse_df, s3=False, is_scale=False)

    positions_df = pd.DataFrame(synapse_df['pos'].tolist(),
                            index=synapse_df.index,
                            columns=['pt_position_xt', 'pt_position_yt', 'pt_position_zt'])
    synapse_df[['pt_position_xt', 'pt_position_yt', 'pt_position_zt']] = positions_df
    return synapse_df


def load_bin_mat(connectivity_dir=CONNECTIVITY_DIR, name='network_synapses'):
    syn_mat_sparse, mapping = load_connectivity(connectivity_dir, name)
    neuron_names = list(mapping.keys())
    reverse_mapping = {v: k for k, v in mapping.items()}
    syn_mat = syn_mat_sparse.toarray().T

    bin_mat_sparse = syn_mat_sparse.T.copy()
    bin_mat_sparse[bin_mat_sparse > 1] = 1
    bin_mat = bin_mat_sparse.toarray()
    return bin_mat, bin_mat_sparse, syn_mat, mapping, reverse_mapping, neuron_names
