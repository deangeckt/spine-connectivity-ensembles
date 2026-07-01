import numpy as np
import pandas as pd
from tqdm import tqdm
from connectivity_matrix_utils import load_connectivity
from connectome_types import CONNECTIVITY_DIR, SPINE_TABLE, SPINE_TABLE_OUTGOING
from sk_load_utils import load_col_skeleton_only, filter_sk_bounds

def calc_basic_degree_conn_features(df: pd.DataFrame) -> None:
    df['incoming_contacts_density'] = df['num_of_incoming_neurons'] / df['dendrite_length']
    df['ex_incoming_contacts_density'] = df['num_of_ex_incoming_neurons'] / df['dendrite_length']
    df['inh_incoming_contacts_density'] = df['num_of_inh_incoming_neurons'] / df['dendrite_length']

    df['incoming_synapses_density'] = df['num_of_incoming_synapses'] / df['dendrite_length']
    df['ex_incoming_synapses_density'] = df['num_of_ex_incoming_synapses'] / df['dendrite_length']
    df['inh_incoming_synapses_density'] = df['num_of_inh_incoming_synapses'] / df['dendrite_length']

    df['incoming_multiple_contacts_ratio'] = df.apply(lambda row: 0 if row['num_of_incoming_synapses'] == 0 else row['num_of_incoming_synapses'] / row['num_of_incoming_neurons'], axis=1)
    df['ex_incoming_multiple_contacts_ratio'] = df.apply(lambda row: 0 if row['num_of_ex_incoming_synapses'] == 0 else row['num_of_ex_incoming_synapses'] / row['num_of_ex_incoming_neurons'], axis=1)
    df['inh_incoming_multiple_contacts_ratio'] = df.apply(lambda row: 0 if row['num_of_inh_incoming_synapses'] == 0 else row['num_of_inh_incoming_synapses'] / row['num_of_inh_incoming_neurons'], axis=1)
    

def calc_sk_length_in_column(neurons_df: pd.DataFrame) -> pd.DataFrame:
    min_x, max_x = neurons_df.pt_position_xt.min(), neurons_df.pt_position_xt.max()
    min_z, max_z = neurons_df.pt_position_zt.min(), neurons_df.pt_position_zt.max()
    print(f"Max x: {max_x}, Min x: {min_x}")
    print(f"Max z: {max_z}, Min z: {min_z}")
    
    def get_axon_length(node_id):
        sk_axon = load_col_skeleton_only(node_id, reset_dentrites=True)
        sk_axon_local = filter_sk_bounds(sk_axon, min_x, max_x, min_z, max_z)
        return sk_axon_local.path_length()

    def get_dendrite_length(node_id):
        sk_dend = load_col_skeleton_only(node_id, reset_axon=True)
        sk_dend_local = filter_sk_bounds(sk_dend, min_x, max_x, min_z, max_z)
        return sk_dend_local.path_length()
    
    neurons_df['axon_local_path_length'] = neurons_df['root_id'].map(get_axon_length)
    neurons_df['dendrite_local_path_length'] = neurons_df['root_id'].map(get_dendrite_length)


def calc_basic_degrree_features(df: pd.DataFrame) -> None:
    # EI ratios
    e_ = 'num_of_ex_incoming_synapses'
    i_ = 'num_of_inh_incoming_synapses'
    we_ = 'ex_incoming_syn_sum_weight'
    wi_ = 'inh_incoming_syn_sum_weight'

    e_par = 'num_of_ex_incoming_neurons'
    i_par = 'num_of_inh_incoming_neurons'

    df['ei_incoming_ratio_norm'] = df.apply(lambda row: row[e_] / (row[i_] + row[e_]) if (row[i_] + row[e_]) != 0 else np.nan, axis=1)
    df['ei_incoming_ratio'] = df.apply(lambda row: row[e_] / row[i_] if row[i_] != 0 else np.nan, axis=1)

    df['ei_incoming_sum_of_weights_ratio_norm'] = df.apply(lambda row: row[we_] / (row[wi_] + row[we_]) if (row[wi_] + row[we_]) != 0 else np.nan, axis=1)
    df['ei_incoming_sum_of_weights_ratio'] = df.apply(lambda row: row[we_] / row[wi_] if row[wi_] != 0 else np.nan, axis=1)

    df['er_incoming_partners_ratio_norm'] = df.apply(lambda row: row[e_par] / (row[i_par] + row[e_par]) if (row[i_par] + row[e_par]) != 0 else np.nan, axis=1)
    # df['ei_outgoing_ratio_norm'] = df.apply(lambda row: row['num_of_ex_outgoing_synapses'] / (row['num_of_inh_outgoing_synapses'] + row['num_of_ex_outgoing_synapses']) if (row['num_of_inh_outgoing_synapses'] + row['num_of_ex_outgoing_synapses']) != 0 else np.nan, axis=1)
    # df['ei_outgoing_ratio'] = df.apply(lambda row: row['num_of_ex_outgoing_synapses'] / row['num_of_inh_outgoing_synapses'] if row['num_of_inh_outgoing_synapses'] != 0 else np.nan, axis=1);

    df['incoming_synapses_in_vol_prop'] = df['num_of_incoming_synapses'] / df['ds_num_of_incoming_synapses']
    df['ds_incoming_synapses_density'] = df['ds_num_of_incoming_synapses'] / df['dendrite_length']
    df['incoming_synapses_per_contact'] = df['num_of_incoming_synapses'] / df['num_of_incoming_neurons']

    df['ds_outgoing_synapses_density'] = df['ds_num_of_outgoing_synapses'] / df['axon_length']

    calc_basic_degree_conn_features(df)


def calc_spines_features(neurons_df: pd.DataFrame,
                        syn_df: pd.DataFrame,
                        spine_table=SPINE_TABLE,
                        spine_table_outgoing=SPINE_TABLE_OUTGOING):
    
    spine_df = pd.read_csv(spine_table) # incoming table
    spine_df_outgoing = pd.read_csv(spine_table_outgoing) # outgoing table
    print('spine table incoming size', spine_df.shape)
    print('spine table outgoing size', spine_df_outgoing.shape)

    syn_with_tags = syn_df[syn_df.id_.isin(spine_df.target_id)].copy()
    syn_with_tags['tag'] = syn_with_tags.id_.map(spine_df.set_index('target_id').tag)
    print(f'neurons: {syn_with_tags.post_id.nunique()}')
    print(f'synapses with tags: {syn_with_tags.shape[0]}')

    root_ids = list(neurons_df.root_id)

    # incoming analysis per neuron
    results = []
    for root_id in tqdm(root_ids):
        ## Incoming
        neuron_incoming = syn_with_tags[syn_with_tags.post_id == root_id]        
        spines = neuron_incoming[neuron_incoming.tag == 'spine']
        shafts = neuron_incoming[neuron_incoming.tag == 'shaft']
        somas = neuron_incoming[neuron_incoming.tag == 'soma']
        # full ds - we trust the dendrite
        neuron_incoming_ds = spine_df[spine_df.post_pt_root_id == root_id]
        ds_spines = neuron_incoming_ds[neuron_incoming_ds.tag == 'spine']
        ds_shafts = neuron_incoming_ds[neuron_incoming_ds.tag == 'shaft']
        ds_somas = neuron_incoming_ds[neuron_incoming_ds.tag == 'soma']

        ## Outgoing
        neuron_outgoing_ds = spine_df_outgoing[spine_df_outgoing.pre_pt_root_id == root_id]
        ds_spines_outgoing = neuron_outgoing_ds[neuron_outgoing_ds.tag == 'spine']
        ds_shafts_outgoing = neuron_outgoing_ds[neuron_outgoing_ds.tag == 'shaft']
        ds_somas_outgoing = neuron_outgoing_ds[neuron_outgoing_ds.tag == 'soma']


        results.append({
            'root_id': root_id,
            'number_syn_on_soma': somas.shape[0],
            'number_syn_on_spines': spines.shape[0],
            'number_syn_on_shafts': shafts.shape[0],
            'spine_ratio': spines.shape[0] / len(neuron_incoming) if len(neuron_incoming) > 0 else np.nan,
            'number_contacts_on_soma': somas.pre_id.nunique(),
            'number_contacts_on_spines': spines.pre_id.nunique(),
            'number_contacts_on_shafts': shafts.pre_id.nunique(),
            'number_of_ex_syn_on_soma': somas[somas.pre_clf_type == 'E'].shape[0],
            'number_of_ex_syn_on_spines': spines[spines.pre_clf_type == 'E'].shape[0],
            'number_of_ex_syn_on_shafts': shafts[shafts.pre_clf_type == 'E'].shape[0],
            'number_of_inh_syn_on_soma': somas[somas.pre_clf_type == 'I'].shape[0],
            'number_of_inh_syn_on_spines': spines[spines.pre_clf_type == 'I'].shape[0],
            'number_of_inh_syn_on_shafts': shafts[shafts.pre_clf_type == 'I'].shape[0],
            # Low Lvl - not realy used so far
            'number_of_inh_contacts_on_spines': spines[spines.pre_clf_type == 'I'].pre_id.nunique(),
            'number_of_ex_contacts_on_spines': spines[spines.pre_clf_type == 'E'].pre_id.nunique(),
            'EI_ratio_on_spines': spines[spines.pre_clf_type == 'E'].shape[0] / spines.shape[0] if spines.shape[0] > 0 else np.nan,

            # Extrinsic 
            'ds_number_syn_on_soma': ds_somas.shape[0],
            'ds_number_syn_on_spines': ds_spines.shape[0],
            'ds_number_syn_on_shafts': ds_shafts.shape[0],
            'ds_spine_ratio': ds_spines.shape[0] / len(neuron_incoming_ds) if len(neuron_incoming_ds) > 0 else np.nan,
            'ds_shaft_ratio': ds_shafts.shape[0] / len(neuron_incoming_ds) if len(neuron_incoming_ds) > 0 else np.nan,

            'ds_number_outgoing_syn_on_soma': ds_somas_outgoing.shape[0],
            'ds_number_outgoing_syn_on_spines': ds_spines_outgoing.shape[0],
            'ds_number_outgoing_syn_on_shafts': ds_shafts_outgoing.shape[0],
            'ds_spine_ratio_outgoing': ds_spines_outgoing.shape[0] / len(neuron_outgoing_ds) if len(neuron_outgoing_ds) > 0 else np.nan,
            'ds_shaft_ratio_outgoing': ds_shafts_outgoing.shape[0] / len(neuron_outgoing_ds) if len(neuron_outgoing_ds) > 0 else np.nan
        })

    neurons_spine_df = pd.DataFrame(results)
    neurons_df = neurons_df.merge(neurons_spine_df, on='root_id', how='left')

    df = neurons_df.sort_values('cell_type')

    df['spine_to_shaft_syn_ratio'] = df['number_syn_on_spines'] / (df['number_syn_on_shafts'] + df['number_syn_on_spines'])
    df['spine_to_shaft_syn_ratio_ds'] = df['ds_number_syn_on_spines'] / (df['ds_number_syn_on_shafts'] + df['ds_number_syn_on_spines'])
    df['spine_to_shaft_contacts_ratio'] = df['number_contacts_on_spines'] / (df['number_contacts_on_shafts'] + df['number_contacts_on_spines'])

    df['spines_per_contact'] = df['number_syn_on_spines'] / df['number_contacts_on_spines']
    df['shafts_per_contact'] = df['number_syn_on_shafts'] / df['number_contacts_on_shafts']

    df['ds_spine_density'] = df['ds_number_syn_on_spines'] / df['dendrite_length']
    df['ds_shaft_density'] = df['ds_number_syn_on_shafts'] / df['dendrite_length']

    df['ds_outgoing_spine_density'] = df['ds_number_outgoing_syn_on_spines'] / df['axon_length']
    df['ds_outgoing_shaft_density'] = df['ds_number_outgoing_syn_on_shafts'] / df['axon_length']

    df['spine_density'] = df['number_syn_on_spines'] / df['dendrite_length']
    df['shaft_density'] = df['number_syn_on_shafts'] / df['dendrite_length']


    return df, syn_with_tags


