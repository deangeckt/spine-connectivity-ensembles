from tqdm import tqdm
import numpy as np
import pandas as pd


def filter_valid_neuron_w_spines(df):
    from utils import load_bin_mat
    print("Filtering neurons with valid spine data...")
    df = df[df.dendrite_length > 0]
    df = df[df.axon_length > 0]
    df = df[~df.ds_spine_ratio.isna()]
    df['spines_tagged_ratio'] = (df.ds_number_syn_on_soma+df.ds_number_syn_on_shafts+df.ds_number_syn_on_spines)/df.ds_num_of_incoming_synapses
    df = df[df.spines_tagged_ratio > 0.9]

    print(f"Remaining neurons after filtering: {len(df)}")
    print('fixing networks')

    _, _, syn_mat, mapping, _, _ = load_bin_mat()

    ex_df = df[df.clf_type == 'E']
    inh_df = df[df.clf_type == 'I']

    filtered_syn_mat, filtered_mapping = filter_syn_mat_by_dfs(syn_mat, mapping, ex_df, inh_df)
    filtered_reverse_mapping = {v: k for k, v in filtered_mapping.items()}
    filtered_bin_mat = filtered_syn_mat.copy()
    filtered_bin_mat[filtered_bin_mat > 1] = 1
    neuron_clf_type = df[['root_id', 'clf_type']].set_index('root_id').to_dict(orient='index')

    df, ex_neurons, inh_neurons = get_connectivity_features(df, filtered_syn_mat, filtered_mapping, neuron_clf_type)
    assert (len(df) == len(filtered_mapping)), "Mismatch between filtered df and mapping sizes!"

    # sanity
    neuron_idx = 50
    row_ = filtered_syn_mat[neuron_idx, :] # row 0
    col_ = filtered_syn_mat[:, neuron_idx] # col 0
    bin_row  = filtered_bin_mat[neuron_idx, :]
    bin_col  = filtered_bin_mat[:, neuron_idx]

    root_id = filtered_mapping[neuron_idx]
    assert np.sum(row_) == df[df.root_id == root_id][['num_of_outgoing_synapses']].values[0]
    assert np.sum(col_) == df[df.root_id == root_id][['num_of_incoming_synapses']].values[0]
    assert np.sum(bin_row) == df[df.root_id == root_id][['num_of_outgoing_neurons']].values[0]
    assert np.sum(bin_col) == df[df.root_id == root_id][['num_of_incoming_neurons']].values[0]
    return df, filtered_syn_mat, filtered_bin_mat, filtered_mapping, filtered_reverse_mapping, ex_neurons, inh_neurons


## Mutual Similarity related utils

def get_incoming_neurons_sets(bin_mat):
    n_neurons = bin_mat.shape[1]
    incoming_neurons_sets = []
    for neuron_idx in range(n_neurons):
        incoming_neurons = set(np.where(bin_mat[:, neuron_idx] > 0)[0])
        incoming_neurons_sets.append(incoming_neurons)
    return incoming_neurons_sets


## CFG null model utils

def configuration_model(M, seed = None):
    import scipy.sparse as sp

    """Function to generate the configuration control model, obtained by
    shuffling the row and column of coo format independently, to create
    new coo matrix, then removing any multiple edges and loops.

    Parameters
    ----------
    adj : coo-matrix
        Adjacency matrix of a directed network.
    seed : int
        Random seed to be used

    Returns
    -------
    csr matrix
        Configuration model control of adj

    See Also
    --------
    [run_SBM](randomization.md#src.connalysis.randomization.randomization.run_SBM) :
    Function which runs the stochastic block model

    [run_DD2](randomization.md#src.connalysis.randomization.randomization.run_DD2) :
    Function which runs the 2nd distance dependent model
    """
    adj=M.copy().tocoo()
    generator = np.random.default_rng(seed)
    R = adj.row
    C = adj.col
    generator.shuffle(R)
    generator.shuffle(C)
    CM_matrix = sp.coo_matrix(([1]*len(R),(R,C)),shape=adj.shape).tocsr()
    CM_matrix.setdiag(0)
    CM_matrix.eliminate_zeros()
    return CM_matrix


## Generic connectivity feature utils

def get_conn_features_from_mat(df, syn_mat, mapping, neuron_clf_type):
    """
    Calculates connectivity features (incoming E/I counts) and adds them to the dataframe.

    Args:
        df (pd.DataFrame): The simple_df containing 'root_id'.
        syn_mat (np.array): The synapse matrix (rows=source, cols=target).
        mapping (dict): Maps matrix index (int) -> root_id (int).
        neuron_clf_type (dict): Maps root_id -> {'clf_type': 'E' or 'I'}.

    Returns:
        pd.DataFrame: The updated dataframe with new columns.
    """
    num_neurons = syn_mat.shape[0]

    # 1. Create Boolean Masks for E and I neurons
    # We need a vector aligned with the matrix rows (0 to N) that tells us the type
    is_exc = np.zeros(num_neurons, dtype=bool)
    is_inh = np.zeros(num_neurons, dtype=bool)

    for idx in range(num_neurons):
        root_id = mapping[idx]
        # Get type safely; defaults to None if missing
        n_type = neuron_clf_type.get(root_id, {}).get('clf_type')

        if n_type == 'E':
            is_exc[idx] = True
        elif n_type == 'I':
            is_inh[idx] = True

    # 2. Pre-calculate Binary Matrix (Connectivity Mask)
    bin_mat = (syn_mat > 0).astype(int)

    # 3. Calculate Global Features (All inputs + outputs)
    total_inc_syn = syn_mat.sum(axis=0)
    total_inc_neu = bin_mat.sum(axis=0)
    total_out_syn = syn_mat.sum(axis=1)   # row sum = outgoing
    total_out_neu = bin_mat.sum(axis=1)

    # 4. Excitatory Features
    ex_inc_syn = syn_mat[is_exc, :].sum(axis=0)
    ex_inc_neu = bin_mat[is_exc, :].sum(axis=0)
    ex_out_syn = syn_mat[:, is_exc].sum(axis=1)
    ex_out_neu = bin_mat[:, is_exc].sum(axis=1)

    # 5. Inhibitory Features
    inh_inc_syn = syn_mat[is_inh, :].sum(axis=0)
    inh_inc_neu = bin_mat[is_inh, :].sum(axis=0)
    inh_out_syn = syn_mat[:, is_inh].sum(axis=1)
    inh_out_neu = bin_mat[:, is_inh].sum(axis=1)

    # 6. Prepare Data for Merge
    # We create a temporary DF indexed by the matrix index 0..N
    stats_df = pd.DataFrame({
        'matrix_idx': range(num_neurons),
        'root_id': [mapping[i] for i in range(num_neurons)],
        'num_of_incoming_neurons': total_inc_neu,
        'num_of_incoming_synapses': total_inc_syn,
        'num_of_ex_incoming_neurons': ex_inc_neu,
        'num_of_inh_incoming_neurons': inh_inc_neu,
        'num_of_ex_incoming_synapses': ex_inc_syn,
        'num_of_inh_incoming_synapses': inh_inc_syn,
        # outgoing
        'num_of_outgoing_neurons':          total_out_neu,
        'num_of_outgoing_synapses':         total_out_syn,
        'num_of_ex_outgoing_neurons':       ex_out_neu,
        'num_of_inh_outgoing_neurons':      inh_out_neu,
        'num_of_ex_outgoing_synapses':      ex_out_syn,
        'num_of_inh_outgoing_synapses':     inh_out_syn,
    })

    # 7. Merge into the original simple_df
    # We merge on root_id to ensure alignment with your existing features
    result_df = df.merge(stats_df.drop(columns=['matrix_idx']), on='root_id', how='left')

    return result_df.copy()


def split_syn_mat_by_type_four(syn_mat, mapping, neuron_clf_type):
    """
    Splits the full syn_mat into the four typed sub-matrices:
      EE (E->E), EI (E->I), IE (I->E), II (I->I).

    Unlike split_syn_mat_by_type (which only returns EE and II), this version
    also returns the cross-type blocks so that each of the four connectivity
    channels can be shuffled independently — creating a stronger "typed"
    binary null model.

    Returns:
        EE_mat, EI_mat, IE_mat, II_mat  – (sub-)matrices, shapes depend on counts
        ex_indices, inh_indices          – global matrix index arrays for reconstruction
    """
    num_neurons = syn_mat.shape[0]
    ex_indices, inh_indices = [], []

    for idx in range(num_neurons):
        root_id = mapping[idx]
        n_type  = neuron_clf_type.get(root_id, {}).get('clf_type')
        if   n_type == 'E': ex_indices.append(idx)
        elif n_type == 'I': inh_indices.append(idx)

    ex_indices  = np.array(ex_indices)
    inh_indices = np.array(inh_indices)

    # rows = source, cols = target
    EE_mat = syn_mat[np.ix_(ex_indices,  ex_indices)]   # E -> E
    EI_mat = syn_mat[np.ix_(ex_indices,  inh_indices)]  # E -> I
    IE_mat = syn_mat[np.ix_(inh_indices, ex_indices)]   # I -> E
    II_mat = syn_mat[np.ix_(inh_indices, inh_indices)]  # I -> I

    return EE_mat, EI_mat, IE_mat, II_mat, ex_indices, inh_indices


def merge_syn_mats_four_blocks(EE_mat, EI_mat, IE_mat, II_mat,
                                ex_indices, inh_indices, n_total):
    """
    Reassembles the four typed sub-matrices into a full N×N matrix.
    Counterpart to split_syn_mat_by_type_four.
    """
    dtype = EE_mat.dtype
    reconstructed = np.zeros((n_total, n_total), dtype=dtype)
    reconstructed[np.ix_(ex_indices,  ex_indices)]  = EE_mat
    reconstructed[np.ix_(ex_indices,  inh_indices)] = EI_mat
    reconstructed[np.ix_(inh_indices, ex_indices)]  = IE_mat
    reconstructed[np.ix_(inh_indices, inh_indices)] = II_mat
    return reconstructed


## ── Connectivity feature extraction ──────────────────────────────────────────

_CONNECTIVITY_COLS = [
    'num_of_incoming_neurons', 'num_of_incoming_synapses',
    'num_of_ex_incoming_neurons', 'num_of_inh_incoming_neurons',
    'num_of_ex_incoming_synapses', 'num_of_inh_incoming_synapses',
    'num_of_outgoing_neurons', 'num_of_outgoing_synapses',
    'num_of_ex_outgoing_neurons', 'num_of_inh_outgoing_neurons',
    'num_of_ex_outgoing_synapses', 'num_of_inh_outgoing_synapses',
    'incoming_contacts_density', 'ex_incoming_contacts_density',
    'inh_incoming_contacts_density', 'incoming_synapses_density',
    'ex_incoming_synapses_density', 'inh_incoming_synapses_density',
    'incoming_multiple_contacts_ratio', 'ex_incoming_multiple_contacts_ratio',
    'inh_incoming_multiple_contacts_ratio',
]


def get_connectivity_features(filtered_df, syn_mat, filtered_mapping, neuron_clf_type):
    from neuron_custom_features import calc_basic_degree_conn_features

    """
    Computes connectivity features for every neuron in *filtered_df* using
    the provided *syn_mat* (which may be the real matrix or any shuffled
    variant produced by get_shuffle).

    Args:
        filtered_df      (pd.DataFrame): Per-neuron metadata; must contain 'root_id'
                                         and 'clf_type'.
        syn_mat          (np.ndarray):   N×N connectivity matrix aligned to
                                         filtered_mapping.
        filtered_mapping (dict):         {matrix_index: root_id}
        neuron_clf_type  (dict):         {root_id: {'clf_type': 'E' | 'I'}}

    Returns:
        conn_df (pd.DataFrame): Full dataframe with new connectivity columns.
        ex_df   (pd.DataFrame): Subset where clf_type == 'E'.
        inh_df  (pd.DataFrame): Subset where clf_type == 'I'.
    """
    # Drop stale connectivity columns so they are recomputed cleanly
    cols_to_drop = [c for c in _CONNECTIVITY_COLS if c in filtered_df.columns]
    simple_df = filtered_df.drop(columns=cols_to_drop).copy()

    conn_df = get_conn_features_from_mat(simple_df, syn_mat, filtered_mapping, neuron_clf_type)
    calc_basic_degree_conn_features(conn_df)
    ex_df  = conn_df[conn_df.clf_type == 'E']
    inh_df = conn_df[conn_df.clf_type == 'I']
    return conn_df, ex_df, inh_df


## ── Shuffle factory ──────────────────────────────────────────────────────────

def _generate_one_shuffle(bin_mat, shuffle_mode, neuron_clf_type,
                           filtered_mapping, shuffle_preserve_EI, seed=None):
    """Internal: generate a single shuffled binary matrix."""
    if shuffle_mode == 'cfg':
        import scipy.sparse as sp
        if shuffle_preserve_EI:
            EE, EI, IE, II, ex_idx, inh_idx = split_syn_mat_by_type_four(
                bin_mat, filtered_mapping, neuron_clf_type)

            def run_cfg_model(block, block_seed):
                sparse_block = sp.coo_matrix((block > 0).astype(int))
                return configuration_model(sparse_block, seed=block_seed).toarray()

            rand_EE = run_cfg_model(EE, seed if seed is None else seed)
            rand_EI = run_cfg_model(EI, seed if seed is None else seed + 1)
            rand_IE = run_cfg_model(IE, seed if seed is None else seed + 2)
            rand_II = run_cfg_model(II, seed if seed is None else seed + 3)
            return merge_syn_mats_four_blocks(rand_EE, rand_EI, rand_IE, rand_II,
                                              ex_idx, inh_idx, bin_mat.shape[0])
        else:
            sparse_mat = sp.coo_matrix(bin_mat)
            return configuration_model(sparse_mat, seed=seed).toarray()

    else:
        raise ValueError(f"Unknown shuffle_mode '{shuffle_mode}'. "
                         "Only 'cfg' is supported.")


def generate_shuffles(filtered_syn_mat, filtered_mapping, neuron_clf_type,
                      shuffle_mode, amount,
                      shuffle_preserve_EI=True, seed=None):
    """Generate shuffled connectivity matrices in memory (no disk I/O).

    Args:
        filtered_syn_mat  (np.ndarray):  N×N connectivity matrix.
        filtered_mapping  (dict):        {matrix_index: root_id}
        neuron_clf_type   (dict):        {root_id: {'clf_type': 'E' | 'I'}}
        shuffle_mode      (str):         'cfg' (configuration model).
        amount            (int):         Number of shuffled networks to generate.
        shuffle_preserve_EI (bool):      Shuffle within EE/EI/IE/II blocks independently.
        seed              (int):         Base random seed (offset by 10 per iteration).

    Returns:
        List[np.ndarray]: *amount* shuffled binary matrices.
    """
    bin_mat = (filtered_syn_mat > 0).astype(int)
    return [
        _generate_one_shuffle(
            bin_mat, shuffle_mode, neuron_clf_type,
            filtered_mapping, shuffle_preserve_EI,
            seed=seed + i * 10 if seed is not None else None
        )
        for i in tqdm(range(amount), desc=f"Shuffling [{shuffle_mode}]")
    ]


def filter_syn_mat_by_dfs(original_syn_mat, original_mapping, *filtered_dfs):
    """
    Creates a new, smaller syn_mat containing only the neurons present
    in the provided filtered dataframes.

    Args:
        original_syn_mat (np.array): The N x N connectivity matrix.
        original_mapping (dict): {index: root_id} for the original matrix.
        *filtered_dfs (pd.DataFrame): Variable number of DFs (e.g., ex_df, inh_df)
                                      that contain the 'root_id's we want to KEEP.

    Returns:
        new_syn_mat (np.array): The M x M filtered matrix.
        new_mapping (dict): {new_index: root_id} for the new matrix.
    """
    # 1. Collect all valid root_ids from the dataframes into a set for fast lookup
    valid_root_ids = set()
    for df in filtered_dfs:
        valid_root_ids.update(df['root_id'].values)

    # 2. Find the original matrix indices that correspond to these valid root_ids
    # We iterate through the original mapping to preserve the relative order (0..N)
    survivor_indices = []

    # We assume original_mapping keys are 0..N
    for original_idx in range(len(original_mapping)):
        root_id = original_mapping[original_idx]
        if root_id in valid_root_ids:
            survivor_indices.append(original_idx)

    # Convert to numpy array for slicing
    survivor_indices = np.array(survivor_indices)

    if len(survivor_indices) == 0:
        raise ValueError("No neurons survived the filtering! Check your logic.")

    print(f"Filtering: Reducing matrix from {original_syn_mat.shape[0]} to {len(survivor_indices)} neurons.")

    # 3. Create the New Matrix
    # We slice both rows and columns to keep connections between survivors
    new_syn_mat = original_syn_mat[np.ix_(survivor_indices, survivor_indices)]

    # 4. Create the New Mapping
    # The new matrix is indexed 0..M. We need to map 0 -> The first survivor's root_id
    new_mapping = {}
    for new_idx, original_idx in enumerate(survivor_indices):
        root_id = original_mapping[original_idx]
        new_mapping[new_idx] = root_id

    return new_syn_mat, new_mapping
