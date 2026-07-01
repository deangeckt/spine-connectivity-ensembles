import pickle
import os
from connectome_types import EM_NEURONS_PATH
from neuron import Neuron
from meshparty.skeleton import Skeleton

import numpy as np

from utils import transform_sk


def load_col_skeleton_only(neuron_id, reset_axon=False, reset_dentrites=False, old=False):
    with open(os.path.join(EM_NEURONS_PATH, f'{neuron_id}.pkl'), 'rb') as f:
        neuron: Neuron = pickle.load(f)
        sk = neuron.load_skeleton()
        transform_sk(sk, s3=True)

        if reset_axon:
            if old: 
                reset_axon_in_skeleton(sk) # better for plotting, since radii doesn't change
            else:
                sk = filter_skeleton(sk, target_compartments=[1, 3, 4]) # better for measuring length
        if reset_dentrites:
            if old:
                reset_dendrite_in_skeleton(sk)
            else:
                sk = filter_skeleton(sk, target_compartments=[1, 2])
        return sk
    

def load_clean_sk(neuron_id):
    with open(os.path.join(EM_NEURONS_PATH, f'{neuron_id}.pkl'), 'rb') as f:
        neuron: Neuron = pickle.load(f)
        sk = neuron.load_skeleton()
        return sk
    

def filter_skeleton(sk, target_compartments=[2]) -> Skeleton:
    # [2] for axon, [3,4] for dendrites
    compartments = np.array(sk.vertex_properties["compartment"], dtype=int)
    multi_mask = np.isin(compartments, target_compartments)
    return sk.apply_mask(multi_mask)


def filter_sk_bounds(sk, min_x, max_x, min_z, max_z) -> Skeleton:
    mask = (
        (sk.vertices[:, 0] >= min_x) & (sk.vertices[:, 0] <= max_x) &
        (sk.vertices[:, 2] >= min_z) & (sk.vertices[:, 2] <= max_z)
    )
    return sk.apply_mask(mask)


def reset_axon_in_skeleton(sk: Skeleton):
    compartments = np.array(sk.vertex_properties["compartment"], dtype=int)
    axon_indices = np.where(compartments == 2)[0]
    sk.vertices[axon_indices] = sk.root_position


def reset_dendrite_in_skeleton(sk: Skeleton):
    compartments = np.array(sk.vertex_properties["compartment"], dtype=int)
    dendrite_indices = np.where((compartments == 3) | (compartments == 4))[0]
    sk.vertices[dendrite_indices] = sk.root_position

